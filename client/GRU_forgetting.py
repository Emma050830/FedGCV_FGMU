import torch
import torch.nn as nn
import numpy as np
import os
from FGMU.utils.dataset import GRU_Dataset

try:
    from torch_geometric.loader import DataLoader as PyGDataLoader
except Exception:
    from torch.utils.data import DataLoader as PyGDataLoader

def add_gaussian_noise(tensor, noise_scale):
    noise = torch.normal(mean=0, std=noise_scale, size=tensor.size()).to(tensor.device)
    return tensor + noise

class GRU_ForgettingController:
    """
    This controller computes gradient statistics on graph mini-batches (sum/mean of absolute gradients)
    to generate parameter-importance scores, and supports gradient orthogonalization against a saved
    `memory_grad` (GRU-like forgetting).
    """
    def __init__(self, config=None):
        self.device = getattr(config, 'device', 'cpu') if config is not None else 'cpu'
        self.batch_size = getattr(config, 'batch_size', 32) if config is not None else 32
        self.num_workers = getattr(config, 'num_workers', 0) if config is not None else 0

        self.gradient_accum = {}
        self.memory_grad = {}
        self.KL_gradient_accum = {}

        # Flattened caches and structure mapping
        self.flattened_gradient = None
        self.flattened_memory_accumulation = None
        self.KL_flattened_gradient = None
        self.structure_map = None

        self.beta = getattr(config, 'npo_beta', 0.1)  # NPO hyperparameter beta
        self.ref_model = None # Reference model (optional)
        # Optional: bound the (projected) update norm to avoid catastrophic utility loss
        self.max_update_norm = getattr(config, 'max_update_norm', None) if config is not None else None
        self.debug = bool(getattr(config, 'debug', False)) if config is not None else False

    # ---------------- Generic loss helpers (PyG subgraphs / masks) ----------------
    def compute_loss_from_outputs(self, outputs, batch, loss_mode='auto', mask_field='train_mask', device=None):
        """
        Compute loss from model outputs and a batch (PyG Data).

        loss_mode: 'auto'|'ce'|'mse'|'neg_ce'|'neg_mse' or a custom callable
        with signature (logits, batch, mask) -> loss.

        - 'auto': use CrossEntropy when logits look like classification (C>1), else use MSE
        - 'neg_*': negate the loss, convenient as an unlearning objective
        """
        device = device or getattr(self, 'device', 'cpu')
        # Handle tuple/list outputs (take the first item as logits)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) and len(outputs) > 0 else outputs
        if not torch.is_tensor(logits):
            logits = torch.as_tensor(logits)

        # Build mask (prefer mask_field if present, otherwise all True)
        if hasattr(batch, mask_field) and getattr(batch, mask_field) is not None:
            mask = getattr(batch, mask_field).bool().to(logits.device)
            if mask.sum() == 0:
                mask = torch.ones(logits.size(0), dtype=torch.bool, device=logits.device)
        else:
            mask = torch.ones(logits.size(0), dtype=torch.bool, device=logits.device)

        # Support custom callable losses
        if callable(loss_mode):
            return loss_mode(logits, batch, mask)

        mode = (loss_mode or 'auto').lower()
        if mode == 'auto':
            mode = 'ce' if (logits.dim() >= 2 and logits.size(-1) > 1) else 'mse'

        # CrossEntropy (classification)
        if mode in ('ce', 'cross_entropy'):
            loss_f = torch.nn.CrossEntropyLoss()
            return loss_f(logits[mask], batch.y[mask].long().to(logits.device))
        # MSE (regression)
        if mode in ('mse', 'l2'):
            loss_f = torch.nn.MSELoss()
            return loss_f(logits[mask].squeeze(), batch.y[mask].float().to(logits.device))
        # Negative variants (for unlearning)
        if mode in ('neg_ce', 'neg_cross_entropy'):
            loss_f = torch.nn.CrossEntropyLoss()
            return - loss_f(logits[mask], batch.y[mask].long().to(logits.device))
        if mode in ('neg_mse',):
            loss_f = torch.nn.MSELoss()
            return - loss_f(logits[mask].squeeze(), batch.y[mask].float().to(logits.device))

        # Fallback
        if logits.dim() >= 2 and logits.size(-1) > 1:
            return torch.nn.CrossEntropyLoss()(logits[mask], batch.y[mask].long().to(logits.device))
        else:
            return torch.nn.MSELoss()(logits[mask].squeeze(), batch.y[mask].float().to(logits.device))
    # ---------------- End helpers ----------------

    def compute_forgetting_scores(self, model, client_data, criterion=None, max_batches=None):
        """
        Compute parameter-importance scores based on the mean absolute gradients on `client_data`.
        """
        model_device = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else torch.device(self.device)
        model.to(model_device)
        dataset = GRU_Dataset(client_data)
        loader = PyGDataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        # Initialize per-parameter sum of abs(grad)
        per_param_sum = {}
        seen_batches = 0

        # Default criterion
        sample = None
        for item in dataset.graphs:
            sample = item
            break
        if criterion is None:
            if sample is not None and hasattr(sample, 'y') and sample.y is not None:
                # Infer multi-class from the y value range
                try:
                    out_dim = int(sample.y.max().item()) + 1 if sample.y.numel() > 0 else 1
                except Exception:
                    out_dim = 1
                # Output shape is not known here; pick at runtime
                criterion_default = None
            else:
                criterion_default = None
        else:
            criterion_default = criterion

        model.eval()
        with torch.no_grad():
            # (Reserved) Forward-only path if shapes are needed; gradients require train() + backward()
            pass

        # Enable grads to collect gradients
        model.train()
        for batch in loader:
            if max_batches is not None and seen_batches >= max_batches:
                break
            # Handle PyG Batch / tuple wrappers
            if isinstance(batch, (tuple, list)) and len(batch) > 0:
                batch = batch[0]
            batch = batch.to(model_device)

            # Infer loss (if caller did not provide one)
            model_output = model(batch)
            outputs = model_output[1] if isinstance(model_output, (tuple, list)) and len(model_output) > 0 else model_output
            if callable(criterion_default):
                loss = criterion_default(outputs, batch)
            else:
                # criterion_default can be None or a string mode like 'ce'/'mse'
                loss_mode = criterion_default if isinstance(criterion_default, str) else 'auto'
                loss = self.compute_loss_from_outputs(outputs, batch, loss_mode=loss_mode, mask_field='train_mask', device=model_device)

            # Clear grads then backprop
            model.zero_grad()
            loss.backward()

            # Accumulate abs-grad for each parameter
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                g = param.grad.detach().abs().cpu()
                if name not in per_param_sum:
                    per_param_sum[name] = torch.zeros_like(g)
                per_param_sum[name] += g

            seen_batches += 1

        # If no gradients were collected, return an all-ones vector
        if len(per_param_sum) == 0:
            # Build structure map and return all ones
            structure_map = [(name, tuple(param.shape)) for name, param in model.named_parameters()]
            parts = []
            for _, shape in structure_map:
                parts.append(torch.ones(int(np.prod(shape)), dtype=torch.float32))
            if parts:
                return torch.cat(parts).to(torch.float32)
            else:
                return torch.tensor([], dtype=torch.float32)

        # Mean abs-grad per parameter (divide by seen_batches)
        for name in per_param_sum:
            per_param_sum[name] = (per_param_sum[name] / max(1, seen_batches)).view(-1)

        # Flatten (follow model.named_parameters order)
        parts = []
        for name, param in model.named_parameters():
            if name in per_param_sum:
                parts.append(per_param_sum[name])
            else:
                parts.append(torch.zeros(int(param.numel()), dtype=torch.float32))
        flattened = torch.cat(parts).to(torch.float32)
        # Normalize by max to avoid large values
        maxv = flattened.max().item() if flattened.numel() > 0 else 1.0
        if maxv > 0:
            flattened = flattened / float(maxv)
        return flattened

    def collect_memory_from_data(self, model, client_data, batches=1, criterion=None):
        """
        Collect memory_grad (retain gradients) on training data for later orthogonalization.

        Usage: call collect_memory_from_data(model, data, batches=N) before unlearning.
        This accumulates gradients into self.memory_grad (keyed by parameter name).
        """
        model_device = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else torch.device(self.device)
        dataset = GRU_Dataset(client_data)
        loader = PyGDataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        model.train()
        collected = 0
        for batch in loader:
            if collected >= batches:
                break
            if isinstance(batch, (tuple, list)) and len(batch) > 0:
                batch = batch[0]
            batch = batch.to(model_device)

            model_output = model(batch)
            outputs = model_output[1] if isinstance(model_output, (tuple, list)) and len(model_output) > 0 else model_output
            if callable(criterion):
                loss = criterion(outputs, batch)
            else:
                loss_mode = criterion if isinstance(criterion, str) else 'auto'
                loss = self.compute_loss_from_outputs(outputs, batch, loss_mode=loss_mode, mask_field='train_mask', device=model_device)

            model.zero_grad()
            loss.backward()
            # Accumulate into memory_grad
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                g = param.grad.detach().cpu()
                if name not in self.memory_grad:
                    self.memory_grad[name] = torch.zeros_like(g)
                self.memory_grad[name] += g
            collected += 1

        # Flatten for later use
        self.flatten_and_store_grads(model)

    def store_grads(self, model, loss=None, typ=None):
        """
        Store current gradients into self.gradient_accum / self.memory_grad / self.KL_gradient_accum.

        typ: "objective", "retain", "KL_objective"
        If `loss` is provided, this method performs backward() internally and clears model grads.
        """
        if loss is not None:
            model.zero_grad()
            loss.backward()

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            g = param.grad.detach().cpu() if param.grad is not None else torch.zeros_like(param.detach().cpu())
            if typ == "objective":
                target_dict = self.gradient_accum
            elif typ == "retain":
                target_dict = self.memory_grad
            elif typ == "KL_objective":
                target_dict = self.KL_gradient_accum
            else:
                raise ValueError("Invalid type specified for gradient storage")
            if name not in target_dict:
                target_dict[name] = torch.zeros_like(g)
            target_dict[name] += g

        if loss is not None:
            model.zero_grad()

    def flatten_and_store_grads(self, model):
        """
        Flatten self.gradient_accum / self.memory_grad / self.KL_gradient_accum following model.named_parameters()
        and record the parameter structure.
        """
        def _flatten_from_dict(grad_dict):
            parts = []
            for name, param in model.named_parameters():
                g = grad_dict.get(name, None)
                if g is None:
                    parts.append(torch.zeros(int(np.prod(param.shape)), dtype=torch.float32))
                else:
                    parts.append(g.detach().cpu().view(-1).to(torch.float32))
            if parts:
                return torch.cat(parts).to(torch.float32)
            else:
                return torch.tensor([], dtype=torch.float32)

        self.structure_map = [(name, tuple(param.shape)) for name, param in model.named_parameters()]
        self.flattened_gradient = _flatten_from_dict(self.gradient_accum)
        self.flattened_memory_accumulation = _flatten_from_dict(self.memory_grad)
        self.KL_flattened_gradient = _flatten_from_dict(self.KL_gradient_accum)

    def compute_total_gradient_dot_product(self, flattened_grads1, structure_map1, flattened_grads2, structure_map2):
        """Compute the total dot product of two flattened gradients (for correlation analysis)."""
        total_dot_product = 0.0
        index = 0
        flattened_grads1 = flattened_grads1.to('cpu')
        flattened_grads2 = flattened_grads2.to('cpu')
        for ((name1, shape1), (name2, shape2)) in zip(structure_map1, structure_map2):
            assert shape1 == shape2, f"Gradient shape mismatch: {name1} vs {name2} or {shape1} vs {shape2}"
            size = np.prod(shape1)
            grad_slice1 = flattened_grads1[index:index + size].view(shape1)
            grad_slice2 = flattened_grads2[index:index + size].view(shape2)
            dot_product = (grad_slice1 * grad_slice2).sum()
            total_dot_product += dot_product.item()
            index += size
        return total_dot_product

    def restore_gradients_from_flat(self, model):
        """
        Restore self.flattened_gradient into model parameter grads (assign to param.grad).
        Assumes self.structure_map matches the order of model.named_parameters().
        """
        if self.flattened_gradient is None:
            return
        index = 0
        for name, shape in self.structure_map:
            size = int(np.prod(shape))
            grad_slice = self.flattened_gradient[index:index + size].view(shape).to(torch.float32)
            param = next((p for n, p in model.named_parameters() if n == name), None)
            if param is None:
                index += size
                continue
            if not hasattr(param, 'grad') or param.grad is None:
                param.grad = grad_slice.to(param.device)
            else:
                param.grad = grad_slice.to(param.device)
            index += size
        # Safety check
        if index != self.flattened_gradient.numel():
            # Do not raise; only warn to avoid interrupting training
            print("Warning: flattened gradient length mismatch in restore_gradients_from_flat")

    def orthogonal_component_precise(self, g, g1):
        """
        Compute the orthogonal component of g w.r.t. g1 (CPU numpy implementation, returns float32 tensor).
        Inputs are 1D torch tensors (CPU).
        """
        if not isinstance(g, torch.Tensor):
            g = torch.tensor(g, dtype=torch.float32)
        if not isinstance(g1, torch.Tensor):
            g1 = torch.tensor(g1, dtype=torch.float32)
        g_np = g.detach().cpu().numpy().astype(np.float64)
        g1_np = g1.detach().cpu().numpy().astype(np.float64)
        if g_np.size != g1_np.size:
            min_len = min(g_np.size, g1_np.size)
            g_np = g_np[:min_len]
            g1_np = g1_np[:min_len]
        denom = np.dot(g1_np, g1_np)
        if denom == 0:
            ortho = g_np
        else:
            projection = (np.dot(g_np, g1_np) / denom) * g1_np
            ortho = g_np - projection
        return torch.from_numpy(ortho.astype(np.float32))

    def compute_unlearning_loss(self, model, forget_source, retain_source=None,
                                loss_type='npo', batches=None, device=None, oracle_model=None, beta=5.0, KL_coef=5.0):
        """
        Compute the unlearning loss.
        """
        device = device or getattr(model, 'device', self.device)
        model_device = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else torch.device(device)
        model.to(model_device)

        def _get_first_batch_from(source):
            # Accept Batch/Data or GRU_Dataset/iterables
            if source is None:
                return None
            if isinstance(source, (list, tuple)) and len(source) > 0:
                src = source[0]
                if hasattr(src, 'x'):
                    return src
            # If it's PyG Batch/Data
            if hasattr(source, 'x') and hasattr(source, 'edge_index'):
                return source
            # Otherwise treat as dataset-like
            ds = GRU_Dataset(source)
            loader = PyGDataLoader(ds, batch_size=1, shuffle=True, num_workers=self.num_workers)
            for b in loader:
                if isinstance(b, (tuple, list)) and len(b) > 0:
                    return b[0]
                return b
            return None

        forget_batch = _get_first_batch_from(forget_source)
        retain_batch = _get_first_batch_from(retain_source)

        if forget_batch is None:
            raise ValueError("Unable to build a batch for the forget set.")

        forget_batch = forget_batch.to(model_device)
        model.eval()
        # loss_type can be 'neg_ce'/'neg_mse'; for classification default to neg_ce
        try:
            if isinstance(loss_type, str) and ('neg' in loss_type or 'ga' in loss_type or 'gru' in loss_type):
                map_mode = 'neg_ce'
            else:
                map_mode = 'auto'
            model_output = model(forget_batch)
            # Handle (logits, hidden) / (logits,) outputs: always take the first as logits
            outputs = model_output[0] if isinstance(model_output, (tuple, list)) and len(model_output) > 0 else model_output
            forget_loss = self.compute_loss_from_outputs(outputs, forget_batch, loss_mode=map_mode, mask_field='train_mask', device=model_device)
        except Exception as e:
            # Fallback to the legacy logic for compatibility
            print("Debug: compute_unlearning_loss helper failed, falling back:", e)
            model_output = model(forget_batch)
            outputs = model_output[0] if isinstance(model_output, (tuple, list)) and len(model_output) > 0 else model_output
            if outputs.dim() >= 2 and outputs.size(-1) > 1:
                ce = torch.nn.CrossEntropyLoss()
                mask = forget_batch.train_mask.bool() if hasattr(forget_batch, 'train_mask') and forget_batch.train_mask is not None else torch.ones(outputs.size(0), dtype=torch.bool, device=model_device)
                if mask.sum() == 0:
                    mask = torch.ones(outputs.size(0), dtype=torch.bool, device=model_device)
                forget_loss = - ce(outputs[mask], forget_batch.y[mask].long().to(model_device))
            else:
                mse = torch.nn.MSELoss()
                mask = forget_batch.train_mask.bool() if hasattr(forget_batch, 'train_mask') and forget_batch.train_mask is not None else torch.ones(outputs.size(0), dtype=torch.bool, device=model_device)
                forget_loss = - mse(outputs[mask].squeeze(), forget_batch.y[mask].float().to(model_device))

        retain_loss = None
        # Compute retain loss when retain_batch is provided
        # if 'gru' in loss_type or 'retain' in loss_type or ('KL' in loss_type and retain_batch is not None):
        if retain_batch is not None:
            retain_batch = retain_batch.to(model_device)
            model.eval()
            r_model_out = model(retain_batch)
            r_out = r_model_out[0] if isinstance(r_model_out, (tuple, list)) and len(r_model_out) > 0 else r_model_out
            if r_out.dim() >= 2 and r_out.size(-1) > 1:
                retain_loss_f = torch.nn.CrossEntropyLoss()
                if hasattr(retain_batch, 'train_mask') and retain_batch.train_mask is not None:
                    rmask = retain_batch.train_mask.bool()
                    if rmask.sum() == 0:
                        retain_loss = torch.tensor(0.0, device=model_device)
                    else:
                        retain_loss = retain_loss_f(r_out[rmask], retain_batch.y[rmask].long().to(model_device))
                else:
                    retain_loss = retain_loss_f(r_out, retain_batch.y.long().to(model_device))
            else:
                retain_loss_f = torch.nn.MSELoss()
                if hasattr(retain_batch, 'train_mask') and retain_batch.train_mask is not None:
                    rmask = retain_batch.train_mask.bool()
                    if rmask.sum() == 0:
                        retain_loss = torch.tensor(0.0, device=model_device)
                    else:
                        retain_loss = retain_loss_f(r_out[rmask].squeeze(), retain_batch.y[rmask].float().to(model_device))
                else:
                    retain_loss = retain_loss_f(r_out.squeeze(), retain_batch.y.float().to(model_device))
        else:
            retain_loss = torch.tensor(0.0, device=model_device)

        # NPO
        # Normalize non-string inputs and ignore case/whitespace
        loss_type_str = str(loss_type).strip().lower() if loss_type is not None else ""
        if self.debug or os.environ.get('FGMU_DEBUG_LOSS', '').strip() == '1':
            print(f"DEBUG compute_unlearning_loss: loss_type={loss_type!r}, normalized={loss_type_str!r}, type={type(loss_type)}")
        if loss_type_str.startswith('npo'):
            if oracle_model is None:
                unlearn = forget_loss 
            else:
                oracle_model.eval() 
                with torch.no_grad():
                    oracle_model_out = oracle_model(forget_batch)
                    oracle_out = oracle_model_out[0] if isinstance(oracle_model_out, (tuple, list)) and len(oracle_model_out) > 0 else oracle_model_out
                
                if oracle_out.dim() >= 2 and oracle_out.size(-1) > 1:
                    # Graph-MIA/MALT and this repo's training both use train_mask; align NPO to the same mask
                    if hasattr(forget_batch, 'train_mask') and forget_batch.train_mask is not None:
                        npo_mask = forget_batch.train_mask.bool().to(model_device)
                        if npo_mask.sum() == 0:
                            npo_mask = torch.ones(forget_batch.y.size(0), dtype=torch.bool, device=model_device)
                    else:
                        npo_mask = torch.ones(forget_batch.y.size(0), dtype=torch.bool, device=model_device)

                    # Per-node CE for current/Oracle (masked)
                    curr_loss_vals = torch.nn.functional.cross_entropy(
                        outputs[npo_mask],
                        forget_batch.y[npo_mask].long().to(model_device),
                        reduction='none'
                    )
                    oracle_loss_vals = torch.nn.functional.cross_entropy(
                        oracle_out[npo_mask],
                        forget_batch.y[npo_mask].long().to(model_device),
                        reduction='none'
                    )

                    # Difference
                    neg_log_ratios = curr_loss_vals - oracle_loss_vals
                    
                    # Scale by 2 / beta
                    unlearn = -torch.nn.functional.logsigmoid(beta * neg_log_ratios).mean() * (2 / beta)
                else:
                    unlearn = forget_loss

            # Add retain loss
            if retain_loss is not None:
                forget_loss = unlearn + retain_loss  # total unlearning objective
            else:
                forget_loss = unlearn

        # Return tensors
        return forget_loss, retain_loss

    def _flatten_current_grads(self, model):
        parts = []
        for name, param in model.named_parameters():
            g = param.grad.detach().cpu() if param.grad is not None else torch.zeros(int(np.prod(param.shape)), dtype=torch.float32)
            parts.append(g.view(-1).to(torch.float32))
        if parts:
            return torch.cat(parts).to(torch.float32)
        else:
            return torch.tensor([], dtype=torch.float32)

    def _restore_flat_to_model(self, flat_vec, model):
        """
        Restore flat_vec (1D CPU tensor) into model parameter grads (write to param.grad).
        Requires self.structure_map to match model.named_parameters order (or rebuild it).
        """
        # Build structure_map if missing
        if self.structure_map is None:
            self.structure_map = [(name, tuple(param.shape)) for name, param in model.named_parameters()]

        index = 0
        for name, shape in self.structure_map:
            size = int(np.prod(shape))
            slice_vec = flat_vec[index:index + size].view(shape).to(torch.float32)
            param = next((p for n, p in model.named_parameters() if n == name), None)
            if param is not None and param.requires_grad:
                param.grad = slice_vec.to(param.device)
            index += size
        if index != flat_vec.numel():
            # Non-fatal; print as debug information
            print("Warning: length mismatch when restoring flattened vector to model")

    def apply_gradient_rectified_unlearning(self, model, unlearn_loss=None, ascend_scale=3.0, eps=1e-12, ascend: bool = True):
        """
        Given an unlearn_loss, compute its gradients w.r.t. the model, flatten them, project
        onto the subspace orthogonal to memory_grad (removing the memory projection), and then
        choose an update direction based on `ascend`:

        - ascend=True: gradient ascent on unlearn_loss (legacy behavior)
        - ascend=False: gradient descent on unlearn_loss (useful when NPO is defined as minimization)

        The resulting vector is written back into model.grad for optimizer.step().
        """
        # Ensure flattened memory exists
        if self.flattened_memory_accumulation is None or self.flattened_memory_accumulation.numel() == 0:
            # Try building from the memory_grad dict
            if len(self.memory_grad) > 0:
                self.flatten_and_store_grads(model)

        if unlearn_loss is not None:
            # Compute gradients of unlearn_loss
            model.zero_grad()
            unlearn_loss.backward()
            flat_un = self._flatten_current_grads(model)  # CPU
            direction = -1.0 if ascend else 1.0
            # Ensure memory exists
            mem = self.flattened_memory_accumulation if (self.flattened_memory_accumulation is not None and self.flattened_memory_accumulation.numel() > 0) else None
            if mem is None or mem.numel() == 0:
                # No memory -> apply scaled gradient directly
                flat_to_apply = (direction * flat_un) * ascend_scale
                self.flattened_gradient = flat_to_apply
                self._restore_flat_to_model(flat_to_apply, model)
                return
            # Align lengths
            min_len = min(flat_un.numel(), mem.numel())
            fu = flat_un[:min_len]
            fm = mem[:min_len]
            denom = (fm * fm).sum().item()
            if denom <= 0:
                proj = torch.zeros_like(fu)
            else:
                coeff = (fu * fm).sum().item() / (denom + eps)
                proj = coeff * fm
            ortho = fu - proj
            # Pad remaining tail if needed
            if flat_un.numel() > min_len:
                tail = flat_un[min_len:]
                full = torch.cat([ortho, tail])
            else:
                full = ortho
            flat_to_apply = (direction * full) * ascend_scale
            # --- Bound update norm if max_update_norm is set
            try:
                tau = getattr(self, 'max_update_norm', None)
                if tau is not None and tau > 0:
                    cur_norm = float(flat_to_apply.norm().cpu().item())
                    if cur_norm > tau:
                        scale = float(tau / (cur_norm + 1e-12))
                        flat_to_apply = flat_to_apply * scale
                        print(f"Debug: scaled flat_to_apply by {scale:.4f} to satisfy max_update_norm={tau}")
            except Exception:
                pass
            self.flattened_gradient = flat_to_apply.clone()
            self._restore_flat_to_model(flat_to_apply, model)
            return

        # Fallback: orthogonalize objective gradients against memory
        self.store_grads(model, typ="objective")
        self.flatten_and_store_grads(model)
        if self.flattened_memory_accumulation is None or self.flattened_memory_accumulation.numel() == 0:
            if self.flattened_gradient is not None and self.flattened_gradient.numel() > 0:
                self.restore_gradients_from_flat(model)
            return
        ortho_flat = self.orthogonal_component_precise(self.flattened_gradient, self.flattened_memory_accumulation)
        self.flattened_gradient = ortho_flat.to(torch.float32)
        self.restore_gradients_from_flat(model)
        print(f"DEBUG: original grad norm: {self.flattened_gradient.norm()}, rectified grad norm: {ortho_flat.norm()}")

    def apply_dp_to_gradients(self, model, noise_scale=1e-2):
        """
        Add Gaussian noise to model.grad (in-place).
        """
        for name, param in model.named_parameters():
            if param.grad is not None:
                param.grad = add_gaussian_noise(param.grad, noise_scale)
