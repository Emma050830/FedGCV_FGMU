"""
Backdoor evaluator for PyG `Data` objects.

Includes utilities for injection, evaluation, unlearning-related debugging, and writing back poison indices.
"""
import copy
import torch
import numpy as np
from torch_geometric.loader import DataLoader as PyGDataLoader
from FGMU.utils.dataset import GRU_Dataset

class BackdoorEvaluator:
    """Backdoor attack evaluator (PyG `Data` only)."""
    def __init__(self, args, device='cpu'):
        # args may contain optional configs such as target_label, batch_size, etc.
        self.args = args
        self.device = torch.device(device if isinstance(device, str) else device)
        # Store per-client trigger patterns (CPU tensors) and target labels
        self.trigger_patterns = {}    # client_id -> torch.Tensor (1d)
        self.target_labels = {}       # client_id -> int
        # Record the most recent injection info for debugging and consistent evaluation indices
        self._last_poison_info = {}   # client_id -> dict{poison_indices, pattern_head, target_label}

    # ---------------- Injection ----------------
    def inject_backdoor_to_client_data(self, client_data, malicious_client_ids,
                                       trigger_type="pattern", poisoning_rate=0.5):
        """
        Inject a backdoor into the training data of specified clients.

        Returns a deep-copied data dictionary, and writes poison indices back to
        `new_data.idx_atk_test` (for evaluation consistency).

        client_data: dict {client_id: Data} or a single Data (treated as {0: Data}).
        """
        # Normalize input to a dict (shallow-copy keys, deep-copy values)
        if isinstance(client_data, dict):
            data_map = {k: copy.deepcopy(v) for k, v in client_data.items()}
        else:
            try:
                data_map = {i: copy.deepcopy(d) for i, d in enumerate(client_data)}
            except Exception:
                data_map = {0: copy.deepcopy(client_data)}

        for cid in malicious_client_ids:
            key = cid if cid in data_map else (str(cid) if str(cid) in data_map else cid)
            if key not in data_map:
                raise KeyError(f"Client {cid} not found in client_data. Available keys: {list(data_map.keys())}")
            data = data_map[key]

            # Candidate nodes for injection: prefer train_mask, otherwise use all nodes
            if hasattr(data, 'train_mask') and data.train_mask is not None:
                cand_idx = torch.nonzero(data.train_mask, as_tuple=False).view(-1).cpu().numpy()
            else:
                n = getattr(data, 'num_nodes', None)
                if n is None and hasattr(data, 'x') and data.x is not None:
                    n = data.x.size(0)
                if n is None:
                    raise ValueError(f"Cannot determine number of nodes for client {cid}")
                cand_idx = np.arange(n)

            if len(cand_idx) == 0:
                continue

            poison_size = max(1, int(len(cand_idx) * float(poisoning_rate)))
            poison_indices = np.random.choice(cand_idx, poison_size, replace=False)
            poison_idx_t = torch.from_numpy(poison_indices).long()

            # Feature dimension
            F = data.x.size(1) if (hasattr(data, 'x') and data.x is not None) else 1

            # Generate trigger pattern (1D tensor)
            if trigger_type == "pattern":
                pattern = self._generate_pattern_trigger(F)
            elif trigger_type == "noise":
                pattern = torch.randn(F) * 0.1
            elif trigger_type == "statistical":
                pattern = torch.ones(F) * 0.2
            else:
                pattern = torch.ones(F) * 0.5

            # Store trigger (keep on CPU)
            self.trigger_patterns[cid] = pattern.detach().cpu().clone()

            # Target label (prefer args.target_label; otherwise use max existing y)
            if hasattr(self.args, 'target_label') and self.args.target_label is not None:
                target_label = int(self.args.target_label)
            else:
                try:
                    target_label = int(data.y.max().item())
                except Exception:
                    target_label = 0
            self.target_labels[cid] = target_label

            # Record injection info (for later debugging and evaluation)
            self._last_poison_info[cid] = {
                'poison_indices': poison_indices.tolist() if isinstance(poison_indices, np.ndarray) else [int(poison_indices)],
                'pattern_first': pattern.detach().cpu().numpy().tolist()[:min(10, pattern.numel())],
                'target_label': target_label
            }
            print(f"DEBUG inject: cid={cid} poison_count={len(poison_indices)} sample_idx={self._last_poison_info[cid]['poison_indices'][:8]} target_label={target_label} pattern_head={self._last_poison_info[cid]['pattern_first']}")

            # Inject into data (only modifies training data)
            new_data = copy.deepcopy(data)
            if not hasattr(new_data, 'x') or new_data.x is None:
                new_data.x = torch.zeros((getattr(new_data, 'num_nodes', len(cand_idx)), F), dtype=pattern.dtype)

            # Ensure pattern length == F
            p = pattern.detach().clone()
            if p.numel() != F:
                if p.numel() > F:
                    p = p[-F:]
                else:
                    pad = torch.zeros(F - p.numel(), dtype=p.dtype)
                    p = torch.cat([pad, p], dim=0)

            # Write back features (preserve original device)
            new_data.x = new_data.x.clone()
            new_data.x[poison_idx_t] = p.view(1, -1).expand(len(poison_idx_t), -1).to(new_data.x.device)

            # Modify labels (if y exists)
            if hasattr(new_data, 'y') and new_data.y is not None:
                new_data.y = new_data.y.clone()
                new_data.y[poison_idx_t] = torch.tensor([target_label] * len(poison_idx_t), dtype=new_data.y.dtype, device=new_data.y.device)

            # Write poison indices back to data.idx_atk_test for consistent evaluation
            try:
                new_data.idx_atk_test = torch.from_numpy(poison_indices).long()
            except Exception:
                try:
                    new_data.idx_atk_test = poison_idx_t
                except Exception:
                    pass

            data_map[key] = new_data

        return data_map

    def _generate_pattern_trigger(self, feature_dim: int):
        """Generate a small sine/cosine-based trigger vector (1D tensor)."""
        L = min(10, feature_dim)
        x = torch.linspace(0, 2 * np.pi, L)
        pattern = (torch.sin(x) * 0.5 + torch.cos(2 * x) * 0.3).to(torch.float32)
        if feature_dim > L:
            pad = torch.zeros(feature_dim - L, dtype=pattern.dtype)
            return torch.cat([pad, pattern], dim=0)
        else:
            return pattern[:feature_dim]

    # ---------------- Test-set construction ----------------
    def create_backdoor_test_data(self, clean_test_data, client_id, attack_on='test_mask'):
        """
        Inject the trigger into the test set (without changing labels) and return a new (deep-copied) Data.

        Priority of target nodes: idx_atk_test / poison_indices recorded at injection / test_mask / all.
        """
        if client_id not in self.trigger_patterns:
            return copy.deepcopy(clean_test_data)

        data = copy.deepcopy(clean_test_data)
        pattern = self.trigger_patterns[client_id].to(device=(data.x.device if hasattr(data,'x') and data.x is not None else self.device))

        # Priority sources: (1) data.idx_atk_test (2) injection record self._last_poison_info (3) test_mask (4) all nodes
        target_idx = None
        if hasattr(data, 'idx_atk_test') and data.idx_atk_test is not None:
            atk_idx = data.idx_atk_test
            if isinstance(atk_idx, (list, tuple, np.ndarray)):
                atk_idx = torch.tensor(atk_idx, dtype=torch.long, device=pattern.device)
            else:
                atk_idx = atk_idx.to(pattern.device)
            target_idx = atk_idx.view(-1)
        elif client_id in self._last_poison_info and 'poison_indices' in self._last_poison_info[client_id]:
            pi = self._last_poison_info[client_id]['poison_indices']
            try:
                pi_t = torch.tensor(pi, dtype=torch.long, device=pattern.device)
                n_nodes = getattr(data, 'num_nodes', data.x.size(0) if hasattr(data,'x') and data.x is not None else 0)
                pi_t = pi_t[(pi_t >= 0) & (pi_t < n_nodes)]
                if pi_t.numel() > 0:
                    target_idx = pi_t.view(-1)
            except Exception:
                target_idx = None

        if target_idx is None:
            if attack_on == 'test_mask' and hasattr(data, 'test_mask') and data.test_mask is not None:
                target_idx = torch.nonzero(data.test_mask, as_tuple=False).view(-1).to(pattern.device)
            else:
                target_idx = torch.arange(getattr(data, 'num_nodes', data.x.size(0) if hasattr(data,'x') and data.x is not None else 0), device=pattern.device)

        # Select the feature field to modify (prefer induct_x)
        if hasattr(data, 'induct_x') and data.induct_x is not None:
            feat_field = 'induct_x'
            feat_tensor = data.induct_x
        elif hasattr(data, 'x') and data.x is not None:
            feat_field = 'x'
            feat_tensor = data.x
        else:
            feat_field = 'x'
            feat_tensor = None

        F = feat_tensor.size(1) if (feat_tensor is not None) else pattern.numel()
        p = pattern
        if p.numel() != F:
            if p.numel() > F:
                p = p[-F:]
            else:
                pad = torch.zeros(F - p.numel(), dtype=p.dtype, device=p.device)
                p = torch.cat([pad, p.to(p.device)], dim=0)

        if feat_tensor is None:
            setattr(data, feat_field, torch.zeros((int(getattr(data, 'num_nodes', target_idx.numel())), F), dtype=p.dtype, device=p.device))
            feat_tensor = getattr(data, feat_field)

        # Debug: print before/after comparison (small sample)
        try:
            sample_idx = target_idx[:8].cpu().numpy().tolist()
            before_vals = feat_tensor[sample_idx].detach().cpu().numpy() if feat_tensor is not None else None
            print(f"DEBUG create_backdoor_test_data BEFORE: cid={client_id} feat_field={feat_field} sample_idx={sample_idx} before_head={None if before_vals is None else before_vals[:1]}")
        except Exception:
            sample_idx = None

        new_feat = feat_tensor.clone()
        new_feat[target_idx] = p.view(1, -1).expand(len(target_idx), -1).to(new_feat.device)
        setattr(data, feat_field, new_feat)

        try:
            if sample_idx is not None:
                after_vals = new_feat[sample_idx].detach().cpu().numpy()
                print(f"DEBUG create_backdoor_test_data AFTER: cid={client_id} sample_idx={sample_idx} after_head={after_vals[:1]}")
        except Exception:
            pass

        return data

    # ---------------- Evaluation ----------------
    def evaluate_backdoor_attack(self, model, test_data_dict, client_id):
        """
        Evaluate backdoor effectiveness for node classification:
        - clean_accuracy: accuracy on test_mask (or all nodes)
        - ASR: fraction of attacked nodes predicted as the target label
        - prediction_change: fraction of attacked nodes whose prediction changes

        Returns a dict: {'clean_accuracy', 'ASR', 'prediction_change'}
        """
        model.eval()
        results = {}

        # Input compatibility: accept dict {'test': Data} or a single Data
        test_data = test_data_dict['test'] if isinstance(test_data_dict, dict) and 'test' in test_data_dict else test_data_dict

        clean_correct = 0
        clean_total = 0
        bd_preds_list = []
        clean_preds_list = []

        with torch.no_grad():
            data = test_data

            # Debug: print the most recent injection record (if exists)
            if client_id in self._last_poison_info:
                print("DEBUG evaluate: last_poison_info:", self._last_poison_info[client_id])
            else:
                print("DEBUG evaluate: no recent poison_info for client", client_id)

            out = model(data)
            if isinstance(out, (tuple, list)):
                logits = out[1]
            else:
                logits = out

            if not torch.is_tensor(logits):
                try:
                    logits = torch.as_tensor(logits)
                except Exception:
                    raise TypeError(f"Model returned non-tensor logits (type: {type(logits)})")

            # Debug: basic logits info
            try:
                print("DEBUG evaluate: logits.shape=", None if logits is None else tuple(logits.shape))
            except Exception:
                pass
            try:
                if hasattr(data, 'y') and data.y is not None:
                    print("DEBUG evaluate: y.shape=", tuple(data.y.shape), " y_unique_head=", torch.unique(data.y)[:10].cpu().numpy().tolist())
            except Exception:
                pass

            # Node-level predictions
            if logits.dim() > 1 and logits.size(-1) > 1:
                preds = logits.argmax(dim=-1)
            else:
                preds = logits.squeeze()

            # Select mask: prefer data.test_mask / data.val_mask, then fall back to dict keys
            if hasattr(data, 'test_mask') and data.test_mask is not None:
                mask = data.test_mask.bool()
            elif hasattr(data, 'val_mask') and data.val_mask is not None:
                mask = data.val_mask.bool()
            elif isinstance(test_data_dict, dict) and 'val_mask' in test_data_dict:
                mask = test_data_dict['val_mask']
            else:
                mask = torch.ones(preds.size(0), dtype=torch.bool, device=preds.device)

            # If mask is empty, fall back to all nodes
            try:
                if mask.sum().item() == 0:
                    print("WARNING: mask.sum()==0; falling back to all nodes")
                    mask = torch.ones(preds.size(0), dtype=torch.bool, device=preds.device)
            except Exception:
                pass

            # Clean accuracy (only on nodes indicated by mask)
            if mask.sum().item() > 0:
                y_true = data.y[mask].to(preds.device)
                pred_sel = preds[mask]
                clean_correct += int((pred_sel == y_true).sum().item())
                clean_total += int(y_true.numel())

            clean_preds_list.append((preds.detach().cpu(), mask.detach().cpu(), data.y.detach().cpu()))

            # If a trigger exists, create a triggered copy and evaluate (also handle tuple returns from model)
            if client_id in self.trigger_patterns:
                trig_key = client_id if client_id in self.trigger_patterns else (str(client_id) if str(client_id) in self.trigger_patterns else None)
                if trig_key is not None:
                    bd_batch = self.create_backdoor_test_data(data.cpu(), client_id, attack_on='test_mask').to(self.device)
                    bd_out = model(bd_batch)
                    if isinstance(bd_out, (tuple, list)):
                        bd_logits = bd_out[1]
                    else:
                        bd_logits = bd_out
                    if not torch.is_tensor(bd_logits):
                        bd_logits = torch.as_tensor(bd_logits)
                    if bd_logits.dim() > 1 and bd_logits.size(-1) > 1:
                        bd_preds = bd_logits.argmax(dim=-1)
                    else:
                        bd_preds = bd_logits.squeeze()

                    # For ASR evaluation, prefer idx_atk_test / poison record / mask
                    atk_mask = None
                    if hasattr(data, 'idx_atk_test') and data.idx_atk_test is not None:
                        atk_idx = data.idx_atk_test
                        if isinstance(atk_idx, (list, tuple, np.ndarray)):
                            atk_idx = torch.tensor(atk_idx, dtype=torch.long)
                        atk_mask = torch.zeros(preds.size(0), dtype=torch.bool)
                        try:
                            atk_mask[atk_idx] = True
                        except Exception:
                            # Indices may be out of range; fall back to mask
                            atk_mask = mask
                    elif client_id in self._last_poison_info and 'poison_indices' in self._last_poison_info[client_id]:
                        pi = self._last_poison_info[client_id]['poison_indices']
                        try:
                            pi_t = torch.tensor(pi, dtype=torch.long)
                            pi_t = pi_t[(pi_t >= 0) & (pi_t < preds.size(0))]
                            atk_mask = torch.zeros(preds.size(0), dtype=torch.bool)
                            if pi_t.numel() > 0:
                                atk_mask[pi_t] = True
                            else:
                                atk_mask = mask
                        except Exception:
                            atk_mask = mask
                    else:
                        atk_mask = mask

                    # Debug: print attacked nodes and predictions
                    try:
                        atk_idx_list = torch.nonzero(atk_mask, as_tuple=False).view(-1).cpu().numpy().tolist()
                        print(f"DEBUG evaluate: atk_idx_list (len={len(atk_idx_list)}) sample:{atk_idx_list[:8]}")
                        if len(atk_idx_list) > 0:
                            print(" DEBUG evaluate: y_atk_sample=", data.y[atk_idx_list[:8]].cpu().numpy().tolist())
                            print(" DEBUG evaluate: clean_pred_atk_sample=", preds[atk_idx_list[:8]].cpu().numpy().tolist())
                            print(" DEBUG evaluate: bd_pred_atk_sample=", bd_preds[atk_idx_list[:8]].cpu().numpy().tolist())
                    except Exception:
                        pass

                    bd_preds_list.append((bd_preds.detach().cpu(), atk_mask.detach().cpu()))
                else:
                    print(f"WARNING: No trigger found for client_id {client_id} (trigger_patterns keys: {list(self.trigger_patterns.keys())})")

        # Aggregate clean accuracy
        clean_acc = float(clean_correct / clean_total) if clean_total > 0 else 0.0
        results['clean_accuracy'] = clean_acc

        # Compute ASR and prediction_change (only when a trigger exists)
        if client_id in self.trigger_patterns and len(bd_preds_list) > 0:
            bd_selected_preds = []
            clean_selected_preds = []
            for (bdp, mask) in bd_preds_list:
                bd_selected_preds.append(bdp[mask.numpy()])
            for (cp, mask, y) in clean_preds_list:
                clean_selected_preds.append(cp[mask.numpy()])

            if bd_selected_preds:
                bd_all = torch.cat([t for t in bd_selected_preds], dim=0)
                clean_all = torch.cat([t for t in clean_selected_preds], dim=0) if clean_selected_preds else None
                target_label = int(self.target_labels.get(client_id, 0))
                asr = float((bd_all == target_label).sum().item() / max(1, bd_all.numel()))
                results['ASR'] = asr
                if clean_all is not None and clean_all.numel() == bd_all.numel():
                    pred_change = float((clean_all != bd_all).sum().item() / max(1, bd_all.numel()))
                else:
                    pred_change = 0.0
                results['prediction_change'] = pred_change
            else:
                results['ASR'] = 0.0
                results['prediction_change'] = 0.0
        else:
            results['ASR'] = 0.0
            results['prediction_change'] = 0.0

        return results

    # ---------------- Unlearning effectiveness ----------------
    def evaluate_unlearning_effectiveness(self, original_results, unlearned_results):
        """
        Compute unlearning effectiveness metrics (ASR reduction, accuracy retention, and a simple success flag).
        """
        effectiveness = {}
        asr_reduction = original_results.get('ASR', 0.0) - unlearned_results.get('ASR', 0.0)
        effectiveness['asr_reduction'] = asr_reduction
        acc_retention = unlearned_results.get('clean_accuracy', 0.0) / max(original_results.get('clean_accuracy', 1e-8), 1e-8)
        effectiveness['accuracy_retention'] = acc_retention
        effectiveness['unlearning_success'] = (asr_reduction > 0.1)
        return effectiveness