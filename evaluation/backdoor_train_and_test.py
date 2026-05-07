import copy
import torch
import numpy as np

from FGMU.client.train_and_test import train_and_test_client, load_client_data
from FGMU.evaluation.backdoor_evaluation import BackdoorEvaluator
from FGMU.utils.model_utils import GCN_Wrapper
from FGMU.utils.dataset import GRU_Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from FGMU.client.GRU_forgetting import GRU_ForgettingController

def _ensure_pyg_list(data):
    """
    Normalize a single torch_geometric.data.Data/Batch or an iterable of Data into list[Data].

    - Supports: a single Data; {'train': Data, 'test': Data}; dict of Data; list/tuple of Data.
    - Raises clear errors for non-iterables (e.g., int) to help debug load_client_data return values.
    """
    # Keep the original logic unchanged
    if hasattr(data, 'x') and hasattr(data, 'edge_index'):
        return [data]

    if isinstance(data, dict):
        if 'train' in data:
            return _ensure_pyg_list(data['train'])
        if 'test' in data:
            return _ensure_pyg_list(data['test'])
        vals = list(data.values())
        datas = []
        for v in vals:
            if hasattr(v, 'x') and hasattr(v, 'edge_index'):
                datas.append(v)
            elif isinstance(v, (list, tuple)):
                for el in v:
                    if hasattr(el, 'x') and hasattr(el, 'edge_index'):
                        datas.append(el)
        if len(datas) > 0:
            return datas
        raise ValueError("No torch_geometric.data.Data found in the provided dict. Please check the structure returned by load_client_data.")
    if isinstance(data, (list, tuple)):
        datas = []
        for el in data:
            if hasattr(el, 'x') and hasattr(el, 'edge_index'):
                datas.append(el)
            elif isinstance(el, dict):
                for v in el.values():
                    if hasattr(v, 'x') and hasattr(v, 'edge_index'):
                        datas.append(v)
                        break
            elif isinstance(el, (list, tuple)):
                for sub in el:
                    if hasattr(sub, 'x') and hasattr(sub, 'edge_index'):
                        datas.append(sub)
        if len(datas) > 0:
            return datas
        raise ValueError("No torch_geometric.data.Data found in the provided list/tuple. Please ensure you pass a list of Data objects.")
    try:
        items = list(data)
    except TypeError:
        raise ValueError(
            f"Cannot convert the input into a PyG Data list: object is not iterable (type={type(data)}). "
            "Please check the return value of load_client_data(client_id)."
        )
    datas = [el for el in items if hasattr(el, 'x') and hasattr(el, 'edge_index')]
    if len(datas) > 0:
        return datas
    raise ValueError("No torch_geometric.data.Data found in the iterable. Please check the data format.")

def add_gaussian_noise(tensor, noise_scale):
    noise = torch.normal(mean=0, std=noise_scale, size=tensor.size()).to(tensor.device)
    return tensor + noise


def _project_model_to_param_ball(model: torch.nn.Module, ref_params, tau: float):
    """Project model parameters so that ||theta - theta0||_2 <= tau.

    ref_params: list of tensors matching model.parameters() order.
    tau <= 0 disables.
    """
    if ref_params is None:
        return
    try:
        tau_val = float(tau)
    except Exception:
        return
    if not (tau_val > 0):
        return

    with torch.no_grad():
        sq = 0.0
        for p, p0 in zip(model.parameters(), ref_params):
            if p is None:
                continue
            d = (p.detach() - p0.to(p.device))
            sq = sq + float((d * d).sum().item())
        cur = float(sq) ** 0.5
        if cur <= tau_val or cur == 0.0:
            return
        scale = tau_val / (cur + 1e-12)
        for p, p0 in zip(model.parameters(), ref_params):
            if p is None:
                continue
            p0d = p0.to(p.device)
            p.copy_(p0d + (p.detach() - p0d) * scale)

def backdoor_aware_train_and_test_client(client_id, raw_data, config, 
                                        backdoor_evaluator=None, is_malicious=False, model=None):
    """
    Single-client backdoor-aware training and evaluation.

    raw_data supports multiple formats but will be normalized by validate_and_extract_raw_data
    into dict {'train', 'test'}.
    """
    print(f"=== Client {client_id} Backdoor-Aware Training (Malicious={is_malicious}) ===")
    # Validate and normalize raw_data
    normalized = validate_and_extract_raw_data(raw_data, client_id)
    original = {'train': normalized['train'], 'test': normalized['test'], 'val_mask':raw_data['val_mask'], 'test_mask':raw_data['test_mask']}

    # If malicious and evaluator is provided, inject backdoor into the training set
    training_train = normalized['train']
    if is_malicious and backdoor_evaluator is not None:
        poisoned = backdoor_evaluator.inject_backdoor_to_client_data(
            {client_id: training_train}, [client_id],
            trigger_type=getattr(config, 'trigger_type', 'pattern'),
            poisoning_rate=getattr(config, 'poisoning_rate', 0.5)
        )
        training_train = poisoned[client_id]
        print(f"[Client {client_id}] Injected backdoor into training set (rate={getattr(config,'poisoning_rate',0.5)})")

    training_data = {'train': training_train, 'test': normalized['test']}

    # Train (skip_forgetting=True returns the pre-unlearning model)
    model = train_and_test_client_with_model_return(client_id, training_data, config, skip_forgetting=True, model=model)

    results = {
        'client_id': client_id,
        'is_malicious': is_malicious,
        'model': model
    }

    # Backdoor evaluation (use original data)
    if backdoor_evaluator is not None:
        bd_res = backdoor_evaluator.evaluate_backdoor_attack(model, training_train, client_id=client_id)
        results['backdoor_results'] = bd_res
        print(f"[Client {client_id}] CleanAcc={bd_res.get('clean_accuracy',0.0):.4f}, ASR={bd_res.get('ASR',0.0):.4f}")
    else:
        results['backdoor_results'] = {'clean_accuracy': 0.0, 'ASR': 0.0}

    return results

def train_and_test_client_with_model_return(client_id, raw_data, config, skip_forgetting=False, model=None, pre_model=None):
    """
    Train a client and return (mape, model).

    raw_data: {'train': Data or iterable of Data, 'test': Data or iterable}
    """
    device = getattr(config, 'device', 'cpu')
    
    # Helper: ensure the input can be used in "for batch in ..."; wrap a single Data/Batch into a list
    def ensure_iterable_for_batch(obj):
        # If it's a single PyG Data/Batch (has x and edge_index), wrap it into a list
        if hasattr(obj, 'x') and hasattr(obj, 'edge_index'):
            return [obj]
        # If it's already a list/tuple, return as-is
        if isinstance(obj, (list, tuple)):
            return obj
        # DataLoader / iterable: return as-is
        if hasattr(obj, '__iter__'):
            return obj
        # Other non-iterables: still wrap into a single-element list to avoid errors
        return [obj]

    # If raw_data is a mapping keyed by client_id (e.g., {id: {'train':..., 'test':...}}), extract the entry first.
    if isinstance(raw_data, dict) and client_id in raw_data:
        print(f"DEBUG: raw_data contains client key {client_id}; extracting raw_data[{client_id}]")
        raw_data = raw_data[client_id]

    # Normalize to list[Data] for PyGDataLoader
    try:
        train_data_list = _ensure_pyg_list(raw_data['train']) if isinstance(raw_data, dict) and 'train' in raw_data else _ensure_pyg_list(raw_data)
        test_data_list  = _ensure_pyg_list(raw_data['test'])  if isinstance(raw_data, dict) and 'test'  in raw_data else _ensure_pyg_list(raw_data)
        retain_data_list = _ensure_pyg_list(raw_data['retain']) if isinstance(raw_data, dict) and 'retain' in raw_data and raw_data['retain'] is not None else None
    except Exception as e:
        # Print detailed debug info to help locate load_client_data return structure issues
        print("ERROR: Failed to construct train/test lists from raw_data. raw_data repr:", repr(raw_data)[:1000])
        raise

    # Further validate that each element in train_data is indeed a PyG Data (duck-typing)
    bad_items = []
    for i, el in enumerate(train_data_list):
        if not (hasattr(el, 'x') and hasattr(el, 'edge_index')):
            bad_items.append((i, type(el), repr(el)[:200]))
    if bad_items:
        print("ERROR: train_data contains non-PyG Data elements. Examples:", bad_items[:5])
        raise TypeError("train_data contains non torch_geometric.data.Data elements. Please check load_client_data returns or preprocessing.")
    
    # Ensure lists
    if not isinstance(train_data_list, list):
        train_data_list = [train_data_list]
    if not isinstance(test_data_list, list):
        test_data_list = [test_data_list]

    # Debug prints to confirm DataLoader args are correct
    # print("DEBUG train_data type:", type(train_data_list), "count:", len(train_data_list))
    # print("DEBUG first element type:", type(train_data_list[0]) if len(train_data_list)>0 else None)

    # train_loader = PyGDataLoader(train_data, batch_size=getattr(config, 'batch_size', 1), shuffle=True, follow_batch=[])
    # test_loader  = PyGDataLoader(test_data,  batch_size=getattr(config, 'batch_size', 1), shuffle=False, follow_batch=[])

    # Infer input/output dims and initialize the model (matches GCN_Wrapper positional args)
    sample = train_data_list[0] if len(train_data_list) > 0 else None
    in_c = getattr(config, 'input_dim', None)
    if in_c is None:
        if sample is not None and hasattr(sample, 'num_node_features'):
            in_c = int(sample.num_node_features)
        elif sample is not None and hasattr(sample, 'x') and sample.x is not None:
            in_c = int(sample.x.size(1))
        else:
            in_c = 1
    out_c = getattr(config, 'output_dim', None)
    if out_c is None:
        if sample is not None and hasattr(sample, 'y') and sample.y is not None:
            try:
                out_c = int(sample.y.max().item()) + 1 if sample.y.numel() > 0 else 1
            except Exception:
                out_c = 1
        else:
            out_c = 1

    # model = GCN_Wrapper(in_c,
    #                     getattr(config, 'hidden_dim', 64),
    #                     out_c,
    #                     num_layers=getattr(config, 'num_layers', 2),
    #                     dropout=getattr(config, 'dropout', 0.5)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=getattr(config, 'lr', 0.01))

    # Initialize forgetting controller (optional)
    forgetting_ctrl = None
    if not skip_forgetting:
        forgetting_ctrl = GRU_ForgettingController(config)

    # Optional: constrain parameter drift during unlearning to avoid catastrophic utility collapse.
    # This implements: ||theta - theta0||_2 <= tau, where theta0 is the starting (pre-unlearn) model.
    unlearn_param_delta_tau = getattr(config, 'unlearn_param_delta_tau', None)
    if unlearn_param_delta_tau is None:
        unlearn_param_delta_tau = getattr(config, 'param_delta_tau', None)
    ref_params = None
    if (not skip_forgetting) and unlearn_param_delta_tau is not None:
        try:
            if float(unlearn_param_delta_tau) > 0:
                ref_params = [p.detach().clone() for p in model.parameters()]
        except Exception:
            ref_params = None

    # If forgetting is enabled, first collect a small number of memory (retain) gradients
    if forgetting_ctrl is not None and not skip_forgetting:
        memory_batches = getattr(config, 'memory_batches', 1)
        collected = 0
        model.zero_grad()

        # memory_grad should come from retain/non-target data; if retain is not provided, fall back to training data (legacy behavior)
        memory_source = retain_data_list if (retain_data_list is not None and len(retain_data_list) > 0) else train_data_list

        # Iterate over memory_source (take the first memory_batches)
        for data in memory_source:
            if collected >= memory_batches:
                break

            data = data.to(device)
            out = model(data)

            # If model returns tuple/list (e.g., (logits, hidden)), use the first element as logits
            if isinstance(out, (tuple, list)):
                logits = out[0]
            else:
                logits = out

            if not torch.is_tensor(logits):
                try:
                    logits = torch.as_tensor(logits)
                except Exception:
                    raise TypeError(f"Model returned non-tensor logits (type: {type(logits)})")

            if logits.dim() >= 2 and logits.size(-1) > 1:
                if hasattr(data, 'train_mask') and data.train_mask is not None:
                    mask = data.train_mask.bool()
                    if mask.sum() == 0:
                        continue
                    loss = torch.nn.CrossEntropyLoss()(logits[mask], data.y[mask].long().to(device))
                else:
                    loss = torch.nn.CrossEntropyLoss()(logits, data.y.long().to(device))
            else:
                if hasattr(data, 'train_mask') and data.train_mask is not None:
                    mask = data.train_mask.bool()
                    if mask.sum() == 0:
                        continue
                    loss = torch.nn.MSELoss()(logits[mask].squeeze(), data.y[mask].float().to(device))
                else:
                    loss = torch.nn.MSELoss()(logits.squeeze(), data.y.float().to(device))

            model.zero_grad()
            loss.backward()
            forgetting_ctrl.store_grads(model, typ="retain")
            model.zero_grad()
            collected += 1

        # Rebuild memory_grad order to match model.named_parameters (for flattening)
        ordered_memory = {}
        for name, param in model.named_parameters():
            if name in forgetting_ctrl.memory_grad:
                ordered_memory[name] = forgetting_ctrl.memory_grad[name].to(param.device)
            else:
                ordered_memory[name] = torch.zeros_like(param).to(param.device)
        forgetting_ctrl.memory_grad = ordered_memory

        # If retain_data_list is provided, rotate sampling retain batches in the unlearning loss
        retain_cursor = 0

        # Training loop (skipped when skip_forgetting=True)
        print(f"Start training: epochs={getattr(config,'epochs',3)}, device={device}")
        for epoch in range(getattr(config, 'epochs', 3)):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            # Iterate over all training graphs
            for data in train_data_list:
                data = data.to(device)

                optimizer.zero_grad()
                out = model(data)

                # If model returns tuple/list (e.g., (logits, hidden)), use the first element as logits
                if isinstance(out, (tuple, list)):
                    logits = out[0]
                else:
                    logits = out

                # Ensure logits is a Tensor (otherwise try to convert or raise)
                if not torch.is_tensor(logits):
                    try:
                        logits = torch.as_tensor(logits)
                    except Exception:
                        raise TypeError(f"Model returned non-tensor logits (type: {type(logits)})")

                if hasattr(data, 'train_mask') and data.train_mask is not None:
                    mask = data.train_mask.bool()
                else:
                    mask = torch.ones(logits.size(0), dtype=torch.bool, device=logits.device)

                if mask.sum() == 0:
                    continue

                if logits.size(-1) > 1:
                    loss = torch.nn.functional.cross_entropy(logits[mask], data.y[mask].long().to(device))
                else:
                    loss = torch.nn.functional.mse_loss(logits[mask].squeeze(), data.y[mask].float().to(device))

                if skip_forgetting:
                    loss.backward()

                # If forgetting is enabled and memory exists, compute unlearn_loss then apply rectified update (avoid extra forwards)
                if forgetting_ctrl is not None and not skip_forgetting and getattr(forgetting_ctrl, 'memory_grad', None):
                    try:
                        forgetting_loss_type = getattr(config, 'forgetting_loss_type', 'npo')
                        forgetting_loss_type_str = str(forgetting_loss_type).strip().lower()

                        retain_for_loss = None
                        if retain_data_list is not None and len(retain_data_list) > 0:
                            retain_for_loss = retain_data_list[retain_cursor % len(retain_data_list)]
                            retain_cursor += 1
                        unlearn_loss, _ = forgetting_ctrl.compute_unlearning_loss(
                            model,
                            forget_source=data,
                            retain_source=retain_for_loss,
                            loss_type=forgetting_loss_type,
                            batches=1,
                            device=device,
                            oracle_model=pre_model,
                            beta=getattr(config, 'beta', 5.0),
                            KL_coef=getattr(config, 'KL_coef', 5.0)
                        )

                        # Optional: align with fixed-pre-threshold Graph-MIA by explicitly pushing the
                        # target CE loss above (fixed_threshold + margin). This is only safe/meaningful
                        # when we are doing gradient *descent* on the unlearning objective (e.g., NPO).
                        try:
                            fixed_thr = getattr(config, 'fixed_mia_loss_threshold', None)
                            fixed_margin = float(getattr(config, 'fixed_mia_target_loss_margin', 0.0))
                            push_lambda = float(getattr(config, 'fixed_mia_push_lambda', 1.0))
                        except Exception:
                            fixed_thr, fixed_margin, push_lambda = None, 0.0, 1.0

                        if (
                            fixed_thr is not None
                            and fixed_margin is not None
                            and float(fixed_margin) > 0
                            and forgetting_loss_type_str.startswith('npo')
                            and logits.dim() >= 2
                            and logits.size(-1) > 1
                        ):
                            # Use the same CE loss notion as MALT (average CE over train_mask).
                            ce_loss = torch.nn.functional.cross_entropy(
                                logits[mask],
                                data.y[mask].long().to(device),
                            )
                            target_loss_floor = torch.as_tensor(
                                float(fixed_thr) + float(fixed_margin),
                                dtype=ce_loss.dtype,
                                device=ce_loss.device,
                            )
                            loss_push = torch.relu(target_loss_floor - ce_loss)
                            if push_lambda is None:
                                push_lambda = 1.0
                            unlearn_loss = unlearn_loss + float(push_lambda) * loss_push
                        forgetting_ctrl.apply_gradient_rectified_unlearning(
                            model,
                            unlearn_loss=unlearn_loss,
                            ascend_scale=getattr(config, 'forget_ascend_scale', 2.0),
                            # NPO objective should be minimized in this repo (use gradient descent);
                            # other loss types keep the legacy "ascent" behavior.
                            ascend=(not forgetting_loss_type_str.startswith('npo'))
                        )
                        optimizer.step()
                        _project_model_to_param_ball(model, ref_params, unlearn_param_delta_tau)
                        epoch_loss += float(unlearn_loss.item()) if isinstance(unlearn_loss, torch.Tensor) else float(unlearn_loss)
                        n_batches += 1
                    except Exception as e:
                        print("Warning: forgetting step failed:", e)

                # Optional DP noise
                if hasattr(config, "dp_noise_scale") and config.dp_noise_scale is not None:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.add_(torch.normal(mean=0.0, std=config.dp_noise_scale, size=p.grad.size(), device=p.grad.device))

            model.eval()
                # optimizer.step()
                # epoch_loss += float(loss.item()) if isinstance(loss, torch.Tensor) else float(loss)
                # n_batches += 1

        print(f"Epoch {epoch+1}/{getattr(config,'epochs',3)} - Loss: {epoch_loss / max(1, n_batches):.6f}")

        # Test evaluation (skipped when skip_forgetting=True)
        model.eval()
        test_loss = 0.0
        test_batches = 0
        correct = 0
        total = 0
        with torch.no_grad():

            # Iterate over all test graphs
            for data in test_data_list:
                data = data.to(device)
                out = model(data)

                # If model returns tuple/list (e.g., (logits, hidden)), use the first element as logits
                if isinstance(out, (tuple, list)):
                    logits = out[0]
                else:
                    logits = out

                # Ensure logits is a Tensor (otherwise try to convert or raise)
                if not torch.is_tensor(logits):
                    try:
                        logits = torch.as_tensor(logits)
                    except Exception:
                        raise TypeError(f"Model returned non-tensor logits (type: {type(logits)})")

                if hasattr(data, 'test_mask') and data.test_mask is not None:
                    mask = data.test_mask.bool()
                else:
                    mask = torch.ones(logits.size(0), dtype=torch.bool, device=logits.device)

                if mask.sum() == 0:
                    continue

                if logits.size(-1) > 1:
                    loss = torch.nn.functional.cross_entropy(logits[mask], data.y[mask].long().to(device))
                    preds = logits[mask].argmax(dim=-1)
                    correct += int((preds == data.y[mask].long().to(device)).sum().item())
                    total += int(data.y[mask].numel())
                else:
                    loss = torch.nn.functional.mse_loss(logits[mask].squeeze(), data.y[mask].float().to(device))
                    total += int(data.y[mask].numel())

                test_loss += float(loss.item())
                test_batches += 1

        mape = test_loss / test_batches if test_batches > 0 else float('inf')
        acc = correct / total if total > 0 and logits.size(-1) > 1 else 0.0
        print(f"Training finished: Test MAPE={mape:.4f}, Test Acc={acc:.4f}")
    else:
        pretrain_epochs = 30
        model.train()
        for epoch in range(pretrain_epochs):
            epoch_loss = 0.0
            for data in train_data_list:
                data = data.to(device)
                optimizer.zero_grad() # 1) clear previous gradients
                
                out = model(data)
                # Handle tuple outputs
                logits = out[0] if isinstance(out, (tuple, list)) else out
                
                # Mask handling
                if hasattr(data, 'train_mask') and data.train_mask is not None:
                    mask = data.train_mask.bool()
                else:
                    mask = torch.ones(logits.size(0), dtype=torch.bool, device=device)
                
                if mask.sum() == 0:
                    continue

                # Loss computation
                if logits.size(-1) > 1:
                    loss = torch.nn.functional.cross_entropy(logits[mask], data.y[mask].long().to(device))
                else:
                    loss = torch.nn.functional.mse_loss(logits[mask].squeeze(), data.y[mask].float().to(device))
                
                loss.backward()   # 2) backprop
                optimizer.step()  # 3) update parameters
                
                epoch_loss += loss.item()
            
                # Print loss to confirm it decreases
            print(f"  Pre-train Epoch {epoch+1} Loss: {epoch_loss}")

            # Run a simple test after training to ensure the model learned something
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_data_list:
            data = data.to(device)
            out = model(data)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            
            if hasattr(data, 'test_mask') and data.test_mask is not None:
                mask = data.test_mask.bool()
            else:
                mask = torch.ones(logits.size(0), dtype=torch.bool, device=device)
                
            if mask.sum() > 0 and logits.size(-1) > 1:
                pred = logits[mask].argmax(dim=1)
                correct += (pred == data.y[mask].to(device)).sum().item()
                total += mask.sum().item()
        
        cur_acc = correct / total if total > 0 else 0.0

    return model


def apply_forgetting_regularization(loss, model, epoch, config):
    """
    Apply forgetting regularization.
    """
    if not hasattr(config, 'forgetting_strength'):
        config.forgetting_strength = 5.0
    
    # Simple L2 regularization as a forgetting mechanism
    l2_reg = torch.tensor(0.).to(config.device)
    for param in model.parameters():
        l2_reg += torch.norm(param)
    
    # Forgetting strength decays with epoch
    forgetting_factor = config.forgetting_strength * (1.0 - epoch / config.epochs)
    
    return loss + forgetting_factor * l2_reg


def federated_unlearning_with_backdoor_evaluation(config, malicious_clients=None):
    """
    Full federated unlearning pipeline with backdoor-attack evaluation.
    
    Args:
        config: configuration
        malicious_clients: list of malicious clients
    
    Returns:
        dict: experiment results
    """
    if malicious_clients is None:
        malicious_clients = getattr(config, 'malicious_clients', [0, 2])
    
    print("=== Federated Unlearning with Backdoor Attack Evaluation ===")
    print(f"Total Clients: {getattr(config, 'num_clients', 10)}")
    print(f"Malicious Clients: {malicious_clients}")
    print(f"Poisoning Rate: {getattr(config, 'poisoning_rate', 0.5)}")
    print(f"Trigger Type: {getattr(config, 'trigger_type', 'pattern')}")
    
    # Initialize backdoor evaluator
    backdoor_evaluator = BackdoorEvaluator(config, config.device)
    
    # Phase 1: Federated training
    print("\n=== Phase 1: Federated Training with Backdoor Injection ===")
    
    client_results = {}
    num_clients = getattr(config, 'num_clients', 10)
    
    for client_id in range(num_clients):
        is_malicious = client_id in malicious_clients
        
        # Load client data
        raw_data = load_client_data(client_id)
        
        # Train client
        result = backdoor_aware_train_and_test_client(
            client_id, raw_data, config, backdoor_evaluator, is_malicious)
        
        client_results[client_id] = result
    
    # Phase 2: Select unlearning target
    print("\n=== Phase 2: Unlearning Target Selection ===")
    
    # Show ASR for all clients
    print("Client ASR Summary:")
    for client_id, result in client_results.items():
        status = "Malicious" if result['is_malicious'] else "Benign"
        asr = result['backdoor_results']['ASR']
        mape = result['mape']
        print(f"  Client {client_id} ({status}): ASR={asr:.4f}, MAPE={mape:.4f}")
    
    # User selects the client to unlearn
    forget_client_id = int(input("\nEnter the client ID to unlearn: "))
    
    if forget_client_id not in client_results:
        print(f"Error: client {forget_client_id} does not exist")
        return client_results
    
    # Phase 3: Execute unlearning
    print(f"\n=== Phase 3: Executing Unlearning for Client {forget_client_id} ===")
    
    # Get pre-unlearning results
    pre_unlearn_results = client_results[forget_client_id]
    
    # Load data for the target client
    forget_data = load_client_data(forget_client_id)
    
    # Execute unlearning (using clean data)
    print("Executing unlearning process...")
    unlearn_mape, unlearn_model = train_and_test_client_with_model_return(
        forget_client_id, forget_data, config)
    
    # Phase 4: Post-unlearning evaluation
    print(f"\n=== Phase 4: Post-Unlearning Evaluation ===")
    
    post_unlearn_backdoor_results = backdoor_evaluator.evaluate_backdoor_attack(
        unlearn_model, forget_data, forget_client_id)
    
    # Phase 5: Effectiveness analysis
    print(f"\n=== Phase 5: Unlearning Effectiveness Analysis ===")
    
    effectiveness = backdoor_evaluator.evaluate_unlearning_effectiveness(
        pre_unlearn_results['backdoor_results'],
        post_unlearn_backdoor_results
    )
    
    # Print detailed results
    print(f"\nUnlearning Results for Client {forget_client_id}:")
    print(f"  Client Type: {'Malicious' if pre_unlearn_results['is_malicious'] else 'Benign'}")
    print(f"  Pre-Unlearning:")
    print(f"    MAPE: {pre_unlearn_results['mape']:.4f}")
    print(f"    ASR: {pre_unlearn_results['backdoor_results']['ASR']:.4f}")
    print(f"    Clean Accuracy: {pre_unlearn_results['backdoor_results']['clean_accuracy']:.4f}")
    print(f"  Post-Unlearning:")
    print(f"    MAPE: {unlearn_mape:.4f}")
    print(f"    ASR: {post_unlearn_backdoor_results['ASR']:.4f}")
    print(f"    Clean Accuracy: {post_unlearn_backdoor_results['clean_accuracy']:.4f}")
    print(f"  Effectiveness Metrics:")
    print(f"    ASR Reduction: {effectiveness['asr_reduction']:.4f}")
    print(f"    Accuracy Retention: {effectiveness['accuracy_retention']:.4f}")
    print(f"    Unlearning Success: {'Yes' if effectiveness['unlearning_success'] else 'No'}")
    
    # Return full results
    unlearn_results = {
        'client_results': client_results,
        'forget_client_id': forget_client_id,
        'pre_unlearn': pre_unlearn_results,
        'post_unlearn': {
            'mape': unlearn_mape,
            'model': unlearn_model,
            'backdoor_results': post_unlearn_backdoor_results
        },
        'effectiveness': effectiveness
    }
    
    return unlearn_results


def evaluate_multiple_clients_unlearning(config, target_clients):
    """
    Evaluate unlearning effectiveness for multiple clients.
    
    Args:
        config: configuration
        target_clients: list of clients to evaluate
    
    Returns:
        dict: multi-client evaluation results
    """
    print(f"=== Evaluating Unlearning for Multiple Clients: {target_clients} ===")
    
    backdoor_evaluator = BackdoorEvaluator(config, config.device)
    all_results = {}
    
    for client_id in target_clients:
        print(f"\n--- Processing Client {client_id} ---")
        
        # Load data
        raw_data = load_client_data(client_id)
        
        # Determine whether this is a malicious client
        malicious_clients = getattr(config, 'malicious_clients', [])
        is_malicious = client_id in malicious_clients
        
        # Train and evaluate
        result = backdoor_aware_train_and_test_client(
            client_id, raw_data, config, backdoor_evaluator, is_malicious)
        
        all_results[client_id] = result
    
    # Summary
    print(f"\n=== Multi-Client Evaluation Summary ===")
    for client_id, result in all_results.items():
        status = "Malicious" if result['is_malicious'] else "Benign"
        print(f"Client {client_id} ({status}):")
        print(f"  MAPE: {result['mape']:.4f}")
        print(f"  ASR: {result['backdoor_results']['ASR']:.4f}")
        print(f"  Clean Accuracy: {result['backdoor_results']['clean_accuracy']:.4f}")
    
    return all_results

def validate_and_extract_raw_data(raw_data, client_id=None):
    """
    Validate raw_data and return {'train': Data_or_list, 'test': Data_or_list}.

    On anomalies, prints debug info and raises a contextual error to help debug
    the return value of load_client_data.
    """
    # Print type summary for debugging
    # print("DEBUG raw_data type:", type(raw_data))
    try:
        # If raw_data is a single PyG Data, treat it as train (and set test to the same)
        if hasattr(raw_data, 'x') and hasattr(raw_data, 'edge_index'):
            # print("DEBUG: raw_data is a single PyG Data; using it for train/test")
            return {'train': raw_data, 'test': raw_data}
        # If it's a dict with train/test, return directly
        if isinstance(raw_data, dict) and 'train' in raw_data and 'test' in raw_data:
            # print("DEBUG: raw_data is a dict containing train/test")
            return {'train': raw_data['train'], 'test': raw_data['test']}
        # If it's a dict with a single Data, extract the first Data as train/test
        if isinstance(raw_data, dict):
            vals = list(raw_data.values())
            # Find the first PyG Data
            for v in vals:
                if hasattr(v, 'x') and hasattr(v, 'edge_index'):
                    print("DEBUG: raw_data is a dict; using the first PyG Data as train/test")
                    return {'train': v, 'test': v}
        # If it's a list/tuple, extract the first element
        if isinstance(raw_data, (list, tuple)):
            if len(raw_data) == 0:
                raise ValueError("raw_data is an empty list/tuple")
            first = raw_data[0]
            if hasattr(first, 'x') and hasattr(first, 'edge_index'):
                print("DEBUG: raw_data is a list/tuple; using the first element as train/test")
                return {'train': first, 'test': first}
        # Otherwise, unrecognized: print detailed repr (truncated) to help debugging
        rep = repr(raw_data)
        rep_short = rep[:1000] + ('...' if len(rep) > 1000 else '')
        raise ValueError(f"Unrecognized raw_data format (client={client_id}), type={type(raw_data)}, content_sample={rep_short}")
    except Exception as e:
        print("DEBUG validate_and_extract_raw_data failed for client", client_id, ":", e)
        raise