import os
import torch
import copy
from torch_geometric.loader import DataLoader as PyGDataLoader
from FGMU.client.client import ClientClass
from FGMU.utils.dataset import GRU_Dataset
from FGMU.utils.model_utils import GCN_Wrapper


def _find_distrib_dir(data_root: str, client_idx: int) -> str:
    distrib_root = os.path.join(data_root, "distrib")
    if not os.path.isdir(distrib_root):
        raise FileNotFoundError(f"distrib directory not found: {distrib_root}")

    candidates = [
        os.path.join(distrib_root, d)
        for d in os.listdir(distrib_root)
        if os.path.isdir(os.path.join(distrib_root, d))
    ]
    for cand in sorted(candidates):
        if os.path.isfile(os.path.join(cand, f"data_{client_idx}.pt")):
            return cand

    raise FileNotFoundError(
        f"Could not find data_{client_idx}.pt under {distrib_root}.\n"
        "Make sure federated training has generated distributed data, or set the FGMU_DISTRIB_DIR environment variable to the correct directory."
    )


def load_client_data(client_idx, data_root: str = None, task: str = None, split: str = None):
    """
    Load `train_{idx}.pt` / `val_{idx}.pt` / `test_{idx}.pt` masks (PyTorch tensors)
    together with `data_{idx}.pt` (a PyG Data object) under the given task/split.

    Returns a dict containing train/test Data objects and the corresponding masks.
    """
    # Prefer environment variables to avoid hard-coding platform-specific paths.
    distrib_dir = os.environ.get("FGMU_DISTRIB_DIR")
    if not distrib_dir:
        data_root = data_root or os.environ.get("FGMU_DATA_ROOT")
        if not data_root:
            raise FileNotFoundError(
                "Unable to locate the client data directory. Please either:\n"
                "1) set args.root in main.py and export FGMU_DATA_ROOT/FGMU_DISTRIB_DIR before running; or\n"
                "2) call load_client_data(client_idx, data_root=...) directly"
            )
        distrib_dir = _find_distrib_dir(data_root, client_idx)

    task = task or os.environ.get("FGMU_TASK", "node_cls")
    split = split or os.environ.get("FGMU_SPLIT", "default_split")

    graph_path = os.path.join(distrib_dir, f"data_{client_idx}.pt")
    # PyG Data can be an arbitrary object: set weights_only=False to avoid FutureWarning.
    graph_data = torch.load(graph_path, weights_only=False)

    mask_dir = os.path.join(distrib_dir, task)
    train_path = os.path.join(mask_dir, split, f"train_{client_idx}.pt")
    val_path = os.path.join(mask_dir, split, f"val_{client_idx}.pt")
    test_path = os.path.join(mask_dir, split, f"test_{client_idx}.pt")
    # Masks are tensors: weights_only=True is safer and avoids FutureWarning.
    train_mask = torch.load(train_path, weights_only=True)
    val_mask = torch.load(val_path, weights_only=True)
    test_mask = torch.load(test_path, weights_only=True)
    
    train_data = copy.deepcopy(graph_data)
    test_data = copy.deepcopy(graph_data)

    train_data.train_mask = train_mask
    train_data.val_mask = val_mask
    train_data.test_mask = test_mask

    test_data.train_mask = train_mask
    test_data.val_mask = val_mask
    test_data.test_mask = test_mask

    return {
        'train': train_data,
        'test': test_data,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
    }

def train_and_test_client(client_id, raw_data, config, return_model=False):
    """
    Client-side node classification training/testing (e.g., Cora subgraphs).
    raw_data: dict with 'train' and 'test' as PyG Data objects.
    """
    device = config.device if hasattr(config, 'device') else 'cpu'

    train_data = raw_data['train']
    test_data = raw_data['test']

    # Build dataset/loader (each pt is usually a single Data object wrapped as a dataset).
    train_dataset = GRU_Dataset(train_data)
    test_dataset = GRU_Dataset(test_data)
    train_loader = PyGDataLoader(train_dataset, batch_size=getattr(config, 'batch_size', 1), shuffle=True)
    test_loader = PyGDataLoader(test_dataset, batch_size=getattr(config, 'batch_size', 1), shuffle=False)

    # Infer input dimension and number of classes (prefer config if provided).
    in_channels = getattr(config, 'input_dim', None)
    if in_channels is None:
        # Try inferring from train_data
        if hasattr(train_data, 'num_node_features'):
            in_channels = int(train_data.num_node_features)
        else:
            in_channels = 1

    out_channels = getattr(config, 'output_dim', None)
    if out_channels is None:
        if hasattr(train_data, 'y') and train_data.y is not None:
            out_channels = int(int(train_data.y.max().item()) + 1) if train_data.y.numel() > 0 else 1
        else:
            out_channels = 1

    model = GCN_Wrapper(in_channels, config.hidden_dim, out_channels,
                        num_layers=getattr(config, 'num_layers', 2),
                        dropout=getattr(config, 'dropout', 0.5)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=getattr(config, 'lr', 1e-3))
    criterion = torch.nn.CrossEntropyLoss() if out_channels > 1 else torch.nn.MSELoss()

    print(f"Training for {config.epochs} epochs...")

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)  # (num_nodes_in_batch, C)

            # Select training nodes: prefer batch.train_mask, otherwise use all nodes.
            if hasattr(batch, 'train_mask') and batch.train_mask is not None:
                mask = batch.train_mask.bool()
            else:
                mask = torch.ones((logits.size(0),), dtype=torch.bool, device=device)

            if mask.sum() == 0:
                continue

            if criterion.__class__.__name__ == 'CrossEntropyLoss':
                loss = criterion(logits[mask], batch.y[mask].long().to(device))
            else:
                # MSE: make sure target shape matches logits
                loss = criterion(logits[mask], batch.y[mask].float().to(device).view(-1, logits.size(-1)))

            loss.backward()

            # Optional DP: add Gaussian noise to gradients (if configured).
            if hasattr(config, "dp_noise_scale") and config.dp_noise_scale is not None:
                for p in model.parameters():
                    if p.grad is not None:
                        noise = torch.normal(mean=0.0, std=config.dp_noise_scale, size=p.grad.size(), device=p.grad.device)
                        p.grad.add_(noise)

            optimizer.step()
            epoch_loss += loss.item() if isinstance(loss, torch.Tensor) else float(loss)
            batches += 1

        avg_loss = epoch_loss / max(1, batches)
        print(f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.6f}")

    # Test-time evaluation (node-level)
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    test_batches = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch)
            if hasattr(batch, 'test_mask') and batch.test_mask is not None:
                mask = batch.test_mask.bool()
            else:
                mask = torch.ones((logits.size(0),), dtype=torch.bool, device=device)

            if mask.sum() == 0:
                continue

            if logits.size(-1) > 1:
                preds = logits[mask].argmax(dim=-1)
                labels = batch.y[mask].long().to(device)
                correct += (preds == labels).sum().item()
                total += labels.numel()
                loss = torch.nn.functional.cross_entropy(logits[mask], labels)
            else:
                preds = logits[mask].squeeze()
                labels = batch.y[mask].float().to(device)
                loss = torch.nn.functional.mse_loss(preds, labels)
                # For regression, we do not compute accuracy.
                total += labels.numel()
            test_loss += loss.item()
            test_batches += 1

    acc = correct / total if total > 0 and logits.size(-1) > 1 else 0.0
    avg_test_loss = test_loss / max(1, test_batches)
    print(f"Training completed. Test loss: {avg_test_loss:.4f}, Test acc: {acc:.4f}")

    if return_model:
        return avg_test_loss, model
    else:
        return avg_test_loss