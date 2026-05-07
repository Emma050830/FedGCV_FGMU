import torch
from torch_geometric.data import Data
import scipy.sparse as sp
import numpy as np

def convert_to_pyg_data(v_adj, v_features, v_metadata):
    """
    Convert VGAE-generated outputs into a PyG Data object.
    """
    # --- 1. Handle features (x) ---
    # Check hasattr(..., 'toarray') to cover all SciPy sparse formats
    if sp.issparse(v_features) or hasattr(v_features, 'toarray'):
        # Convert sparse -> dense NumPy -> Tensor
        # For typical sizes like 289x1433, converting to dense is safe
        np_feat = v_features.toarray()
        x = torch.from_numpy(np_feat).float()
    elif isinstance(v_features, np.ndarray):
        x = torch.from_numpy(v_features).float()
    elif torch.is_tensor(v_features):
        x = v_features.float()
    else:
        # Final fallback
        try:
            x = torch.as_tensor(v_features, dtype=torch.float)
        except Exception:
            # If it is a single scalar
            x = torch.tensor([v_features], dtype=torch.float).view(1, 1)

    # Ensure features are 2D (N, D)
    if x.dim() == 1:
        x = x.unsqueeze(0) if x.size(0) > 1 else x.unsqueeze(1)

    # --- 2. Handle adjacency / edge_index ---
    # If v_adj is a SciPy sparse matrix
    if sp.issparse(v_adj):
        v_adj = v_adj.tocoo()
        row = torch.from_numpy(v_adj.row).to(torch.long)
        col = torch.from_numpy(v_adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
    # If v_adj is already a (2, E) ndarray or tensor
    elif isinstance(v_adj, (np.ndarray, torch.Tensor)) and v_adj.shape[0] == 2:
        edge_index = torch.as_tensor(v_adj, dtype=torch.long)
    else:
        # If it is dense, convert to indices
        adj_tensor = torch.as_tensor(v_adj)
        edge_index = adj_tensor.nonzero().t().contiguous()

    # --- 3. Handle labels (y) ---
    # If v_metadata is a dict, use 'y' if present
    if isinstance(v_metadata, dict) and 'y' in v_metadata:
        y = torch.as_tensor(v_metadata['y'], dtype=torch.long)
    else:
        # If metadata is directly a label array
        try:
            y = torch.as_tensor(v_metadata, dtype=torch.long)
        except Exception:
            # If no labels are provided, use zeros as placeholders
            y = torch.zeros(x.size(0), dtype=torch.long)

    # --- 4. Assemble Data ---
    # You can add train_mask/val_mask/etc. as needed
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # For full-graph training, default to an all-True mask
    num_nodes = x.size(0)
    data.train_mask = torch.ones(num_nodes, dtype=torch.bool)
    
    return data