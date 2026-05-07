import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool
except Exception:
    GCNConv = None
    SAGEConv = None
    global_mean_pool = None

def init_gnn_weights(module):
    """A simple weight initialization helper."""
    for name, param in module.named_parameters():
        if 'weight' in name:
            try:
                nn.init.xavier_uniform_(param)
            except Exception:
                pass
        elif 'bias' in name:
            try:
                nn.init.constant_(param, 0.0)
            except Exception:
                pass

class GCN_Wrapper(nn.Module):
    """
    A simple GCN-based node-classification wrapper.

    Accepts a PyG Batch (batch.x, batch.edge_index, etc.) and returns node-level logits
    with shape (num_nodes, num_classes).
    """
    def __init__(self, in_channels, hidden_dim, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        if GCNConv is None and SAGEConv is None:
            raise RuntimeError("torch_geometric is required for GCN_Wrapper")

        self.convs = nn.ModuleList()
        conv_cls = GCNConv if GCNConv is not None else SAGEConv
        if num_layers <= 1:
            self.convs.append(conv_cls(in_channels, hidden_dim))
        else:
            self.convs.append(conv_cls(in_channels, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(conv_cls(hidden_dim, hidden_dim))
            self.convs.append(conv_cls(hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.attr_mlp = nn.Linear(hidden_dim, out_channels)
        self.init_weights()

    def init_weights(self):
        init_gnn_weights(self)
        try:
            nn.init.xavier_uniform_(self.attr_mlp.weight)
            if self.attr_mlp.bias is not None:
                nn.init.constant_(self.attr_mlp.bias, 0.0)
        except Exception:
            pass

    def forward(self, batch):
        """
        Input: PyG Batch or (x, edge_index) / dict {'x':..., 'edge_index':...}
        Output: node-level logits tensor with shape (num_nodes, out_channels)
        """
        # Handle common wrappers: (batch, lengths) or (x, edge_index)
        # If batch is a sequence, try to find a PyG Data / Batch object inside.
        if isinstance(batch, (tuple, list)):
            # Find an element that has .x and .edge_index
            found = False
            for el in batch:
                if hasattr(el, 'x') and hasattr(el, 'edge_index'):
                    batch = el
                    found = True
                    break
            if not found:
                # Might be (x, edge_index, ...) form
                if len(batch) >= 2 and isinstance(batch[0], torch.Tensor) and isinstance(batch[1], torch.Tensor):
                    x, edge_index = batch[0], batch[1]
                    # Continue to the x/edge_index branch below
                else:
                    # Cannot parse; let the checks below raise an error
                    pass

        # If dict-like input, extract x and edge_index
        if isinstance(batch, dict):
            x = batch.get('x', None)
            edge_index = batch.get('edge_index', None)
            if x is None or edge_index is None:
                raise TypeError("GCN_Wrapper expects dict input to contain keys 'x' and 'edge_index'")
        else:
            # If x and edge_index were extracted above (tuple case), variables exist
            if 'x' in locals() and 'edge_index' in locals():
                pass
            else:
                # Regular PyG Batch object
                if not hasattr(batch, 'x') or not hasattr(batch, 'edge_index'):
                    raise TypeError("GCN_Wrapper expects a PyG Batch with attributes 'x' and 'edge_index'")
                x, edge_index = batch.x, batch.edge_index

        if x is None:
            raise ValueError("input features x is None")

        x = x.to(next(self.parameters()).device).float()
        edge_index = edge_index.to(next(self.parameters()).device)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
        logits = self.attr_mlp(x)
        return logits