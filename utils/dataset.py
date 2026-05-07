import torch
from torch.utils.data import Dataset

# Try to import the PyG Data type
try:
    from torch_geometric.data import Data as PyGData
except Exception:
    PyGData = None

def _to_pyg_data(item):
    """
    Convert common data representations into torch_geometric.data.Data.

    Supported inputs:
      - Already a PyG Data -> return as-is
      - dict containing 'x' / 'edge_index' / 'y' -> construct Data
      - tuple/list: (x, edge_index) or (x, edge_index, y)
      - torch tensors / numpy arrays are automatically converted to torch.tensor

    Otherwise, return the original object (caller can handle it accordingly).
    """
    # Ensure PyGData is imported correctly
    try:
        from torch_geometric.data import Data as PyGData
    except ImportError:
        PyGData = None
    
    # Already a PyG Data -> return as-is
    if PyGData is not None and isinstance(item, PyGData):
        return item

    # Helper: ensure input is a tensor
    def _ensure_tensor(t):
        if t is None:
            return None
        if isinstance(t, torch.Tensor):
            return t.contiguous()
        try:
            import numpy as np
            if isinstance(t, np.ndarray):
                return torch.from_numpy(t)
            return torch.tensor(t)
        except Exception:
            return None  # Return None on conversion failure (rather than original value)

    # dict -> try extracting common fields
    if isinstance(item, dict):
        x = item.get('x', None)
        edge_index = item.get('edge_index', None)
        y = item.get('y', None)
        
        # If x is missing but features exists, use features
        if x is None and 'features' in item:
            x = item.get('features')
        
        # Convert to tensors
        x = _ensure_tensor(x)
        edge_index = _ensure_tensor(edge_index)
        y = _ensure_tensor(y)
        
        # Ensure at least x or edge_index exists
        if x is not None or edge_index is not None:
            data_kwargs = {}
            if x is not None:
                data_kwargs['x'] = x
            if edge_index is not None:
                data_kwargs['edge_index'] = edge_index
            if y is not None:
                data_kwargs['y'] = y
            
            # Add error handling
            if PyGData is not None:
                try:
                    return PyGData(**data_kwargs)
                except TypeError as e:
                    print(f"[ERROR] Failed to create PyGData from dict: {e}")
                    print(f"[ERROR] data_kwargs keys: {data_kwargs.keys()}")
                    for k, v in data_kwargs.items():
                        print(f"[ERROR]   {k}: type={type(v)}, shape={v.shape if hasattr(v, 'shape') else 'N/A'}")
                    raise
        
        # Cannot construct -> return original object
        return item

    # tuple / list -> (x, edge_index, [y])
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        x = item[0]
        edge_index = item[1]
        y = item[2] if len(item) > 2 else None
        
        # Convert to tensors
        x = _ensure_tensor(x)
        edge_index = _ensure_tensor(edge_index)
        y = _ensure_tensor(y) if y is not None else None
        
        # Ensure conversion succeeded
        if x is None or edge_index is None:
            return item  # Conversion failed; return original object
        
        if PyGData is not None:
            kwargs = {'x': x, 'edge_index': edge_index}
            if y is not None:
                kwargs['y'] = y
            
            # Add error handling
            try:
                return PyGData(**kwargs)
            except TypeError as e:
                print(f"[ERROR] Failed to create PyGData from tuple/list: {e}")
                print(f"[ERROR] x: type={type(x)}, shape={x.shape if hasattr(x, 'shape') else 'N/A'}")
                print(f"[ERROR] edge_index: type={type(edge_index)}, shape={edge_index.shape if hasattr(edge_index, 'shape') else 'N/A'}")
                if y is not None:
                    print(f"[ERROR] y: type={type(y)}, shape={y.shape if hasattr(y, 'shape') else 'N/A'}")
                raise

    # Otherwise, return the original object
    return item


class GRU_Dataset(Dataset):
    """
    Graph dataset wrapper (ensures outputs are torch_geometric.data.Data when possible).

    Usage:
      ds = GRU_Dataset(graphs)
      graphs can be:
        - a single PyG Data
        - dict (client_id -> Data) or dict of Data
        - list/tuple of Data / list of dict / list of (x, edge_index[, y])
        - other iterable objects

    __getitem__ always returns a PyG Data if convertible; otherwise it returns the original element.
    """
def __init__(self, graphs):
    # Ensure scope is correct
    try:
        from torch_geometric.data import Data as PyGData
        self.PyGData = PyGData
    except ImportError:
        raise ImportError("torch_geometric not installed")
    
    if graphs is None:
        raise ValueError("graphs is None")

    self.graphs = []

    # Check whether it is a single PyG Data
    def _is_pyg_like(obj):
        return isinstance(obj, PyGData) or (hasattr(obj, 'x') and hasattr(obj, 'edge_index'))

    # If a single Data-like object is provided, wrap it into a list
    if _is_pyg_like(graphs):
        converted = _to_pyg_data(graphs) if not isinstance(graphs, PyGData) else graphs
        if not isinstance(converted, PyGData):
            raise TypeError(f"Failed to convert to PyGData, got {type(converted)}")
        self.graphs = [converted]
    
    # list/tuple: process element-wise
    elif isinstance(graphs, (list, tuple)):
        for el in graphs:
            if isinstance(el, PyGData):
                self.graphs.append(el)
            else:
                conv = _to_pyg_data(el)
                if isinstance(conv, PyGData):
                    self.graphs.append(conv)
        
        if len(self.graphs) == 0:
            raise ValueError("No valid PyG Data objects found in list")
    
    # dict: extract values
    elif isinstance(graphs, dict):
        for v in graphs.values():
            if isinstance(v, PyGData):
                self.graphs.append(v)
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, PyGData):
                        self.graphs.append(item)
                    else:
                        conv = _to_pyg_data(item)
                        if isinstance(conv, PyGData):
                            self.graphs.append(conv)
            else:
                conv = _to_pyg_data(v)
                if isinstance(conv, PyGData):
                    self.graphs.append(conv)
        
        if len(self.graphs) == 0:
            raise ValueError("No valid PyG Data objects found in dict")
    
    # Other cases: try conversion
    else:
        conv = _to_pyg_data(graphs)
        if isinstance(conv, PyGData):
            self.graphs = [conv]
        else:
            raise TypeError(f"Cannot convert {type(graphs)} to PyG Data")
    
    # Add a batch attribute for all graphs
    for i, graph in enumerate(self.graphs):
        try:
            if not hasattr(graph, 'batch') or graph.batch is None:
                num_nodes = graph.x.size(0) if hasattr(graph, 'x') and graph.x is not None else graph.num_nodes
                graph.batch = torch.zeros(num_nodes, dtype=torch.long)
                print(f"[DEBUG] Added batch to graph {i}, shape: {graph.batch.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to add batch to graph {i}: {e}")
            raise

        # Ensure all tensors are contiguous (required by PyG DataLoader)
        if hasattr(graph, 'x') and graph.x is not None:
            graph.x = graph.x.contiguous()
        if hasattr(graph, 'edge_index') and graph.edge_index is not None:
            graph.edge_index = graph.edge_index.contiguous()
        if hasattr(graph, 'y') and graph.y is not None:
            graph.y = graph.y.contiguous()
        if hasattr(graph, 'batch') and graph.batch is not None:
            graph.batch = graph.batch.contiguous()
        if hasattr(graph, 'edge_index') and graph.edge_index is not None:
            if not graph.edge_index.is_contiguous():
                graph.edge_index = graph.edge_index.contiguous()
                print(f"[DEBUG] Made edge_index contiguous for graph {i}")

        # Ensure test_mask/train_mask/val_mask are also contiguous
        for mask_name in ['test_mask', 'train_mask', 'val_mask']:
            if hasattr(graph, mask_name):
                mask = getattr(graph, mask_name)
                if mask is not None and isinstance(mask, torch.Tensor):
                    setattr(graph, mask_name, mask.contiguous())

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.graphs):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self.graphs)}")
        
        # Return directly; do not call _to_pyg_data again
        return self.graphs[idx]
    
    def get(self, idx):
        """
        PyG may sometimes call get() instead of __getitem__().
        """
        return self.__getitem__(idx)
    
    @staticmethod
    def build_client_graph_dataset(all_clients_graphs, client_id):
        """
        Build a single client's graph dataset from a global container (static method).

        Supports dict or list-like inputs and returns a GRU_Dataset.
        """
        if isinstance(all_clients_graphs, dict):
            key = client_id if client_id in all_clients_graphs else str(client_id)
            if key not in all_clients_graphs:
                raise KeyError(f"client_id {client_id} not found in provided mapping")
            graphs = all_clients_graphs[key]
        elif isinstance(all_clients_graphs, (list, tuple)):
            try:
                graphs = all_clients_graphs[client_id]
            except Exception:
                raise KeyError(f"client_id {client_id} not found in provided list/tuple")
        else:
            graphs = all_clients_graphs
        return GRU_Dataset(graphs)