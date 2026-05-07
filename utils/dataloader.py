import torch
import numpy as np
from torch.utils.data import DataLoader


try:
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch_geometric.data import Batch
except Exception:
    PyGDataLoader = None
    Batch = None

class GRU_DataLoader:
    """
    Graph data loader.

    Expects dataset to be a GRU_Dataset (each item is a PyG Data).
    Returns (batched_data, None) to stay compatible with training loops that unpack (batch, lengths).
    """
    def __init__(self, dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=False):
        if PyGDataLoader is None:
            raise RuntimeError("torch_geometric is required for GRU_DataLoader")
        # PyG DataLoader yields a Batch object
        self.loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    def get_loader(self):
        # wrap iterator so each item yields (batch, None)
        class _Wrapper:
            def __init__(self, loader):
                self.loader = loader
            def __iter__(self):
                for batch in self.loader:
                    yield (batch, None)
            def __len__(self):
                return len(self.loader)
        return _Wrapper(self.loader)