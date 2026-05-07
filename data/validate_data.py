import torch
import os
from torch_geometric.data import Data

def validate_client_data(data_dir, num_clients):
    for i in range(num_clients):
        path = os.path.join(data_dir, f"client_{i}.pt")
        try:
            data = torch.load(path)
            assert isinstance(data, Data), "Data must be a PyG Data object"
            assert data.x.dim() == 2, "Node features must be a 2D matrix"
            assert data.edge_index.dim() == 2, "edge_index must be a 2D matrix"
            print(f"✓ Client {i} data validation passed")
        except Exception as e:
            print(f"✗ Client {i} data validation failed: {str(e)}")

validate_client_data("FGMU\\Cora_overlapping", 10)