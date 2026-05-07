import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from FGMU.vgae.vgae_virtual_client import train_client_vgae


def main():
    paths = [
        "FGMU/subgraph_data/subgraph_fl_metis_CiteSeer_client_10/data_0.pt",
        "FGMU/subgraph_data/subgraph_fl_metis_Amazon-ratings_client_10/data_0.pt",
    ]
    for p in paths:
        d = torch.load(p, map_location="cpu")
        print(f"Loaded: {p}")
        print("  x:", tuple(d.x.shape))
        m = train_client_vgae(d, num_epochs=1)
        print("  VGAE OK; model:", type(m).__name__)


if __name__ == "__main__":
    main()
