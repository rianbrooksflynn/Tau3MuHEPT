import torch
from . import Transformer
from fvcore.nn import FlopCountAnalysis, flop_count_table


def get_model(model_kwargs, dataset=None, test_N=10000, test_k=100):
    model = Transformer(
        in_dim=dataset.x_dim,
        coords_dim=2,
        **model_kwargs,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")
    #count_flops_and_params(model, dataset, test_N, test_k)
    return model

@torch.no_grad()
def count_flops_and_params(model, dataset, N, k):
    E = k * N
    x = torch.randn((N, dataset.x_dim))
    edge_index = torch.randint(0, N, (2, E))
    coords = torch.randn((N, dataset.coords_dim))
    pos = coords[..., :2]
    batch = torch.zeros(N, dtype=torch.long)
    edge_weight = torch.randn((E, 1))

    if dataset.dataset_name == "pileup":
        x[..., -2:] = 0.0

    data = {"x": x, "edge_index": edge_index, "coords": coords, "pos": pos, "batch": batch, "edge_weight": edge_weight}
    print(flop_count_table(FlopCountAnalysis(model, data), max_depth=1))
