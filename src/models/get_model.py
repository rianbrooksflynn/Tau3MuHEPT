import torch

from fvcore.nn import FlopCountAnalysis, flop_count_table


def get_model(model_kwargs, dataset=None, test_N=10000, test_k=100, count_flops=False, max_depth=3):
    
    if model_kwargs['qat'] == True:
        from . import QTransformer as Transformer
    else:
        from . import Transformer
        
    model = Transformer(
        in_dim=dataset.x_dim,
        coords_dim=2,
        **model_kwargs,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")
    if count_flops: count_flops_and_params(model, dataset.x_dim, 2, test_N, test_k, max_depth=max_depth, print_table=True)
    
    return model

@torch.no_grad()
def count_flops_and_params(model, x_dim, coords_dim, N, k, max_depth=3, print_table=False):
    E = k * N
    x = torch.randn((N, x_dim))
    edge_index = torch.randint(0, N, (2, E))
    coords = torch.randn((N, coords_dim))
    pos = coords[..., :2]
    batch = torch.zeros(N, dtype=torch.long)
    edge_weight = torch.randn((E, 1))
    flops = FlopCountAnalysis(model, (x,coords,batch))
    #print(flops.by_module_and_operator())
    
    #data = {"x": x, "edge_index": edge_index, "coords": coords, "pos": pos, "batch": batch, "edge_weight": edge_weight}
    if print_table: print(flop_count_table(flops, max_depth=max_depth))
    
    return flops