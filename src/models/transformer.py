import torch
from torch import nn
from torch.nn.functional import tanh
from torch_geometric.nn import MLP
from . import HEPTAttention

from .model_utils.mask_utils import FullMask
from .model_utils.hash_utils import pad_to_multiple, get_bins, quantile_binning
from .model_utils.window_utils import discretize_coords, FlattenedWindowMapping, get_pe_func
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.pool import global_add_pool, global_mean_pool
from torch.utils.checkpoint import checkpoint
from einops import rearrange



def prepare_input(x, coords, batch, helper_funcs):
    kwargs = {}

    #assert batch.max() == 0
    key_padding_mask = None
    mask = None
    kwargs["key_padding_mask"] = key_padding_mask
    kwargs["coords"] = coords


    with torch.no_grad():
        block_size = helper_funcs["block_size"]
        kwargs["raw_size"] = x.shape[0]
        x = pad_to_multiple(x, block_size, dims=0)
        kwargs["coords"] = pad_to_multiple(kwargs["coords"], block_size, dims=0, value=float("inf"))
        sorted_eta_idx = torch.argsort(kwargs["coords"][..., 0], dim=-1)
        sorted_phi_idx = torch.argsort(kwargs["coords"][..., 1], dim=-1)
        bins = helper_funcs["bins"]
        bins_h = rearrange(bins, "c a h -> a (c h)")
        bin_indices_eta = quantile_binning(sorted_eta_idx, bins_h[0][:, None])
        bin_indices_phi = quantile_binning(sorted_phi_idx, bins_h[1][:, None])
        kwargs["bin_indices"] = [bin_indices_eta, bin_indices_phi]
        kwargs["bins_h"] = bins_h
        kwargs["coords"][kwargs["raw_size"]:] = 0.0


    return x, mask, kwargs


class Transformer(nn.Module):
    def __init__(self, in_dim, coords_dim, **kwargs):
        super().__init__()

        self.n_layers = kwargs["n_layers"]
        self.h_dim = kwargs["h_dim"]
        self.use_ckpt = kwargs.get("use_ckpt", False)



        self.feat_encoder = nn.Sequential(
            nn.Linear(in_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
        )

        self.attns = nn.ModuleList()
        for _ in range(self.n_layers):
            self.attns.append(Attn(coords_dim, **kwargs))

        self.dropout = nn.Dropout(0.)
        self.W = nn.Linear(self.h_dim * (self.n_layers + 1), int(self.h_dim // 2), bias=False)
        self.mlp_out = MLP(
            in_channels=int(self.h_dim // 2),
            out_channels=int(self.h_dim // 2),
            hidden_channels=256,
            num_layers=5,
            norm="layer_norm",
            act="tanh",
            norm_kwargs={"mode": "graph"},
        )

        self.helper_funcs = {}

        self.helper_funcs["block_size"] = kwargs["block_size"]
        self.bins = nn.Parameter(get_bins(kwargs["num_buckets"], kwargs["n_hashes"], kwargs["num_heads"]), requires_grad=False)
        self.helper_funcs["bins"] = self.bins

        self.out_proj = nn.Linear(int(self.h_dim // 2), 1)

    def forward(self, data):
        x, coords, batch = data.x, data.coords, data.batch
        #print(data.y)
        #print(x)
        #print(coords)
        #print()
        x, mask, kwargs = prepare_input(x, coords, batch, self.helper_funcs)
        #print(x)
        #print()
        encoded_x = self.feat_encoder(x)
        #print(encoded_x)
        #print()
        all_encoded_x = [encoded_x]
        for i in range(self.n_layers):

            if self.use_ckpt:
                encoded_x = checkpoint(self.attns[i], encoded_x, kwargs)
            else:
                encoded_x = self.attns[i](encoded_x, kwargs)
                
            all_encoded_x.append(encoded_x)

        encoded_x = tanh(self.W(torch.cat(all_encoded_x, dim=-1)))
        #print(encoded_x)
        #print()
        out = encoded_x + self.dropout(self.mlp_out(encoded_x))
        #print(out)
        #print(out-encoded_x)
        #print()
        if kwargs.get("raw_size", False):
            out = out[:kwargs["raw_size"]]

        if mask is not None:
            out = out[mask]
        
        out = global_mean_pool(out, batch)
        #print(out)
        #print()
        out = self.out_proj(out)
        #print(out)
        #print()
        #print(out.sigmoid())
        #q
        return out


class Attn(nn.Module):
    def __init__(self, coords_dim, **kwargs):
        super().__init__()
        self.dim_per_head = kwargs["h_dim"]
        self.num_heads = kwargs["num_heads"]

        self.w_q = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
        self.w_k = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
        self.w_v = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)

        self.attn = HEPTAttention(self.dim_per_head + coords_dim, **kwargs)

        self.dropout = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(self.dim_per_head)
        self.norm2 = nn.LayerNorm(self.dim_per_head)
        self.ff = nn.Sequential(
            nn.Linear(self.dim_per_head, self.dim_per_head),
            nn.ReLU(),
            nn.Linear(self.dim_per_head, self.dim_per_head),
        )

        # eta/phi from data.pos use the same weights as they are used to calc dR
        self.w_rpe = nn.Linear(kwargs["num_w_per_dist"] * (coords_dim - 1), self.num_heads * self.dim_per_head)
        self.pe_func = get_pe_func(kwargs["pe_type"], coords_dim, kwargs)

    def forward(self, x, kwargs):
        pe = kwargs["coords"] if self.pe_func is None else self.pe_func(kwargs["coords"])
 
        x_pe = x + pe if self.pe_func is not None else x
        x_normed = self.norm1(x_pe)
        q, k, v = self.w_q(x_normed), self.w_k(x_normed), self.w_v(x_normed)
        aggr_out = self.attn(q, k, v, pe=pe, w_rpe=self.w_rpe, **kwargs)

        x = x + self.dropout(aggr_out)
        ff_output = self.ff(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x
