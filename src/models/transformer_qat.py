import torch
from torch import nn
from torch.nn import MultiheadAttention
from torch.nn.functional import tanh
from torch_geometric.nn.norm import LayerNorm
from . import QHEPTAttention as HEPTAttention

from .model_utils.mask_utils import FullMask
from .model_utils.hash_utils import get_regions, quantile_partition
from .model_utils.window_utils import discretize_coords, FlattenedWindowMapping, get_pe_func
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.pool import global_add_pool, global_mean_pool
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from brevitas.quant import Int32Bias
import brevitas.nn as qnn

def bit_shift(base, shift_idx):
    max_base = base.max(dim=1, keepdim=True).values
    num_bits = torch.ceil(torch.log2(max_base + 1)).long()
    return (shift_idx << num_bits) | base


def pad_and_unpad(batch, block_size, region_indices, raw_sizes):
    padded_sizes = ((raw_sizes + block_size - 1) // block_size) * block_size
    pad_sizes = padded_sizes - raw_sizes

    pad_cumsum = padded_sizes.cumsum(0)
    pad_seq = torch.arange(pad_cumsum[-1], device=batch.device)
    unpad_seq = torch.ones(pad_cumsum[-1], device=batch.device).bool()

    sorted_region_indices = region_indices.argsort()
    for i in range(len(raw_sizes)):
        idx_to_fill = pad_cumsum[i] - block_size - pad_sizes[i] + torch.arange(pad_sizes[i], device=batch.device)
        if i >= 1:
            pad_seq[pad_cumsum[i - 1] :] -= pad_sizes[i - 1]
            idx_to_fill -= pad_sizes[:i].sum()
        pad_seq[pad_cumsum[i] - pad_sizes[i] : pad_cumsum[i]] = sorted_region_indices[idx_to_fill]
        unpad_seq[pad_cumsum[i] - pad_sizes[i] : pad_cumsum[i]] = False
    return pad_seq, unpad_seq


def prepare_input(x, coords, batch, helper_params):
    kwargs = {}
    regions = rearrange(helper_params["regions"], "c a h -> a (c h)")
    with torch.no_grad():
        block_size, num_heads = helper_params["block_size"], helper_params["num_heads"]
        graph_sizes = batch.bincount()
        graph_size_cumsum = graph_sizes.cumsum(0)

        region_indices_eta, region_indices_phi = [], []
        for graph_idx in range(len(graph_size_cumsum)):
            start_idx = 0 if graph_idx == 0 else graph_size_cumsum[graph_idx - 1]
            end_idx = graph_size_cumsum[graph_idx]
            sorted_eta_idx = torch.argsort(coords[start_idx:end_idx, 0], dim=-1)
            sorted_phi_idx = torch.argsort(coords[start_idx:end_idx, 1], dim=-1)

            region_indices_eta.append(quantile_partition(sorted_eta_idx, regions[0][:, None]))
            region_indices_phi.append(quantile_partition(sorted_phi_idx, regions[1][:, None]))
        region_indices_eta = torch.cat(region_indices_eta, dim=-1)
        region_indices_phi = torch.cat(region_indices_phi, dim=-1)

        combined_shifts = bit_shift(region_indices_eta.long(), region_indices_phi.long())
        combined_shifts = bit_shift(combined_shifts, batch[None])
        combined_shifts = rearrange(combined_shifts, "(c h) n -> c h n", h=num_heads)

        pad_seq, unpad_seq = pad_and_unpad(batch, block_size, combined_shifts[0, 0], graph_sizes)
        
        x = x[pad_seq]
    
        kwargs["combined_shifts"] = combined_shifts[..., pad_seq]
        kwargs["coords"] = coords[pad_seq]
    return x, kwargs, unpad_seq


class QTransformer(nn.Module):
    def __init__(self, in_dim, coords_dim, dropout=0.0, **kwargs):
        super().__init__()
        self.n_layers = kwargs["n_layers"]
        self.h_dim = kwargs["h_dim"]
        self.out_dim = kwargs['out_dim']
        self.baseline = kwargs.get('baseline', False)
        self.mlp_out_layers = kwargs['mlp_out_layers']
        self.mlp_out_hdim = kwargs['mlp_out_hdim']
        
        self.weight_bit_width = kwargs.get('weight_bit_width', 4)
        self.bias_quant = kwargs.get('bias_quant', None)
        return_quant_tensor = self.bias_quant
        
        self.Tanh = qnn.QuantTanh()
        self.feat_encoder = nn.Sequential(
            qnn.QuantLinear(in_dim, self.h_dim, bias=True, weight_bit_width=self.weight_bit_width, bias_quant=self.bias_quant, return_quant_tensor=return_quant_tensor),
            qnn.QuantReLU(),
            qnn.QuantLinear(self.h_dim, self.h_dim, bias=True, weight_bit_width=self.weight_bit_width, bias_quant=self.bias_quant, return_quant_tensor=return_quant_tensor),
        )

        self.attns = nn.ModuleList()
        for _ in range(self.n_layers):
            self.attns.append(QAttn(coords_dim, **kwargs))

        self.dropout = nn.Dropout(dropout)
        self.W = qnn.QuantLinear(self.h_dim * (self.n_layers + 1), int(self.h_dim // 2), bias=False, weight_bit_width=self.weight_bit_width, bias_quant=self.bias_quant, return_quant_tensor=return_quant_tensor)
        
        self.mlp_out = []
        
        self.mlp_out.append(qnn.QuantLinear(int(self.h_dim // 2), self.mlp_out_hdim, bias=True, weight_bit_width=self.weight_bit_width, bias_quant=self.bias_quant, return_quant_tensor=return_quant_tensor))
        self.mlp_out.append(qnn.QuantTanh())
        self.mlp_out.append(LayerNorm(self.mlp_out_hdim))
        
        if self.mlp_out_layers > 2:
            for _ in range(self.mlp_out_layers-2):
                self.mlp_out.append(qnn.QuantLinear(self.mlp_out_hdim, self.mlp_out_hdim, bias=True, weight_bit_width=self.weight_bit_width, bias_quant=self.bias_quant, return_quant_tensor=return_quant_tensor))
                self.mlp_out.append(qnn.QuantTanh())
                self.mlp_out.append(LayerNorm(self.mlp_out_hdim))
            
        self.mlp_out.append(qnn.QuantLinear(self.mlp_out_hdim, int(self.h_dim // 2), bias=True, weight_bit_width=self.weight_bit_width, bias_quant=self.bias_quant, return_quant_tensor=return_quant_tensor))
        self.mlp_out.append(qnn.QuantTanh())
        self.mlp_out.append(LayerNorm(int(self.h_dim // 2)))
        
        self.mlp_out = nn.Sequential(*self.mlp_out)
        #print(self.mlp_out)
        self.helper_params = {}

        self.helper_params["block_size"] = kwargs["block_size"]
        self.regions = nn.Parameter(
            get_regions(kwargs["num_regions"], kwargs["n_hashes"], kwargs["num_heads"]), requires_grad=False
        )
        self.helper_params["regions"] = self.regions
        self.helper_params["num_heads"] = kwargs["num_heads"]
    
        self.out_proj = qnn.QuantLinear(int(self.h_dim // 2), self.out_dim, bias=True, weight_bit_width=self.weight_bit_width, bias_quant=self.bias_quant, return_quant_tensor=return_quant_tensor)

    def forward(self, x, coords, batch):
        if self.baseline == False:
            x, kwargs, unpad_seq = prepare_input(x, coords, batch, self.helper_params)

            encoded_x = self.feat_encoder(x)
            all_encoded_x = [encoded_x]
            for i in range(self.n_layers):
                encoded_x = self.attns[i](encoded_x, kwargs)
                all_encoded_x.append(encoded_x)
            
            encoded_x = self.W(torch.cat(all_encoded_x, dim=-1))
            encoded_x = self.Tanh(encoded_x)
 
            out = encoded_x + self.dropout(self.mlp_out(encoded_x))
            
            out = global_mean_pool(out[unpad_seq], batch)
            
            out = self.out_proj(out)
        else:

            encoded_x = self.feat_encoder(x)
            all_encoded_x = [encoded_x]
            for i in range(self.n_layers):
                encoded_x = self.attns[i](encoded_x, None)
                all_encoded_x.append(encoded_x)
            
            encoded_x = tanh(self.W(torch.cat(all_encoded_x, dim=-1)))
            out = encoded_x + self.dropout(self.mlp_out(encoded_x))
            
            out = global_mean_pool(out, batch)
            
            out = self.out_proj(out)
            
        return out


class QAttn(nn.Module):
    def __init__(self, coords_dim, **kwargs):
        super().__init__()
        self.dim_per_head = kwargs["h_dim"]
        self.num_heads = kwargs["num_heads"]
        self.baseline = kwargs.get('baseline', False)
        
        self.weight_bit_width = kwargs.get('weight_bit_width', 4)
        self.bias_quant = kwargs.get('bias_quant', None)
        return_quant_tensor = self.bias_quant
        
        self.w_q = qnn.QuantLinear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False, weight_bit_width=self.weight_bit_width, bias_quant=self.bias_quant, return_quant_tensor=return_quant_tensor)
        self.w_k = qnn.QuantLinear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False, weight_bit_width=self.weight_bit_width, bias_quant=self.bias_quant, return_quant_tensor=return_quant_tensor)
        self.w_v = qnn.QuantLinear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False, weight_bit_width=self.weight_bit_width, bias_quant=self.bias_quant, return_quant_tensor=return_quant_tensor)

        # +2 for data.pos
        if self.baseline:
            self.attn = qnn.QuantMultiheadAttention(self.num_heads*self.dim_per_head, self.num_heads, packed_in_proj=False)
            self.out_linear = qnn.QuantLinear(self.num_heads * self.dim_per_head, self.dim_per_head, bias=True, weight_bit_width=self.weight_bit_width, bias_quant=self.bias_quant, return_quant_tensor=return_quant_tensor)
        else:
            self.attn = HEPTAttention(self.dim_per_head + coords_dim, **kwargs)
            
        #print('Attention Layer: ',self.attn)
        self.dropout = nn.Dropout(0.1)
        self.norm1 = LayerNorm(self.dim_per_head)
        self.norm2 = LayerNorm(self.dim_per_head)
        self.ff = nn.Sequential(
            qnn.QuantLinear(self.dim_per_head, self.dim_per_head, bias=True, weight_bit_width=self.weight_bit_width, bias_quant=self.bias_quant, return_quant_tensor=return_quant_tensor),
            qnn.QuantReLU(),
            qnn.QuantLinear(self.dim_per_head, self.dim_per_head, bias=True, weight_bit_width=self.weight_bit_width, bias_quant=self.bias_quant, return_quant_tensor=return_quant_tensor),
        )

        # eta/phi from data.pos use the same weights as they are used to calc dR
        self.w_rpe = qnn.QuantLinear(kwargs["num_w_per_dist"] * (coords_dim - 1), self.num_heads * self.dim_per_head, bias=True, weight_bit_width=self.weight_bit_width, bias_quant=self.bias_quant, return_quant_tensor=return_quant_tensor)

    def forward(self, x, kwargs):
        x_normed = self.norm1(x)
        q, k, v = self.w_q(x_normed), self.w_k(x_normed), self.w_v(x_normed)
        
        del x_normed
        
        if self.baseline == False:
            aggr_out = self.attn(q, k, v, pe=kwargs["coords"], w_rpe=self.w_rpe, **kwargs)
        else:
            aggr_out = self.attn(q, k, v, need_weights=False)[0]
            aggr_out = self.out_linear(aggr_out)

        x = x + self.dropout(aggr_out)
        x = self.ff(self.norm2(x))
        x = x + self.dropout(x)

        return x


class QDecoder(nn.Module):
    def __init__(self, in_dim, **kwargs):
        super().__init__()
        
        self.weight_bit_width = kwargs.get('weight_bit_width', 4)
        self.bias_quant = kwargs.get('bias_quant', None)
        return_quant_tensor = self.bias_quant
        
        self.mlp = nn.Sequential(
            qnn.QuantLinear(in_dim, in_dim*2, bias=True, weight_bit_width=self.weight_bit_width, bias_quant=self.bias_quant, return_quant_tensor=return_quant_tensor),
            qnn.QuantReLU(),
            LayerNorm(in_dim*2, mode='graph'),
            
            qnn.QuantLinear(in_dim*2, in_dim, bias=True, weight_bit_width=self.weight_bit_width, bias_quant=self.bias_quant, return_quant_tensor=return_quant_tensor),
            qnn.QuantReLU(),
            LayerNorm(in_dim, mode='graph'),
            
            qnn.QuantLinear(in_dim, in_dim*2, bias=True, weight_bit_width=self.weight_bit_width, bias_quant=self.bias_quant, return_quant_tensor=return_quant_tensor),
            qnn.QuantReLU(),
            LayerNorm(in_dim*2, mode='graph'),
            
            qnn.QuantLinear(in_dim*2, in_dim, bias=True, weight_bit_width=self.weight_bit_width, bias_quant=self.bias_quant, return_quant_tensor=return_quant_tensor),
            qnn.QuantReLU(),
            LayerNorm(in_dim, mode='graph'),
            
            )
        
            
        self.out = qnn.QuantLinear(in_dim, 1, bias=True, weight_bit_width=self.weight_bit_width, bias_quant=self.bias_quant, return_quant_tensor=return_quant_tensor)
    
    def forward(self, x):
        x = self.mlp(x)
        x = self.out(x)
        
        return x

    
    