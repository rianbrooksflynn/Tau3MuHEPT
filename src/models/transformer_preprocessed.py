import torch
from torch import nn


class TranformerPreprocessed(nn.Module):
    def __init__(self, in_dim, coords_dim, **kwargs):
        super().__init__()
        self.coords_dim = coords_dim
        self.n_layers = 2
        self.dim_per_head = kwargs["h_dim"]
        self.num_heads = kwargs["num_heads"]
        self.out_dim = kwargs['out_dim']
        self.block_size = kwargs["block_size"]
        self.n_hashes = kwargs["n_hashes"]
        self.num_w_per_dist = kwargs["num_w_per_dist"]
        self.mlp_out_hdim = kwargs['mlp_out_hdim']

        # Feature encoder
        self.feat_encoder = nn.Sequential(
            nn.Linear(in_dim, self.dim_per_head),
            nn.ReLU(),
            nn.Linear(self.dim_per_head, self.dim_per_head),
        )

        # Attention layer 1
        self.norm1_1 = nn.LayerNorm(self.dim_per_head)
        self.norm1_2 = nn.LayerNorm(self.dim_per_head)
        self.w_q1 = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
        self.w_k1 = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
        self.w_v1 = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
        self.w_rpe1 = nn.Parameter(torch.randn(self.num_heads * self.dim_per_head, self.num_w_per_dist * (self.coords_dim - 1)))
        self.alpha1 = nn.Parameter(torch.normal(0, 1, (self.num_heads, self.dim_per_head + self.coords_dim, self.n_hashes)))
        self.alpha1.requires_grad = False
        self.out_linear1 = nn.Linear(self.num_heads * self.dim_per_head, self.dim_per_head)
        self.ff1 = nn.Sequential(
            nn.Linear(self.dim_per_head, self.dim_per_head),
            nn.ReLU(),
            nn.Linear(self.dim_per_head, self.dim_per_head),
        )

        # Attention layer 2
        self.norm2_1 = nn.LayerNorm(self.dim_per_head)
        self.norm2_2 = nn.LayerNorm(self.dim_per_head)
        self.w_q2 = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
        self.w_k2 = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
        self.w_v2 = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
        self.w_rpe2 = nn.Parameter(torch.randn(self.num_heads * self.dim_per_head, self.num_w_per_dist * (self.coords_dim - 1)))
        self.alpha2 = nn.Parameter(torch.normal(0, 1, (self.num_heads, self.dim_per_head + self.coords_dim, self.n_hashes)))
        self.alpha2.requires_grad = False
        self.out_linear2 = nn.Linear(self.num_heads * self.dim_per_head, self.dim_per_head)
        self.ff2 = nn.Sequential(
            nn.Linear(self.dim_per_head, self.dim_per_head),
            nn.ReLU(),
            nn.Linear(self.dim_per_head, self.dim_per_head),
        )

        # After attentions
        self.W = nn.Linear(self.dim_per_head * (self.n_layers + 1), int(self.dim_per_head // 2), bias=False)

        self.mlp_out = nn.Sequential(
            nn.Linear(int(self.dim_per_head // 2), self.mlp_out_hdim),
            nn.LayerNorm(self.mlp_out_hdim),
            nn.Tanh(),
            nn.Linear(self.mlp_out_hdim, self.mlp_out_hdim),
            nn.LayerNorm(self.mlp_out_hdim),
            nn.Tanh(),
            nn.Linear(self.mlp_out_hdim, self.mlp_out_hdim),
            nn.LayerNorm(self.mlp_out_hdim),
            nn.Tanh(),
            nn.Linear(self.mlp_out_hdim, self.mlp_out_hdim),
            nn.LayerNorm(self.mlp_out_hdim),
            nn.Tanh(),
            nn.Linear(self.mlp_out_hdim, int(self.dim_per_head // 2)),
        )

        self.out_proj = nn.Linear(int(self.dim_per_head // 2), self.out_dim)

    def forward(self, x, combined_shifts, coords, unpad_seq):
        # Encode features
        x = self.feat_encoder(x) # (padded_size, dim_per_head)

        all_encoded_x = x

        ###########################################################################################
        # Attention layer 1
        x_normed = self.norm1_1(x)

        q, k, v = self.w_q1(x_normed), self.w_k1(x_normed), self.w_v1(x_normed)
        q = q.view(-1, self.num_heads, self.dim_per_head)
        k = k.view(-1, self.num_heads, self.dim_per_head)
        v = v.view(-1, self.num_heads, self.dim_per_head)

        # Prep q, k
        w = self.w_rpe1.view(self.num_heads, self.dim_per_head, -1, self.num_w_per_dist)
        qw = w.sum(dim=1).clamp(max=50).exp().sum(dim=-1)
        new_qw_expand_dim = torch.cat([qw[:, :1], qw], dim=-1) # (num_heads, coords_dim)
        sqrt_w_r = torch.sqrt(2 * new_qw_expand_dim).unsqueeze(0) * coords.unsqueeze(1) # (padded_size, num_heads, coords_dim)
        q_hat = torch.cat([q, sqrt_w_r], dim=-1).permute(1, 0, 2) # (num_heads, padded_size, dim_per_head + coords_dim)
        k_hat = torch.cat([k, sqrt_w_r], dim=-1).permute(1, 0, 2) # (num_heads, padded_size, dim_per_head + coords_dim)
        value = v.permute(1, 0, 2) # (num_heads, padded_size, dim_per_head)

        # E2LSH
        with torch.no_grad():
            q_hashed = torch.bmm(q_hat, self.alpha1).permute(2, 0, 1) # (n_hashes, num_heads, padded_size)
            k_hashed = torch.bmm(k_hat, self.alpha1).permute(2, 0, 1) # (n_hashes, num_heads, padded_size)
            max_hash_shift = torch.max(q_hashed.max(-1, keepdim=True).values, k_hashed.max(-1, keepdim=True).values)
            min_hash_shift = torch.min(q_hashed.min(-1, keepdim=True).values, k_hashed.min(-1, keepdim=True).values)
            hash_shift = max_hash_shift - min_hash_shift # (n_hashes, num_heads, 1)
        
        combined_shifts = combined_shifts * hash_shift # (n_hashes, num_heads, padded_size)
        q_hashed = q_hashed + combined_shifts
        k_hashed = k_hashed + combined_shifts

        q_positions = q_hashed.argsort(dim=-1)
        k_positions = k_hashed.argsort(dim=-1)

        # Sort to buckets
        q_positions_expanded = q_positions.unsqueeze(-1).expand(q_positions.shape + (self.dim_per_head + self.coords_dim,))
        q_hat_expanded = q_hat.unsqueeze(0).expand(q_positions_expanded.shape[:-2] + q_hat.shape[-2:]) # (n_hashes, num_heads, padded_size, dim_per_head + coords_dim)
        q_batch_selected = q_hat_expanded.gather(-2, q_positions_expanded)
        s_query = q_batch_selected.view(self.n_hashes, self.num_heads, -1, self.block_size, self.dim_per_head + self.coords_dim)

        k_positions_unsqueezed = k_positions.unsqueeze(-1)
        k_positions_expanded = k_positions_unsqueezed.expand(k_positions.shape + (self.dim_per_head + self.coords_dim,))
        k_hat_expanded = k_hat.unsqueeze(0).expand(k_positions_expanded.shape[:-2] + k_hat.shape[-2:]) # (n_hashes, num_heads, padded_size, dim_per_head + coords_dim)
        k_batch_selected = k_hat_expanded.gather(-2, k_positions_expanded)
        s_key = k_batch_selected.view(self.n_hashes, self.num_heads, -1, self.block_size, self.dim_per_head + self.coords_dim)

        v_positions_expanded = k_positions_unsqueezed.expand(k_positions.shape + (self.dim_per_head,))
        v_hat_expanded = value.unsqueeze(0).expand(v_positions_expanded.shape[:-2] + value.shape[-2:]) # (n_hashes, num_heads, padded_size, dim_per_head)
        v_batch_selected = v_hat_expanded.gather(-2, v_positions_expanded)
        s_value = v_batch_selected.view(self.n_hashes, self.num_heads, -1, self.block_size, self.dim_per_head)

        # Compute attention
        q_sq_05 = -0.5 * (s_query**2).sum(dim=-1, keepdim=True)
        k_sq_05 = -0.5 * (s_key**2).sum(dim=-1, keepdim=True)

        clustered_dists = torch.einsum("...id,...jd->...ij", s_query, s_key) # (n_hashes, num_heads, -1, block_size, block_size)
        qk = (clustered_dists + q_sq_05 + k_sq_05.transpose(-1, -2)).clamp(max=0.0).exp()

        denom = qk.sum(dim=-1, keepdim=True) + (1e-20) # (n_hashes, num_heads, -1, block_size, 1)
        so = torch.einsum("...ij,...jd->...id", qk, s_value) # (n_hashes, num_heads, -1, block_size, dim_per_head)

        # Unsort from buckets
        arange = torch.arange(q_positions.shape[-1], device=q_positions.device).expand_as(q_positions)
        q_rev_positions = torch.empty_like(q_positions).scatter(-1, q_positions, arange)

        so_squeezed = so.view(self.n_hashes, self.num_heads, -1, self.dim_per_head)
        q_rev_positions_expanded = q_rev_positions.unsqueeze(-1).expand(q_rev_positions.shape + (self.dim_per_head,))
        so_squeezed_expanded = so_squeezed.expand(q_rev_positions_expanded.shape[:-2] + so_squeezed.shape[-2:])
        o = so_squeezed_expanded.gather(-2, q_rev_positions_expanded)

        denom_squeezed = denom.view(self.n_hashes, self.num_heads, -1, 1)
        q_rev_positions_expanded = q_rev_positions.unsqueeze(-1)
        denom_squeezed_expanded = denom_squeezed.expand(q_rev_positions_expanded.shape[:-2] + denom_squeezed.shape[-2:])
        logits = denom_squeezed_expanded.gather(-2, q_rev_positions_expanded)

        aggr_out = o.sum(dim=0) / logits.sum(dim=0) # (num_heads, padded_size, dim_per_head)
        aggr_out = aggr_out.view(-1, self.num_heads * self.dim_per_head)
        aggr_out = self.out_linear1(aggr_out) # (padded_size, dim_per_head)

        x = x + aggr_out
        x = x + self.ff1(self.norm1_2(x))

        all_encoded_x = torch.cat([all_encoded_x, x], dim=-1)

        ###########################################################################################
        # Attention layer 2
        x_normed = self.norm2_1(x)

        q, k, v = self.w_q2(x_normed), self.w_k2(x_normed), self.w_v2(x_normed)
        q = q.view(-1, self.num_heads, self.dim_per_head)
        k = k.view(-1, self.num_heads, self.dim_per_head)
        v = v.view(-1, self.num_heads, self.dim_per_head)

        # Prep q, k
        w = self.w_rpe2.view(self.num_heads, self.dim_per_head, -1, self.num_w_per_dist)
        qw = w.sum(dim=1).clamp(max=50).exp().sum(dim=-1)
        new_qw_expand_dim = torch.cat([qw[:, :1], qw], dim=-1) # (num_heads, coords_dim)
        sqrt_w_r = torch.sqrt(2 * new_qw_expand_dim).unsqueeze(0) * coords.unsqueeze(1) # (padded_size, num_heads, coords_dim)
        q_hat = torch.cat([q, sqrt_w_r], dim=-1).permute(1, 0, 2) # (num_heads, padded_size, dim_per_head + coords_dim)
        k_hat = torch.cat([k, sqrt_w_r], dim=-1).permute(1, 0, 2) # (num_heads, padded_size, dim_per_head + coords_dim)
        value = v.permute(1, 0, 2) # (num_heads, padded_size, dim_per_head)

        # E2LSH
        with torch.no_grad():
            q_hashed = torch.bmm(q_hat, self.alpha2).permute(2, 0, 1) # (n_hashes, num_heads, padded_size)
            k_hashed = torch.bmm(k_hat, self.alpha2).permute(2, 0, 1) # (n_hashes, num_heads, padded_size)
            max_hash_shift = torch.max(q_hashed.max(-1, keepdim=True).values, k_hashed.max(-1, keepdim=True).values)
            min_hash_shift = torch.min(q_hashed.min(-1, keepdim=True).values, k_hashed.min(-1, keepdim=True).values)
            hash_shift = max_hash_shift - min_hash_shift # (n_hashes, num_heads, 1)
        
        combined_shifts = combined_shifts * hash_shift # (n_hashes, num_heads, padded_size)
        q_hashed = q_hashed + combined_shifts
        k_hashed = k_hashed + combined_shifts

        q_positions = q_hashed.argsort(dim=-1)
        k_positions = k_hashed.argsort(dim=-1)

        # Sort to buckets
        q_positions_expanded = q_positions.unsqueeze(-1).expand(q_positions.shape + (self.dim_per_head + self.coords_dim,))
        q_hat_expanded = q_hat.unsqueeze(0).expand(q_positions_expanded.shape[:-2] + q_hat.shape[-2:]) # (n_hashes, num_heads, padded_size, dim_per_head + coords_dim)
        q_batch_selected = q_hat_expanded.gather(-2, q_positions_expanded)
        s_query = q_batch_selected.view(self.n_hashes, self.num_heads, -1, self.block_size, self.dim_per_head + self.coords_dim)

        k_positions_expanded = k_positions.unsqueeze(-1).expand(k_positions.shape + (self.dim_per_head + self.coords_dim,))
        k_hat_expanded = k_hat.unsqueeze(0).expand(k_positions_expanded.shape[:-2] + k_hat.shape[-2:]) # (n_hashes, num_heads, padded_size, dim_per_head + coords_dim)
        k_batch_selected = k_hat_expanded.gather(-2, k_positions_expanded)
        s_key = k_batch_selected.view(self.n_hashes, self.num_heads, -1, self.block_size, self.dim_per_head + self.coords_dim)

        v_positions_expanded = k_positions.unsqueeze(-1).expand(k_positions.shape + (self.dim_per_head,))
        v_hat_expanded = value.unsqueeze(0).expand(v_positions_expanded.shape[:-2] + value.shape[-2:]) # (n_hashes, num_heads, padded_size, dim_per_head)
        v_batch_selected = v_hat_expanded.gather(-2, v_positions_expanded)
        s_value = v_batch_selected.view(self.n_hashes, self.num_heads, -1, self.block_size, self.dim_per_head)

        # Compute attention
        q_sq_05 = -0.5 * (s_query**2).sum(dim=-1, keepdim=True)
        k_sq_05 = -0.5 * (s_key**2).sum(dim=-1, keepdim=True)

        clustered_dists = torch.einsum("...id,...jd->...ij", s_query, s_key) # (n_hashes, num_heads, -1, block_size, block_size)
        qk = (clustered_dists + q_sq_05 + k_sq_05.transpose(-1, -2)).clamp(max=0.0).exp()

        denom = qk.sum(dim=-1, keepdim=True) + (1e-20) # (n_hashes, num_heads, -1, block_size, 1)
        so = torch.einsum("...ij,...jd->...id", qk, s_value) # (n_hashes, num_heads, -1, block_size, dim_per_head)

        # Unsort from buckets
        arange = torch.arange(q_positions.shape[-1], device=q_positions.device).expand_as(q_positions)
        q_rev_positions = torch.empty_like(q_positions).scatter(-1, q_positions, arange)

        so_squeezed = so.view(self.n_hashes, self.num_heads, -1, self.dim_per_head)
        q_rev_positions_expanded = q_rev_positions.unsqueeze(-1).expand(q_rev_positions.shape + (self.dim_per_head,))
        so_squeezed_expanded = so_squeezed.expand(q_rev_positions_expanded.shape[:-2] + so_squeezed.shape[-2:])
        o = so_squeezed_expanded.gather(-2, q_rev_positions_expanded)

        denom_squeezed = denom.view(self.n_hashes, self.num_heads, -1, 1)
        q_rev_positions_expanded = q_rev_positions.unsqueeze(-1)
        denom_squeezed_expanded = denom_squeezed.expand(q_rev_positions_expanded.shape[:-2] + denom_squeezed.shape[-2:])
        logits = denom_squeezed_expanded.gather(-2, q_rev_positions_expanded)

        aggr_out = o.sum(dim=0) / logits.sum(dim=0) # (num_heads, padded_size, dim_per_head)
        aggr_out = aggr_out.view(-1, self.num_heads * self.dim_per_head)
        aggr_out = self.out_linear2(aggr_out) # (padded_size, dim_per_head)

        x = x + aggr_out
        x = x + self.ff2(self.norm2_2(x))

        all_encoded_x = torch.cat([all_encoded_x, x], dim=-1)

        ###########################################################################################

        x = torch.tanh(self.W(all_encoded_x))
        x = x + self.mlp_out(x)

        out = self.out_proj(torch.mean(x[unpad_seq], dim=0))

        return out
