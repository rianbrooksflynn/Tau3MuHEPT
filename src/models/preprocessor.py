import torch


class Preprocessor():
    def __init__(self, num_regions, num_heads, n_hashes, block_size, device):
        self.num_regions = num_regions
        self.num_heads = num_heads
        self.n_hashes = n_hashes
        self.block_size = block_size
        self.device = device

        self.regions = self._get_regions().to(self.device)
    
    def prepare_input(
            self,
            x: torch.Tensor, 
            coords: torch.Tensor, 
            batch: torch.Tensor, 
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare the input for the Transformer.

        Args:
            x (torch.Tensor): The input features, of shape (num_elements, in_dim).
            coords (torch.Tensor): The coordinates of the nodes, of shape (num_elements, coords_dim).
            batch (torch.Tensor): The batch vector, of shape (num_elements).

        Returns:
            tuple: A tuple containing:
                - The padded input features, of shape (padded_size, in_dim).
                - The combined shifts, of shape (n_hashes, num_heads, padded_size).
                - The padded coordinates, of shape (padded_size, coords_dim).
                - The unpadding sequence, of shape (padded_size).
                - The original batch vector, of shape (num_elements).
        """
        with torch.no_grad():
            graph_sizes = batch.bincount()

            graph_size_cumsum = graph_sizes.cumsum(0)

            region_indices_eta, region_indices_phi = [], []
            for graph_idx in range(len(graph_size_cumsum)):
                start_idx = 0 if graph_idx == 0 else graph_size_cumsum[graph_idx - 1]
                end_idx = graph_size_cumsum[graph_idx]

                sorted_eta_idx = torch.argsort(coords[start_idx:end_idx, 0], dim=-1)
                sorted_phi_idx = torch.argsort(coords[start_idx:end_idx, 1], dim=-1)

                region_indices_eta.append(self._quantile_partition(sorted_eta_idx, self.regions[0][:, None]))
                region_indices_phi.append(self._quantile_partition(sorted_phi_idx, self.regions[1][:, None]))

            region_indices_eta = torch.cat(region_indices_eta, dim=-1) # (num_heads * n_hashes, num_elements)
            region_indices_phi = torch.cat(region_indices_phi, dim=-1) # (num_heads * n_hashes, num_elements)

            combined_shifts = self._bit_shift(region_indices_eta.long(), region_indices_phi.long())
            combined_shifts = self._bit_shift(combined_shifts, batch[None])
            combined_shifts = combined_shifts.view(self.n_hashes, self.num_heads, -1)

            pad_seq, unpad_seq = self._pad_and_unpad(batch, combined_shifts[0, 0], graph_sizes)

            x = x[pad_seq]
            combined_shifts = combined_shifts[..., pad_seq]
            coords = coords[pad_seq]

        return x, combined_shifts, coords, unpad_seq, batch

    def _get_regions(self, num_and_hashes: int = 2) -> torch.Tensor:
        lower = 2
        upper = 2 * self.num_regions ** (1 / num_and_hashes) - lower

        regions = torch.tensor([
            [torch.rand(1).item() * (upper - lower) + lower for _ in range(num_and_hashes)]
            for _ in range(self.num_heads * self.n_hashes)
        ])

        regions = (self.num_regions / regions.prod(dim=1, keepdim=True)) ** (1 / num_and_hashes) * regions
        regions = torch.round(regions * 3) / 3
        return regions.permute(1, 0)

    def _quantile_partition(self, sorted_indices: torch.Tensor, regions: torch.Tensor) -> torch.Tensor:
        num_elements = sorted_indices.shape[-1]
        region_size = torch.ceil(num_elements / regions)
        inverse_indices = torch.argsort(sorted_indices, dim=-1)

        base = torch.arange(num_elements, device=sorted_indices.device)[None]
        region_indices = base // region_size + 1
        reassigned_regions = region_indices[:, inverse_indices]
        return reassigned_regions


    def _bit_shift(self, base: torch.Tensor, shift_idx: torch.Tensor) -> torch.Tensor:
        max_base = base.max(dim=1, keepdim=True).values
        num_bits = torch.ceil(torch.log2(max_base + 1)).long()
        return (shift_idx << num_bits) | base


    def _pad_and_unpad(
            self,
            batch: torch.Tensor, 
            region_indices: torch.Tensor, 
            raw_sizes: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
        padded_sizes = ((raw_sizes + self.block_size - 1) // self.block_size) * self.block_size
        pad_sizes = padded_sizes - raw_sizes

        pad_cumsum = padded_sizes.cumsum(0)
        pad_seq = torch.arange(pad_cumsum[-1], device=batch.device)
        unpad_seq = torch.ones(pad_cumsum[-1], device=batch.device).bool()

        sorted_region_indices = region_indices.argsort()
        for i in range(len(raw_sizes)):
            idx_to_fill = pad_cumsum[i] - self.block_size - pad_sizes[i] + torch.arange(pad_sizes[i], device=batch.device)
            if i >= 1:
                pad_seq[pad_cumsum[i - 1] :] -= pad_sizes[i - 1]
                idx_to_fill -= pad_sizes[:i].sum()
            pad_seq[pad_cumsum[i] - pad_sizes[i] : pad_cumsum[i]] = sorted_region_indices[idx_to_fill]
            unpad_seq[pad_cumsum[i] - pad_sizes[i] : pad_cumsum[i]] = False
        
        return pad_seq, unpad_seq
