"""ESM-based single representation and pairwise update (MSA-like for proteins)."""

import torch
from torch import Tensor, nn

from nesso.model.layers import initialize


class ESMModule(nn.Module):
    """MSA-like module using ESM2 + s_inputs"""

    def __init__(
        self,
        token_s: int,
        token_z: int,
        esm_embed_dim: int,
        use_esm_all_layers: bool = False,
        esm_num_layers: int = 37,
        dropout: float = 0.0,
        **kwargs,  # for unused kwargs
    ) -> None:
        super().__init__()
        self.use_esm_all_layers = use_esm_all_layers
        self.esm_num_layers = esm_num_layers
        if use_esm_all_layers:
            self.esm_layer_weights = nn.Parameter(torch.zeros(esm_num_layers))

        self.esm_mlp = nn.Sequential(
            nn.LayerNorm(esm_embed_dim),
            nn.Linear(esm_embed_dim, token_s),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(token_s, token_s),
        )
        self.s_inputs_proj = nn.Linear(token_s, token_s)

        # Single-to-pair (outer product in token_z)
        self.esm_z_1 = nn.Linear(token_s, token_z, bias=False)
        self.esm_z_2 = nn.Linear(token_s, token_z, bias=False)
        initialize.gating_init_(self.esm_z_1.weight)
        initialize.gating_init_(self.esm_z_2.weight)
        initialize.gating_init_(self.s_inputs_proj.weight)

    def forward(
        self,
        z: Tensor,
        s_inputs: Tensor,
        s_esm: Tensor,
        pair_mask: Tensor,
        use_kernels: bool = False,
    ) -> Tensor:
        """Update z using s_inputs and s_esm."""
        if s_esm.dim() == 4 and self.use_esm_all_layers:
            weights = self.esm_layer_weights.softmax(0)
            s_esm = torch.einsum("bnld,l->bnd", s_esm, weights)

        s_esm_proj = self.esm_mlp(s_esm)
        s = self.s_inputs_proj(s_inputs) + s_esm_proj

        left = self.esm_z_1(s)  # [B, N, token_z]
        right = self.esm_z_2(s)  # [B, N, token_z]
        delta_z = left[:, :, None, :] + right[:, None, :, :]  # [B, N, N, token_z]
        delta_z = delta_z * pair_mask.unsqueeze(
            -1
        )  # [B, N, N, token_z] (mask out invalid pairs)
        z = z + delta_z  # update pairwise representation with the delta_z

        return z
