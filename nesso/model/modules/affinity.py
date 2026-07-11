# Adapted from https://github.com/jwohlwend/boltz, MIT License, Copyright (c) 2024 Jeremy Wohlwend

import torch
from torch import nn

import nesso.model.layers.initialize as init
from nesso.model.layers.pairformer import PairformerNoSeqModule
from nesso.model.modules.encoders import PairwiseConditioning
from nesso.model.modules.utils import LinearNoBias


class AffinityModule(nn.Module):
    """Affinity Module for predicting binding affinity from pairwise representations."""

    def __init__(
        self,
        token_s: int,
        token_z: int,
        pairformer_args: dict,
        transformer_args: dict,
        num_dist_bins: int = 64,
        max_dist: float = 22.0,
        esm_embed_dim: int = 1280,
        **kwargs,
    ):
        super().__init__()
        self.distogram_proj = LinearNoBias(num_dist_bins, token_z)
        init.gating_init_(self.distogram_proj.weight)

        self.esm_proj = nn.Sequential(
            nn.LayerNorm(esm_embed_dim),
            nn.Linear(esm_embed_dim, token_s),
            nn.ReLU(),
            nn.Linear(token_s, token_s),
        )
        init.gating_init_(self.esm_proj[-1].weight)
        init.bias_init_zero_(self.esm_proj[-1].bias)

        self.z_norm = nn.LayerNorm(token_z)
        self.z_linear = LinearNoBias(token_z, token_z)

        self.s_to_z_prod_in1 = LinearNoBias(token_s, token_z)
        self.s_to_z_prod_in2 = LinearNoBias(token_s, token_z)

        self.pairwise_conditioner = PairwiseConditioning(
            token_z=token_z,
            dim_token_rel_pos_feats=token_z,
            num_transitions=2,
        )

        self.pairformer_stack = PairformerNoSeqModule(token_z, **pairformer_args)

        classification_head = transformer_args.get("classification_head", False)
        self.affinity_heads = AffinityHeadsTransformer(
            token_z=token_z,
            hidden_dim=token_s,
            classification_head=classification_head,
        )

    def forward(
        self,
        s_inputs: torch.Tensor,
        z: torch.Tensor,
        pdistogram: torch.Tensor,
        feats: dict[str, torch.Tensor],
        use_kernels: bool = False,
    ) -> dict[str, torch.Tensor]:
        pad_token_mask = feats["token_pad_mask"]
        is_protein = ((feats["mol_type"] == 0).float() * pad_token_mask).unsqueeze(-1)

        s_esm_proj = self.esm_proj(feats["s_esm"])
        s_inputs = s_inputs + (s_esm_proj * is_protein)

        z = z.float()
        z = self.z_linear(self.z_norm(z))
        z = (
            z
            + self.s_to_z_prod_in1(s_inputs)[:, :, None, :]
            + self.s_to_z_prod_in2(s_inputs)[:, None, :, :]
        )

        distogram = self.distogram_proj(pdistogram)
        z = z + self.pairwise_conditioner(z_trunk=z, token_rel_pos_feats=distogram)

        rec_mask = (feats["mol_type"] == 0).float() * pad_token_mask
        lig_mask = feats["affinity_token_mask"].float() * pad_token_mask
        cross_pair_mask = (
            lig_mask[:, :, None] * rec_mask[:, None, :]
            + rec_mask[:, :, None] * lig_mask[:, None, :]
            + lig_mask[:, :, None] * lig_mask[:, None, :]
        )

        z = self.pairformer_stack(z, pair_mask=cross_pair_mask, use_kernels=use_kernels)

        return self.affinity_heads(z=z, feats=feats)


class AffinityHeadsTransformer(nn.Module):
    def __init__(
        self,
        token_z: int,
        hidden_dim: int,
        **kwargs,
    ):
        super().__init__()
        self.affinity_out_mlp = nn.Sequential(
            nn.Linear(token_z, token_z),
            nn.ReLU(),
            nn.Linear(token_z, hidden_dim),
            nn.ReLU(),
        )
        self.to_affinity_pred_value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.to_affinity_pred_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.to_affinity_logits_binary = nn.Linear(1, 1)

    def forward(
        self,
        z: torch.Tensor,
        feats: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        pad_token_mask = feats["token_pad_mask"].unsqueeze(-1)
        rec_mask = (feats["mol_type"] == 0).float().unsqueeze(-1) * pad_token_mask
        lig_mask = feats["affinity_token_mask"].float().unsqueeze(-1) * pad_token_mask

        cross_pair_mask = (
            lig_mask[:, :, None] * rec_mask[:, None, :]
            + rec_mask[:, :, None] * lig_mask[:, None, :]
            + (lig_mask[:, :, None] * lig_mask[:, None, :])
        ) * (
            1
            - torch.eye(lig_mask.shape[1], device=lig_mask.device)
            .unsqueeze(-1)
            .unsqueeze(0)
        )

        g = torch.sum(z * cross_pair_mask, dim=(1, 2)) / (
            torch.sum(cross_pair_mask, dim=(1, 2)) + 1e-7
        )
        g = self.affinity_out_mlp(g)
        affinity_pred_value = self.to_affinity_pred_value(g).reshape(-1, 1)
        out_dict = {"affinity_pred_value": affinity_pred_value, "affinity_repr": g}

        affinity_pred_score = self.to_affinity_pred_score(g).reshape(-1, 1)
        out_dict["affinity_logits_binary"] = self.to_affinity_logits_binary(
            affinity_pred_score
        ).reshape(-1, 1)

        return out_dict
