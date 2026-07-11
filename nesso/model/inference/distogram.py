"""Distogram inference utilities (expected distances, entropy)."""

import math
from typing import Optional

import torch
from torch import Tensor

from nesso.data import const


def distogram_bin_centers(
    num_bins: int = 64,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Bin centers for a distogram."""
    boundaries = torch.linspace(
        min_dist, max_dist, num_bins - 1, device=device, dtype=dtype
    )
    centers = torch.empty(num_bins, device=device, dtype=dtype)
    centers[0] = 1.5
    centers[-1] = 24.5
    centers[1:-1] = (boundaries[:-1] + boundaries[1:]) * 0.5
    return centers


def compute_expected_distance(
    pdistogram: Tensor,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
) -> Tensor:
    """Expected pairwise distance from distogram logits.

    Parameters
    ----------
    pdistogram
        Logits of shape ``[B, N, N, num_bins]``.

    Returns
    -------
    Tensor
        Expected distances ``[B, N, N]``.
    """
    num_bins = pdistogram.shape[-1]
    probs = pdistogram.softmax(dim=-1).float()
    centers = distogram_bin_centers(
        num_bins=num_bins,
        min_dist=min_dist,
        max_dist=max_dist,
        device=pdistogram.device,
        dtype=probs.dtype,
    )
    return torch.einsum("...b,b->...", probs, centers)


def compute_distogram_entropy(
    pdistogram: Tensor,
    feats: dict[str, Tensor],
    custom_mask: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    """Normalized pairwise distogram entropy with PP/PL/LL masked means.

    Parameters
    ----------
    pdistogram
        Logits of shape ``[B, N, N, num_bins]``.
    feats
        Feature dict containing ``token_disto_mask`` and ``mol_type``.
    custom_mask
        Optional custom mask ``[B, N]`` replacing ``token_disto_mask``.

    Returns
    -------
    dict with keys ``entropy_pair``, ``entropy_pp``, ``entropy_pl``, ``entropy_ll``.
    """
    with torch.autocast("cuda", enabled=False):
        pred = pdistogram.float()
        n_bins = pred.shape[-1]

        base_mask = feats["token_disto_mask"] if custom_mask is None else custom_mask
        base_mask = base_mask.float()
        base_mask = base_mask[:, None, :] * base_mask[:, :, None]
        base_mask = base_mask * (
            1 - torch.eye(base_mask.shape[1], device=base_mask.device)[None]
        )

        mol_type = feats["mol_type"].long()
        is_protein = (mol_type == const.chain_type_ids["PROTEIN"]).float()
        is_ligand = (mol_type == const.chain_type_ids["NONPOLYMER"]).float()

        pp_mask = base_mask * (is_protein[:, :, None] * is_protein[:, None, :])
        ll_mask = base_mask * (is_ligand[:, :, None] * is_ligand[:, None, :])
        pl_mask = base_mask * (
            is_protein[:, :, None] * is_ligand[:, None, :]
            + is_ligand[:, :, None] * is_protein[:, None, :]
        )

        log_Q = torch.nn.functional.log_softmax(pred, dim=-1)
        Q = log_Q.exp()
        H_pair = -torch.sum(Q * log_Q, dim=-1) / math.log(n_bins)

        def _masked_mean_batch(val: Tensor, m: Tensor) -> Tensor:
            num = torch.sum(val * m, dim=(-1, -2))
            den = torch.sum(m, dim=(-1, -2)).clamp(min=1.0)
            return num / den

        return {
            "entropy_pair": H_pair,
            "entropy_pp": _masked_mean_batch(H_pair, pp_mask),
            "entropy_pl": _masked_mean_batch(H_pair, pl_mask),
            "entropy_ll": _masked_mean_batch(H_pair, ll_mask),
        }
