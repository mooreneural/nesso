"""Extract single- and pairwise features from Nesso predictions.safetensors.

Requires predictions saved with ``--save_metadata`` (includes ``z`` and ``mol_type``).

Run from the repo root::

    python tutorial/extract_features.py predictions/MY_RECORD/predictions.safetensors
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from nesso.data import const


def extract_features(safetensors_path: Path | str) -> dict[str, torch.Tensor]:
    """Load predictions.safetensors and slice pairwise ``z`` by chain type.

    Parameters
    ----------
    safetensors_path
        Path to ``predictions.safetensors`` from a Nesso run with
        ``--save_metadata``.

    Returns
    -------
    dict[str, Tensor]
        Pairwise blocks (aligned with refined tokens in ``z``):

        - ``z_pp``: ``[N_prot, N_prot, d]`` protein-protein
        - ``z_pl``: ``[N_prot, N_lig, d]`` protein-ligand
        - ``z_ll``: ``[N_lig, N_lig, d]`` ligand-ligand

        Single (per-token) features from row-mean of ``z``:

        - ``s_prot``: ``[N_prot, d]``
        - ``s_lig``: ``[N_lig, d]``
    """
    path = Path(safetensors_path)
    data = load_file(str(path))

    if "mol_type" not in data:
        msg = (
            f"{path} has no mol_type tensor. Re-run prediction with a version that "
            "saves mol_type, or use --save_metadata on a current build."
        )
        raise KeyError(msg)

    z = data["z"].float()
    mol_type = data["mol_type"]

    protein_id = const.chain_type_ids["PROTEIN"]
    ligand_id = const.chain_type_ids["NONPOLYMER"]
    prot_idx = (mol_type == protein_id).nonzero(as_tuple=True)[0]
    lig_idx = (mol_type == ligand_id).nonzero(as_tuple=True)[0]

    z_pp = z[prot_idx][:, prot_idx]
    z_pl = z[prot_idx][:, lig_idx]
    z_ll = z[lig_idx][:, lig_idx]

    s_prot = z[prot_idx].mean(dim=1)
    s_lig = z[lig_idx].mean(dim=1)

    return {
        "z_pp": z_pp,
        "z_pl": z_pl,
        "z_ll": z_ll,
        "s_prot": s_prot,
        "s_lig": s_lig,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract protein/ligand single and pairwise features from predictions.safetensors."
    )
    parser.add_argument(
        "safetensors_path",
        type=Path,
        help="Path to predictions/MY_RECORD/predictions.safetensors",
    )
    args = parser.parse_args()

    features = extract_features(args.safetensors_path)
    for name, tensor in features.items():
        print(f"{name}: {tuple(tensor.shape)}")


if __name__ == "__main__":
    main()
