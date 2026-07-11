"""Tests for the tutorial feature-extraction helper (slices ``z`` by ``mol_type``)."""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from nesso.data import const

PROT = const.chain_type_ids["PROTEIN"]
LIG = const.chain_type_ids["NONPOLYMER"]


def test_extract_features_slices_by_mol_type(tmp_path, extract_features):
    n_prot, n_lig, dim = 4, 2, 8
    n = n_prot + n_lig
    z = torch.randn(n, n, dim)
    mol_type = torch.tensor([PROT] * n_prot + [LIG] * n_lig)

    path = tmp_path / "predictions.safetensors"
    save_file({"z": z, "mol_type": mol_type}, str(path))

    feats = extract_features.extract_features(path)

    assert feats["z_pp"].shape == (n_prot, n_prot, dim)
    assert feats["z_pl"].shape == (n_prot, n_lig, dim)
    assert feats["z_ll"].shape == (n_lig, n_lig, dim)
    assert feats["s_prot"].shape == (n_prot, dim)
    assert feats["s_lig"].shape == (n_lig, dim)

    # the protein-protein block should equal the corresponding slice of z
    torch.testing.assert_close(feats["z_pp"], z[:n_prot, :n_prot])


def test_extract_features_requires_mol_type(tmp_path, extract_features):
    path = tmp_path / "predictions.safetensors"
    save_file({"z": torch.randn(3, 3, 4)}, str(path))  # no mol_type

    with pytest.raises(KeyError, match="mol_type"):
        extract_features.extract_features(path)
