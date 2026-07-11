"""Tests for pocket token selection (nesso/data/crop.py)."""

from __future__ import annotations

import numpy as np
import pytest

from nesso.data.crop import select_pocket_token_indices


def _single_chain_case(max_tokens: int, threshold: float):
    """One protein chain (tokens 10..14) + one ligand token (99)."""
    return dict(
        protein_token_indices=np.array([10, 11, 12, 13, 14]),
        protein_asym_ids=np.array([0, 0, 0, 0, 0]),
        protein_res_idxs=np.array([0, 1, 2, 3, 4]),
        dists_to_ligand=np.array([1.0, 2.0, 3.0, 20.0, 21.0]),
        ligand_token_indices=np.array([99]),
        max_tokens=max_tokens,
        threshold_distance=threshold,
    )


def test_keeps_ligand_and_within_cutoff_protein():
    # cutoff admits tokens 10,11,12; budget (4 - 1 ligand = 3) fits all three.
    out = select_pocket_token_indices(**_single_chain_case(max_tokens=4, threshold=5.0))
    assert out == [10, 11, 12, 99]


def test_budget_keeps_nearest_first():
    # all five within cutoff, but budget leaves room for only 2 protein tokens.
    out = select_pocket_token_indices(
        **_single_chain_case(max_tokens=3, threshold=100.0)
    )
    assert out == [10, 11, 99]  # nearest two by distance + ligand


def test_ligand_always_retained_even_when_no_protein_in_cutoff():
    out = select_pocket_token_indices(**_single_chain_case(max_tokens=4, threshold=0.0))
    assert out == [99]


def test_requires_at_least_one_ligand_token():
    case = _single_chain_case(max_tokens=4, threshold=5.0)
    case["ligand_token_indices"] = np.array([], dtype=np.int64)
    with pytest.raises(ValueError, match="at least one ligand"):
        select_pocket_token_indices(**case)
