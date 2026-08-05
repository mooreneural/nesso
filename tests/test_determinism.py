"""Predictions must be reproducible for a fixed seed.

Two inputs to the model were previously drawn from unseeded RNGs, so the same
YAML gave different atomistic features on every run even though
``docs/prediction.md`` documents ``--seed`` as giving "reproducible predictions":

* the ligand 3D conformer, embedded by RDKit ETKDG with no ``randomSeed`` (its
  default is ``-1``, i.e. random);
* the per-residue roto-translation of ``ref_pos``, drawn from the *global* torch
  RNG, which ``seed_everything(..., workers=True)`` seeds per DataLoader worker,
  making the result depend on ``--num_workers``.

A third source sat between them: ``preprocess_yamls`` collected results via
``as_completed``, so the manifest order followed worker scheduling rather than
input order, and the per-record RNG was seeded from that position. Seeding the
augmentation was therefore not enough on its own.

These tests pin all three. They are ligand-only so they need no CCD asset and run
on plain CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import DataLoader

from nesso.data.featurizer import NessoFeaturizer
from nesso.data.inference import (
    InferenceDataset,
    inference_collate,
    record_rng_seed,
)
from nesso.data.types import Manifest
from nesso.data.yaml_input import (
    DEFAULT_CONFORMER_SEED,
    conformer_seed,
    get_conformer,
    parse_yaml,
)

# Flexible, drug-like: many rotatable bonds, so ETKDG genuinely varies.
_SMILES = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
_YAML = f'version: 1\nsequences:\n  - ligand:\n      id: B\n      smiles: "{_SMILES}"\n'


def _parse_coords(tmp_path: Path, name: str, seed: int | None) -> np.ndarray:
    work = tmp_path / name
    mol_dir = work / "rdkit_conformers"
    mol_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = work / "lig.yaml"
    yaml_path.write_text(_YAML)
    struct, _, _, _ = parse_yaml(
        yaml_path, mol_dir, ccd_dict=None, record_id="lig", base_seed=seed
    )
    return struct.coords["coords"].copy()


def _internal_distances(x: np.ndarray) -> np.ndarray:
    """Rotation- and translation-invariant description of a conformer."""
    return np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)


def test_same_seed_gives_identical_conformer(tmp_path: Path) -> None:
    a = _parse_coords(tmp_path, "a", DEFAULT_CONFORMER_SEED)
    b = _parse_coords(tmp_path, "b", DEFAULT_CONFORMER_SEED)
    assert np.array_equal(a, b)
    # Invariant under pose, so this also rules out "same shape, different frame".
    assert np.array_equal(_internal_distances(a), _internal_distances(b))


def test_different_seed_gives_different_conformer(tmp_path: Path) -> None:
    """Guards against the seed being accepted but ignored."""
    a = _parse_coords(tmp_path, "a", DEFAULT_CONFORMER_SEED)
    c = _parse_coords(tmp_path, "c", DEFAULT_CONFORMER_SEED + 1)
    assert not np.array_equal(a, c)


def test_conformer_seed_depends_on_molecule_not_order() -> None:
    """`preprocess_yamls` embeds in a ProcessPoolExecutor, so the seed must not
    depend on processing order."""
    mol_a = Chem.AddHs(Chem.MolFromSmiles(_SMILES))
    mol_b = Chem.AddHs(Chem.MolFromSmiles(_SMILES))
    other = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))

    assert conformer_seed(mol_a, 42) == conformer_seed(mol_b, 42)
    assert conformer_seed(mol_a, 42) != conformer_seed(other, 42)
    assert conformer_seed(mol_a, 42) != conformer_seed(mol_a, 43)
    # RDKit takes a signed 32-bit seed.
    assert 0 <= conformer_seed(mol_a, 42) < 2**31 - 1


def test_conformer_seed_can_be_opted_out() -> None:
    """`base_seed=None` keeps RDKit's default (non-deterministic) behaviour."""
    mol = Chem.AddHs(Chem.MolFromSmiles(_SMILES))
    get_conformer(mol, None)
    assert mol.GetNumConformers() == 1


def _ref_pos_by_record(
    tmp_path: Path, num_workers: int, n_records: int = 3, reverse: bool = False
) -> dict[str, np.ndarray]:
    """Featurize several records through a real DataLoader at a given worker count."""
    mol_dir = tmp_path / "rdkit_conformers"
    struct_dir = tmp_path / "structures"
    esm_dir = tmp_path / "esm"
    for directory in (mol_dir, struct_dir, esm_dir):
        directory.mkdir(parents=True, exist_ok=True)

    records = []
    for i in range(n_records):
        yaml_path = tmp_path / f"lig{i}.yaml"
        yaml_path.write_text(_YAML)
        struct, record, _, _ = parse_yaml(
            yaml_path, mol_dir, ccd_dict=None, record_id=f"lig{i}"
        )
        struct.dump(struct_dir / f"{record.id}.npz")
        records.append(record)

    if reverse:
        records = list(reversed(records))

    dataset = InferenceDataset(
        manifest=Manifest(records),
        target_dir=tmp_path,
        featurizer=NessoFeaturizer(
            esm_emb_dir=esm_dir, esm_emb_dim=1280, esm_num_layers=33
        ),
        ligand_dir=mol_dir,
        ccd_pkl=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=inference_collate,
    )
    return {b["record"][0].id: b["ref_pos"][0].numpy().copy() for b in loader}


def test_ref_pos_independent_of_num_workers(tmp_path: Path) -> None:
    """ref_pos augmentation must not read the global (per-worker) torch RNG."""
    torch.manual_seed(0)
    single = _ref_pos_by_record(tmp_path / "w0", 0)
    torch.manual_seed(0)
    multi = _ref_pos_by_record(tmp_path / "w2", 2)

    assert set(single) == set(multi)
    for key in single:
        assert np.array_equal(single[key], multi[key]), key


def test_ref_pos_unaffected_by_global_rng_state(tmp_path: Path) -> None:
    """Consuming the global RNG beforehand must not change featurization."""
    torch.manual_seed(0)
    baseline = _ref_pos_by_record(tmp_path / "base", 0)
    torch.manual_seed(0)
    _ = torch.randn(17)  # perturb global RNG stream position
    shifted = _ref_pos_by_record(tmp_path / "shift", 0)

    for key in baseline:
        assert np.array_equal(baseline[key], shifted[key]), key


_OTHER_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
]


def _screen(tmp_path: Path, name: str, smiles: list[str]) -> dict[str, np.ndarray]:
    """Parse a batch of ligands, returning coordinates keyed by SMILES."""
    work = tmp_path / name
    mol_dir = work / "rdkit_conformers"
    mol_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, np.ndarray] = {}
    for i, smi in enumerate(smiles):
        yaml_path = work / f"c{i}.yaml"
        yaml_path.write_text(
            f'version: 1\nsequences:\n  - ligand:\n      id: B\n      smiles: "{smi}"\n'
        )
        struct, _, _, _ = parse_yaml(
            yaml_path, mol_dir, ccd_dict=None, record_id=f"c{i}"
        )
        out[smi] = struct.coords["coords"].copy()
    return out


def test_conformer_independent_of_batch_composition_and_order(tmp_path: Path) -> None:
    """A ligand's geometry must depend only on that ligand.

    Previously RDKit's unseeded RNG advanced with every embedding, so a compound's
    conformer depended on what else was in the batch and on the order it was
    processed in. That silently changed already-computed results whenever a
    screening library was extended or its inputs reordered.
    """
    alone = _screen(tmp_path, "alone", [_SMILES])
    with_others = _screen(tmp_path, "with_others", [*_OTHER_SMILES, _SMILES])
    reordered = _screen(tmp_path, "reordered", [_SMILES, *reversed(_OTHER_SMILES)])

    assert np.array_equal(alone[_SMILES], with_others[_SMILES])
    assert np.array_equal(alone[_SMILES], reordered[_SMILES])


def test_record_rng_seed_depends_on_id_not_position() -> None:
    """The per-record RNG must be keyed on identity, not on the dataset index.

    ``as_completed`` in ``preprocess_yamls`` made the manifest order follow worker
    scheduling, so seeding from the index made ``ref_pos`` depend on
    ``--num_workers``. Adding, removing or renaming inputs shifts it too.
    """
    assert record_rng_seed("lig") == record_rng_seed("lig")
    assert record_rng_seed("lig") != record_rng_seed("other")
    # numpy requires a uint32 seed.
    assert 0 <= record_rng_seed("lig") < 2**32


def test_ref_pos_independent_of_record_position(tmp_path: Path) -> None:
    """The same record must featurize identically wherever it sits in the batch.

    Seeding the per-record RNG from the dataset index made a record's features
    depend on its position, so reordering the manifest (which `as_completed` did
    on its own at ``--num_workers > 1``) changed ``ref_pos``.
    """
    forward = _ref_pos_by_record(tmp_path / "fwd", 0, n_records=3)
    # Identical records, reversed order, so every index changes.
    backward = _ref_pos_by_record(tmp_path / "rev", 0, n_records=3, reverse=True)

    assert set(forward) == set(backward)
    for key in forward:
        assert np.array_equal(forward[key], backward[key]), key
