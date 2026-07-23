"""Inference featurization does not build training-only targets.

The forward pass never reads ``disto_target`` or ``token_to_rep_atom`` (the
ground-truth distogram label and the token-to-representative-atom gather
inherited from the Boltz training featurizer). Since the codebase is
inference-only, those tensors are not built at all.

These tests pin that: the featurizer omits both keys, and the model forward is
bit-identical whether or not those tensors are present in the batch (proving it
never consumed them). Ligand-only input keeps this CCD-free so it runs on plain
CI.
"""

from __future__ import annotations

from pathlib import Path

import torch

from nesso.data.featurizer import NessoFeaturizer
from nesso.data.inference import InferenceDataset, inference_collate
from nesso.data.tokenize import tokenize_structure
from nesso.data.types import Manifest, Structure, Tokenized
from nesso.data.yaml_input import parse_yaml
from nesso.model.models.nesso1 import Nesso1
from numpy.random import RandomState

REMOVED_KEYS = {"disto_target", "token_to_rep_atom"}

_LIGAND_SMILES = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
_YAML = (
    "version: 1\n"
    "sequences:\n"
    "  - ligand:\n"
    "      id: B\n"
    f'      smiles: "{_LIGAND_SMILES}"\n'
)


def _featurize(tmp_path: Path) -> dict:
    mol_dir = tmp_path / "rdkit_conformers"
    structures_dir = tmp_path / "structures"
    esm_dir = tmp_path / "esm"
    for d in (mol_dir, structures_dir, esm_dir):
        d.mkdir(parents=True, exist_ok=True)

    yaml_path = tmp_path / "lig.yaml"
    yaml_path.write_text(_YAML)
    struct, record, _, _ = parse_yaml(
        yaml_path, mol_dir, ccd_dict=None, record_id="lig"
    )
    struct.dump(structures_dir / f"{record.id}.npz")

    struct = Structure.load(structures_dir / f"{record.id}.npz")
    struct = struct.remove_invalid_chains(struct.mask.copy())
    tokens, bonds = tokenize_structure(struct)
    tokenized = Tokenized(tokens=tokens, bonds=bonds, structure=struct, record=record)

    ds = InferenceDataset(
        manifest=Manifest([record]),
        target_dir=tmp_path,
        featurizer=NessoFeaturizer(
            esm_emb_dir=esm_dir, esm_emb_dim=1280, esm_num_layers=33
        ),
        ligand_dir=mol_dir,
        ccd_pkl=None,
    )
    molecules = ds._setup_molecules(struct, str(record.id))
    torch.manual_seed(0)
    feats = ds.featurizer.process(
        tokenized,
        record=record,
        molecules=molecules,
        random=RandomState(0),
        atoms_per_window_queries=32,
        binder_pocket_conditioned_prop=0.0,
        max_tokens=None,
    )
    feats["affinity_token_mask"] = (feats["mol_type"] == 3).float()
    return inference_collate([feats])


def _tiny_model(*, affinity: bool = False) -> Nesso1:
    return Nesso1(
        atom_s=16,
        atom_z=16,
        token_s=32,
        token_z=32,
        atom_feature_dim=387,
        embedder_args={"atom_encoder_depth": 1, "atom_encoder_heads": 2},
        pairformer_model_args={"num_blocks": 1},
        esm_module_args={"esm_embed_dim": 1280},
        affinity_prediction=affinity,
        affinity_model_args={
            "pairformer_args": {"num_blocks": 1},
            "transformer_args": {},
        }
        if affinity
        else None,
        use_kernels=False,
        predict_args={
            "refine_protein_inference": False,
            "affinity_protein_cutoff": 15.0,
        },
    ).eval()


def _assert_forward_ignores_injection(batch: dict, model: Nesso1) -> None:
    n = batch["token_pad_mask"].shape[1]
    m = batch["atom_pad_mask"].shape[1]
    with torch.no_grad():
        out_lean = model(dict(batch), recycling_steps=2)
    # Re-add the removed tensors; a forward that truly ignores them is unchanged.
    injected = dict(batch)
    injected["disto_target"] = torch.randn(1, n, n, 64)
    injected["token_to_rep_atom"] = torch.zeros(1, n, m)
    with torch.no_grad():
        out_injected = model(injected, recycling_steps=2)

    assert set(out_lean) == set(out_injected)
    for key, value in out_lean.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, out_injected[key]), key


def test_featurizer_omits_training_targets(tmp_path: Path) -> None:
    batch = _featurize(tmp_path)
    keys = {k for k, v in batch.items() if isinstance(v, torch.Tensor)}
    assert REMOVED_KEYS.isdisjoint(keys), keys & REMOVED_KEYS
    # Sanity: the live features the model does read are still present.
    for live in ("res_type", "s_esm", "atom_to_token", "token_pad_mask"):
        assert live in keys, live


def test_trunk_forward_ignores_injected_targets(tmp_path: Path) -> None:
    torch.manual_seed(0)
    _assert_forward_ignores_injection(_featurize(tmp_path), _tiny_model())


def test_affinity_forward_ignores_injected_targets(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _tiny_model(affinity=True)
    _assert_forward_ignores_injection(_featurize(tmp_path), model)
