"""Tests for the YAML input contract (nesso/data/yaml_input.py).

The validation paths here are the first thing every user hits, and the
`parse_yaml` integration test exercises the whole featurization front-end
(including RDKit conformer generation) *without* loading the model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nesso.data import const
from nesso.data.yaml_input import esm_keys, parse_schema, parse_yaml, validate_schema

_PROTEIN = (
    "MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGE"
    "LARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINV"
)


def _smiles_schema() -> dict:
    return {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": _PROTEIN}},
            {"ligand": {"id": "B", "smiles": "N[C@@H](Cc1ccc(O)cc1)C(=O)O"}},
        ],
        "properties": [{"affinity": {"binder": "B"}}],
    }


# --- validation -----------------------------------------------------------


def test_rejects_wrong_version():
    with pytest.raises(ValueError, match="version: 1"):
        parse_schema({"version": 2, "sequences": []})


def test_requires_sequences_key():
    with pytest.raises(ValueError, match="sequences"):
        parse_schema({"version": 1})


def test_rejects_unknown_entity_type():
    schema = {"version": 1, "sequences": [{"rna": {"id": "A", "sequence": "ACGU"}}]}
    with pytest.raises(ValueError, match="Unsupported entity type"):
        parse_schema(schema)


def test_ligand_needs_exactly_one_input():
    none = {"version": 1, "sequences": [{"ligand": {"id": "B"}}]}
    with pytest.raises(ValueError, match="exactly one"):
        parse_schema(none)

    both = {
        "version": 1,
        "sequences": [{"ligand": {"id": "B", "smiles": "C", "ccd": "TYR"}}],
    }
    with pytest.raises(ValueError, match="exactly one"):
        parse_schema(both)


def test_rejects_binder_as_list():
    schema = _smiles_schema()
    schema["properties"] = [{"affinity": {"binder": ["B", "C"]}}]
    with pytest.raises(ValueError, match="single chain id"):
        validate_schema(schema)


def test_rejects_binder_nonexistent_chain():
    schema = _smiles_schema()
    schema["properties"] = [{"affinity": {"binder": "Z"}}]
    with pytest.raises(ValueError, match="does not match any chain id"):
        validate_schema(schema)


def test_rejects_binder_pointing_to_protein():
    schema = _smiles_schema()
    schema["properties"] = [{"affinity": {"binder": "A"}}]
    with pytest.raises(ValueError, match="refers to a protein chain"):
        validate_schema(schema)


def test_valid_single_binder_passes():
    schema = _smiles_schema()
    validate_schema(schema)


# --- happy path -----------------------------------------------------------


def test_parse_schema_smiles_builds_two_chains(ccd_dict):
    struct, rec, entity_to_seq, entity_to_esm = parse_schema(
        _smiles_schema(), record_id="unit", ccd_dict=ccd_dict
    )

    assert len(struct.chains) == 2
    mol_types = {int(c["mol_type"]) for c in struct.chains}
    assert mol_types == {
        const.chain_type_ids["PROTEIN"],
        const.chain_type_ids["NONPOLYMER"],
    }
    assert rec.id == "unit"
    assert list(entity_to_seq.values()) == [_PROTEIN]
    assert entity_to_esm == {}  # no esm path provided
    assert rec.binder_asym_id == 1


def test_esm_keys_hashes_only_proteins(ccd_dict):
    _, _, entity_to_seq, _ = parse_schema(
        _smiles_schema(), record_id="unit", ccd_dict=ccd_dict
    )
    keys = esm_keys(entity_to_seq)
    assert list(keys.values()) == [_PROTEIN]
    assert all(len(k) == 32 for k in keys)  # md5 hexdigest


# --- integration: real tutorial YAMLs through the full front-end ----------


def test_parse_yaml_tutorial_smiles(repo_root: Path, ccd_dict):
    struct, rec, _, _ = parse_yaml(
        repo_root / "tutorial" / "smiles.yaml", ccd_dict=ccd_dict
    )
    assert rec.id == "smiles"
    assert len(struct.chains) == 2
    assert len(struct.atoms) > 0


def test_parse_yaml_tutorial_sdf(repo_root: Path, ccd_dict, monkeypatch):
    # the sdf yaml references a path relative to the working directory
    monkeypatch.chdir(repo_root)
    struct, rec, _, _ = parse_yaml(Path("tutorial") / "sdf.yaml", ccd_dict=ccd_dict)
    assert rec.id == "sdf"
    assert len(struct.chains) == 2
