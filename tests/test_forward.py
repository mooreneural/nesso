"""Weight-free smoke test for Nesso1.forward (including affinity heads)."""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path

import torch
from safetensors.torch import save_file

from nesso.data.featurizer import NessoFeaturizer
from nesso.data.inference import InferenceDataset, inference_collate
from nesso.main import preprocess_yamls
from nesso.model.models.nesso1 import Nesso1

_PROTEIN = (
    "MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGE"
    "LARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINV"
)
_GOOD_YAML = textwrap.dedent(
    f"""\
    version: 1
    sequences:
      - protein:
          id: A
          sequence: {_PROTEIN}
      - ligand:
          id: B
          smiles: "N[C@@H](Cc1ccc(O)cc1)C(=O)O"
    properties:
      - affinity:
          binder: B
    """
)

_AFF = {"pairformer_args": {"num_blocks": 1}, "transformer_args": {}}


def test_forward_smoke(tmp_path: Path, ccd_pkl_path: Path) -> None:
    yaml_path = tmp_path / "example.yaml"
    yaml_path.write_text(_GOOD_YAML)
    processed = tmp_path / "processed"
    mol_dir = processed / "rdkit_conformers"
    structures_dir = processed / "structures"
    records_dir = processed / "records"
    esm_dir = processed / "esm_embeddings"
    mol_dir.mkdir(parents=True)
    esm_dir.mkdir()

    manifest, _ = preprocess_yamls(
        [yaml_path],
        mol_dir=mol_dir,
        ccd_pkl=ccd_pkl_path,
        structures_dir=structures_dir,
        records_dir=records_dir,
        num_workers=1,
    )
    msa_id = hashlib.md5(_PROTEIN.encode()).hexdigest()
    save_file(
        {"embeddings": torch.zeros(1, len(_PROTEIN) + 2, 1280)},
        esm_dir / f"{msa_id}.safetensors",
    )

    featurizer = NessoFeaturizer(
        esm_emb_dir=esm_dir, esm_emb_dim=1280, esm_num_layers=33
    )
    sample = InferenceDataset(
        manifest=manifest,
        target_dir=processed,
        featurizer=featurizer,
        ligand_dir=mol_dir,
        ccd_pkl=ccd_pkl_path,
    )[0]
    assert not sample.get("exception"), sample
    batch = inference_collate([sample])
    predict_args = {
        "refine_protein_inference": True,
        "refine_protein_cutoff": 22.0,
        "refine_protein_tokens_budget": 256,
        "affinity_protein_cutoff": 15.0,
        "save_metadata": False,
    }

    model = Nesso1(
        atom_s=16,
        atom_z=16,
        token_s=32,
        token_z=32,
        atom_feature_dim=387,
        embedder_args={"atom_encoder_depth": 1, "atom_encoder_heads": 2},
        pairformer_model_args={"num_blocks": 1},
        esm_module_args={"esm_embed_dim": 1280},
        affinity_prediction=True,
        affinity_model_args=_AFF,
        use_kernels=False,
        predict_args=predict_args,
    ).eval()
    out = model.forward(batch, recycling_steps=0)
    assert "pdistogram" in out
    assert "affinity_pred_value" in out
