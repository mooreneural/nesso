"""Nesso1: fast binding affinity model — inference only."""

import gc
import json
from pathlib import Path
from typing import Any, Optional

import torch
import torch._dynamo
from lightning.pytorch import LightningModule
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.nn import Linear, LayerNorm

from nesso.data import const
from nesso.data.crop import select_pocket_token_indices
from nesso.model.layers import initialize
from nesso.model.inference.distogram import (
    compute_expected_distance,
    compute_distogram_entropy,
)
from nesso.model.modules.affinity import AffinityModule
from nesso.model.modules.encoders import RelativePositionEncoder
from nesso.model.layers.pairformer import PairformerNoSeqModule
from nesso.model.modules.esm_module import ESMModule
from nesso.model.modules.trunk import InputEmbedder

HPARAMS_NAME = "hparams.json"
WEIGHTS_NAME = "model.safetensors"


class Nesso1(LightningModule):
    """Nesso-1: inference-only structure and affinity predictor."""

    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        embedder_args: dict[str, Any],
        atom_feature_dim: int = 128,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        predict_args: Optional[dict[str, Any]] = None,
        num_dist_bins: int = 64,
        max_dist: float = 22.0,
        pairformer_model_args: Optional[dict[str, Any]] = None,
        use_kernels: bool = False,
        esm_module_args: Optional[dict[str, Any]] = None,
        affinity_prediction: bool = False,
        affinity_model_args: Optional[dict[str, Any]] = None,
        affinity_model_args2: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.predict_args = predict_args
        self.affinity_prediction = affinity_prediction

        self.input_embedder = InputEmbedder(
            atom_s,
            atom_z,
            token_s,
            token_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_feature_dim=atom_feature_dim,
            **embedder_args,
        )

        self.use_kernels = use_kernels

        self.rel_pos = RelativePositionEncoder(token_z)

        self.z_init_1 = Linear(token_s, token_z, bias=False)
        self.z_init_2 = Linear(token_s, token_z, bias=False)
        self.token_bonds = Linear(1, token_z, bias=False)
        self.token_bonds_type = torch.nn.Embedding(len(const.bond_types) + 1, token_z)

        initialize.gating_init_(self.z_init_1.weight)
        initialize.gating_init_(self.z_init_2.weight)
        initialize.gating_init_(self.token_bonds.weight)
        initialize.gating_init_(self.token_bonds_type.weight)

        self.z_norm = LayerNorm(token_z)
        self.z_recycle = Linear(token_z, token_z, bias=False)
        initialize.gating_init_(self.z_recycle.weight)

        esm_kw = dict(esm_module_args or {})
        self.esm_module = ESMModule(token_s=token_s, token_z=token_z, **esm_kw)

        torch._dynamo.config.cache_size_limit = 512
        torch._dynamo.config.accumulated_cache_size_limit = 512

        self.pairformer_module = PairformerNoSeqModule(
            token_z=token_z, **pairformer_model_args
        )

        # Distogram head (kept for pocket masks and entropy; no loss at inference)
        self.num_dist_bins = num_dist_bins
        self.max_dist = max_dist
        self.distogram_head = Linear(token_z, num_dist_bins)

        if self.affinity_prediction:
            self.affinity_module = AffinityModule(
                token_s=token_s, token_z=token_z, **(affinity_model_args or {})
            )
            self.affinity_module2 = AffinityModule(
                token_s=token_s,
                token_z=token_z,
                **(affinity_model_args2 or affinity_model_args or {}),
            )

    def get_pocket_mask(
        self, z_exp: Tensor, feats: dict[str, Any], cutoff: float = 15.0
    ) -> Tensor:
        mol_type = feats["mol_type"].long()
        is_protein = mol_type == const.chain_type_ids["PROTEIN"]
        is_ligand = mol_type == const.chain_type_ids["NONPOLYMER"]
        valid = feats["token_pad_mask"].bool()

        has_lig = is_ligand.any(dim=-1, keepdim=True)
        large = torch.full((), 1e6, device=z_exp.device, dtype=z_exp.dtype)
        dist_to_lig = torch.where(is_ligand[:, None, :], z_exp, large)
        min_to_ligand = dist_to_lig.min(dim=-1).values
        near = min_to_ligand <= cutoff
        pocket_with_lig = (is_ligand | (is_protein & near)) & valid
        pocket = torch.where(has_lig.expand(-1, z_exp.shape[1]), pocket_with_lig, valid)
        return pocket.float()

    @staticmethod
    def _crop_feats_by_indices(
        feats: dict[str, Any], mask: Tensor, keys: tuple[str, ...]
    ) -> dict[str, Any]:
        new = dict(feats)
        for k in keys:
            v = feats.get(k)
            if isinstance(v, Tensor):
                new[k] = v[:, mask].contiguous()
        return new

    @torch.no_grad()
    def _select_pocket_indices(
        self,
        pdistogram: Tensor,
        feats: dict[str, Any],
        *,
        cutoff: float,
        max_tokens: int,
    ) -> Tensor:
        d_exp = compute_expected_distance(
            pdistogram, min_dist=2.0, max_dist=self.max_dist
        )
        mol_type = feats["mol_type"][0]
        pad = feats["token_pad_mask"][0].bool()
        prot_pos = torch.where((mol_type == const.chain_type_ids["PROTEIN"]) & pad)[0]
        lig_pos = torch.where((mol_type == const.chain_type_ids["NONPOLYMER"]) & pad)[0]
        if prot_pos.numel() == 0 or lig_pos.numel() == 0:
            raise ValueError("No protein or ligand tokens found in the batch")

        min_d = d_exp[0][prot_pos[:, None], lig_pos[None, :]].min(dim=1).values
        asym_np = feats["asym_id"][0].detach().cpu().numpy()
        res_np = feats["residue_index"][0].detach().cpu().numpy()
        prot_np = prot_pos.detach().cpu().numpy()

        keep_list = select_pocket_token_indices(
            protein_token_indices=prot_np,
            protein_asym_ids=asym_np[prot_np],
            protein_res_idxs=res_np[prot_np],
            dists_to_ligand=min_d.float().detach().cpu().numpy(),
            ligand_token_indices=lig_pos.detach().cpu().numpy(),
            max_tokens=max_tokens,
            threshold_distance=cutoff,
        )
        return torch.tensor(keep_list, device=pdistogram.device, dtype=torch.long)

    @torch.no_grad()
    def pocket_crop(self, z: Tensor, feats: dict[str, Any]) -> tuple[Tensor, Tensor]:
        pa = self.predict_args or {}
        threshold = pa.get("refine_protein_cutoff", 22.0)
        budget = pa.get("refine_protein_tokens_budget", 196)

        z_sym_i = z + z.transpose(1, 2)
        pdistogram_full = self.distogram_head(z_sym_i).detach().clone()
        keep_t = self._select_pocket_indices(
            pdistogram_full, feats, cutoff=threshold, max_tokens=budget
        )
        return keep_t, pdistogram_full

    def forward(
        self,
        feats: dict[str, Tensor],
        recycling_steps: int = 0,
        refine_protein_inference: bool = False,
    ) -> dict[str, Tensor]:
        dict_out: dict[str, Tensor] = {}
        pdistogram_local: Optional[Tensor] = None

        s_inputs = self.input_embedder(feats)
        relative_position_encoding = self.rel_pos(feats)

        z_init = (
            self.z_init_1(s_inputs)[:, :, None] + self.z_init_2(s_inputs)[:, None, :]
        )
        z_init = z_init + relative_position_encoding
        z_init = z_init + self.token_bonds(feats["token_bonds"].float())
        z_init = z_init + self.token_bonds_type(feats["type_bonds"].long())

        z = torch.zeros_like(z_init)
        mask = feats["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]
        s_esm = feats["s_esm"]

        pdistogram_full: Optional[Tensor] = None
        keep_for_merge: Optional[Tensor] = None
        refine_pocket = (
            refine_protein_inference and recycling_steps >= 1 and not self.training
        )

        for i in range(recycling_steps + 1):
            z = z_init + self.z_recycle(self.z_norm(z))
            z = self.esm_module(
                z,
                s_inputs=s_inputs,
                s_esm=s_esm,
                pair_mask=pair_mask,
                use_kernels=self.use_kernels,
            )
            z = self.pairformer_module(
                z, pair_mask=pair_mask, use_kernels=self.use_kernels
            )

            if refine_pocket and i == 0:
                keep_t, pdistogram_full = self.pocket_crop(z, feats=feats)
                keep_for_merge = keep_t
                if keep_for_merge is not None:
                    dict_out["keep_indices"] = keep_for_merge
                dict_out["crop_protein_tokens"] = keep_t

                z = z[:, keep_t][:, :, keep_t].contiguous()
                z_init = z_init[:, keep_t][:, :, keep_t].contiguous()
                s_inputs = s_inputs[:, keep_t].contiguous()

                feats = self._crop_feats_by_indices(
                    feats,
                    mask=keep_t,
                    keys=(
                        "mol_type",
                        "token_pad_mask",
                        "s_esm",
                        "asym_id",
                        "residue_index",
                        "affinity_token_mask",
                    ),
                )
                s_esm = feats["s_esm"]
                mask = feats["token_pad_mask"].float()
                pair_mask = mask[:, :, None] * mask[:, None, :]

        z_sym = z + z.transpose(1, 2)
        pdistogram_crop = self.distogram_head(z_sym)
        pdistogram_local = pdistogram_crop

        if pdistogram_full is not None and keep_for_merge is not None:
            km = keep_for_merge
            pdistogram_full[0, km[:, None], km[None, :], :] = pdistogram_crop[0]
            dict_out["pdistogram"] = pdistogram_full
        else:
            dict_out["pdistogram"] = pdistogram_crop

        dict_out["z"] = z

        if self.affinity_prediction:
            pa = self.predict_args or {}
            cutoff = pa.get("affinity_protein_cutoff", 15.0)
            z_exp_local = compute_expected_distance(
                pdistogram_local, min_dist=2.0, max_dist=self.max_dist
            )
            pocket_mask_bool = self.get_pocket_mask(z_exp_local, feats, cutoff=cutoff)[
                0
            ].bool()
            keep_t = pocket_mask_bool.nonzero(as_tuple=True)[0]
            dict_out["affinity_keep_indices"] = keep_t

            s_aff = s_inputs[:, keep_t].contiguous()
            z_aff = z[:, keep_t][:, :, keep_t].contiguous()
            pdist_logits = pdistogram_local[:, keep_t][:, :, keep_t].contiguous()
            with torch.no_grad():
                pdistogram_aff = pdist_logits.softmax(dim=-1)

            feats_aff = self._crop_feats_by_indices(
                feats,
                mask=keep_t,
                keys=(
                    "mol_type",
                    "token_pad_mask",
                    "s_esm",
                    "affinity_token_mask",
                    "asym_id",
                    "residue_index",
                ),
            )

            with torch.autocast("cuda", enabled=False):
                dict_out_affinity = self.affinity_module(
                    s_inputs=s_aff.detach(),
                    z=z_aff.detach(),
                    pdistogram=pdistogram_aff.detach(),
                    feats=feats_aff,
                    use_kernels=self.use_kernels,
                )
            dict_out.update(dict_out_affinity)

            with torch.autocast("cuda", enabled=False):
                dict_out_affinity2 = self.affinity_module2(
                    s_inputs=s_aff.detach(),
                    z=z_aff.detach(),
                    pdistogram=pdistogram_aff.detach(),
                    feats=feats_aff,
                    use_kernels=self.use_kernels,
                )
            dict_out["affinity_pred_value1"] = dict_out["affinity_pred_value"]
            dict_out["affinity_pred_value2"] = dict_out_affinity2["affinity_pred_value"]
            dict_out["affinity_pred_value"] = (
                dict_out["affinity_pred_value1"] + dict_out["affinity_pred_value2"]
            ) / 2.0

            p1 = torch.sigmoid(dict_out["affinity_logits_binary"])
            p2 = torch.sigmoid(dict_out_affinity2["affinity_logits_binary"])
            dict_out["affinity_probability_binary"] = (p1 + p2) / 2.0
            dict_out["affinity_logits_binary"] = torch.logit(
                dict_out["affinity_probability_binary"].clamp(1e-6, 1 - 1e-6)
            )

        return dict_out

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> dict:
        try:
            if batch.get("exception"):
                exc = batch["exception"]
                if isinstance(exc, list) and exc[0]:
                    return {
                        "exception": True,
                        "record_id": batch.get("record_id", ["unknown"])[0],
                    }
                elif exc is True:
                    return {
                        "exception": True,
                        "record_id": batch.get("record_id", "unknown"),
                    }

            predict_args = self.predict_args or {}
            recycling_steps = predict_args.get("recycling_steps", 3)
            refine_prot_inference = predict_args.get("refine_protein_inference", False)

            out = self(
                batch,
                recycling_steps=recycling_steps,
                refine_protein_inference=refine_prot_inference,
            )
            pred_dict: dict[str, Any] = {"exception": False}

            rec = batch.get("record")
            if rec is not None:
                r0 = rec[0] if isinstance(rec, list) else rec
                pred_dict["record_id"] = r0.id

            pred_dict["pdistogram"] = out["pdistogram"]
            pred_dict["token_pad_mask"] = batch["token_pad_mask"]

            z_exp = compute_expected_distance(
                out["pdistogram"], min_dist=2.0, max_dist=self.max_dist
            )
            protein_cutoff = predict_args.get("pose_protein_cutoff", 15.0)
            pocket_mask = self.get_pocket_mask(z_exp, batch, cutoff=protein_cutoff)

            entropy_out = compute_distogram_entropy(out["pdistogram"], batch)
            pred_dict.update(entropy_out)

            disto_crop = batch["token_pad_mask"].float() * pocket_mask
            entropy_crop = compute_distogram_entropy(
                out["pdistogram"], batch, custom_mask=disto_crop
            )
            pred_dict["entropy_crop_pp"] = entropy_crop["entropy_pp"]
            pred_dict["entropy_crop_pl"] = entropy_crop["entropy_pl"]
            pred_dict["entropy_crop_ll"] = entropy_crop["entropy_ll"]

            pred_dict["pocket_mask"] = pocket_mask
            pred_dict["token_mask"] = batch["token_pad_mask"].float() * pocket_mask

            if self.affinity_prediction and "affinity_pred_value" in out:
                pred_dict["affinity_pred_value"] = out["affinity_pred_value"].squeeze(
                    -1
                )
                if "affinity_pred_value1" in out:
                    pred_dict["affinity_pred_value1"] = out[
                        "affinity_pred_value1"
                    ].squeeze(-1)
                    pred_dict["affinity_pred_value2"] = out[
                        "affinity_pred_value2"
                    ].squeeze(-1)
                pred_dict["affinity_logits_binary"] = out[
                    "affinity_logits_binary"
                ].squeeze(-1)
                pred_dict["affinity_probability_binary"] = out[
                    "affinity_probability_binary"
                ].squeeze(-1)

            if predict_args.get("save_metadata", False):
                N = batch["token_pad_mask"].shape[1]
                device = out["z"].device
                refine_mask_full = torch.zeros(N, dtype=torch.bool, device=device)
                if "keep_indices" in out:
                    refine_mask_full[out["keep_indices"]] = True
                else:
                    refine_mask_full = batch["token_pad_mask"][0].bool().clone()
                pocket_mask_full = pocket_mask[0].bool() & refine_mask_full
                z_exp_refined = z_exp[0, refine_mask_full][:, refine_mask_full]
                pred_dict["z_full"] = out["z"][0].to(torch.bfloat16)
                pred_dict["expected_distances_full"] = z_exp_refined
                pred_dict["refine_mask_full"] = refine_mask_full
                pred_dict["pocket_mask_full"] = pocket_mask_full
                pred_dict["mol_type_refined"] = batch["mol_type"][0][refine_mask_full]

            return pred_dict

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("WARNING: ran out of memory, skipping batch")
                torch.cuda.empty_cache()
                gc.collect()
                return {"exception": True}
            raise

    def configure_optimizers(self):  # type: ignore[override]
        raise NotImplementedError(
            "configure_optimizers is not available in the inference-only mode."
        )

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save hyperparameters and weights as ``hparams.json`` + ``model.safetensors``."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        hparams = {k: v for k, v in self.hparams.items()}
        with (save_directory / HPARAMS_NAME).open("w", encoding="utf-8") as handle:
            json.dump(hparams, handle, indent=2)

        save_file(self.state_dict(), str(save_directory / WEIGHTS_NAME))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        revision: Optional[str] = None,
        cache_dir: str | Path | None = None,
        map_location: str = "cpu",
    ) -> "Nesso1":
        """Load a model from a local directory or a Hugging Face Hub repo id.

        ``pretrained_model_name_or_path`` may be either a local directory
        containing ``hparams.json`` + ``model.safetensors``, or a Hub repo id
        (e.g. ``"recursionpharma/nesso"``), in which case both files are
        downloaded (and cached) via ``huggingface_hub`` from
        ``{revision}/hparams.json`` and ``{revision}/model.safetensors``.
        """
        local_dir = Path(pretrained_model_name_or_path)
        if local_dir.is_dir():
            hparams_path = local_dir / HPARAMS_NAME
            weights_path = local_dir / WEIGHTS_NAME
        else:
            from importlib.metadata import PackageNotFoundError, version as pkg_version

            from huggingface_hub import hf_hub_download

            hub_revision = revision
            if hub_revision is None:
                try:
                    hub_revision = f"v{pkg_version('nesso')}"
                except PackageNotFoundError:
                    hub_revision = "main"

            hparams_path = Path(
                hf_hub_download(
                    repo_id=str(pretrained_model_name_or_path),
                    filename=f"{hub_revision}/{HPARAMS_NAME}",
                    revision=hub_revision,
                    cache_dir=cache_dir,
                )
            )
            weights_path = Path(
                hf_hub_download(
                    repo_id=str(pretrained_model_name_or_path),
                    filename=f"{hub_revision}/{WEIGHTS_NAME}",
                    revision=hub_revision,
                    cache_dir=cache_dir,
                )
            )

        if not hparams_path.exists():
            raise FileNotFoundError(f"Missing hyperparameters file: {hparams_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing weights file: {weights_path}")

        with hparams_path.open("r", encoding="utf-8") as handle:
            hparams = json.load(handle)

        model = cls(**hparams)
        state_dict = load_file(str(weights_path), device=map_location)
        model.load_state_dict(state_dict, strict=True)
        return model

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int = 0
    ) -> Any:
        if isinstance(batch, dict):
            return {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
