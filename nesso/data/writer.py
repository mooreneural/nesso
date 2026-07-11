import json
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from safetensors.torch import save_file


class NessoWriter(BasePredictionWriter):
    """Writes the predict_step output of Nesso1:
    - scalars to affinity.json,
    - optionally tensors to predictions.safetensors.

    predictions.safetensors contains the following tensors:
    - z: Pairformer representation.
    - pdistogram: pdistogram tensor from the distogram head.
    - refine_mask: Sequence/complex tokens mask for refinement pass.
    - pocket_mask: Pocket token mask.
    - token_pad_mask: Original padding mask for tokens.
    - mol_type: Chain type per refined token (aligned with z).
    """

    def __init__(
        self,
        output_dir: str | Path,
        save_metadata: bool = False,
    ) -> None:
        super().__init__(write_interval="batch")
        self.save_metadata = save_metadata
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.failed = 0

    def write_on_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        prediction: dict[str, Any],
        batch_indices: list[int],  # noqa: ARG002
        batch: Any,  # noqa: ARG002
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int,  # noqa: ARG002
    ) -> None:
        exc = prediction.get("exception")
        if exc is True or (
            isinstance(exc, torch.Tensor) and exc.numel() > 0 and bool(exc.any().item())
        ):
            self.failed += 1
            return

        rid = prediction.get("record_id")
        if not rid:
            msg = "predict_step output missing record_id"
            raise ValueError(msg)

        record_dir = self.output_dir / str(rid)
        record_dir.mkdir(parents=True, exist_ok=True)

        # Extract only affinity and entropy scalars
        stats: dict[str, float] = {}
        for k, v in prediction.items():
            if (
                k.startswith("affinity_") or k.startswith("entropy_")
            ) and k != "entropy_pair":
                if isinstance(v, torch.Tensor):
                    stats[k] = float(v.item())
                elif isinstance(v, (int, float)):
                    stats[k] = float(v)

        stats_path = record_dir / "affinity.json"
        stats_path.write_text(json.dumps(stats, indent=2))

        # Save additional pairwise representations if requested
        if self.save_metadata:
            if "z_full" not in prediction:
                raise ValueError("z_full not found in prediction")
            safetensors_data = {
                "z": prediction["z_full"].detach().cpu(),
                "pdistogram": prediction["pdistogram"].squeeze(0).detach().cpu(),
                "refine_mask": prediction["refine_mask_full"].detach().cpu(),
                "pocket_mask": prediction["pocket_mask_full"].detach().cpu(),
                "token_pad_mask": prediction["token_pad_mask"]
                .squeeze(0)
                .detach()
                .cpu(),
                "mol_type": prediction["mol_type_refined"].detach().cpu(),
            }
            save_file(safetensors_data, record_dir / "predictions.safetensors")

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        print(f"Number of failed examples: {self.failed}")  # noqa: T201
