# Adapted from https://github.com/jwohlwend/boltz, MIT License, Copyright (c) 2024 Jeremy Wohlwend

"""Nesso main CLI: Click-based entry point for running predictions."""

from __future__ import annotations

import hashlib
import os
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version as pkg_version
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

import click
import torch
import lightning.pytorch as pl
import yaml as pyyaml
from lightning.pytorch import Trainer
from safetensors.torch import save_file
from tqdm import tqdm

from nesso.data.inference import NessoInferenceDataModule
from nesso.data.types import Manifest, Record
from nesso.data.writer import NessoWriter
from nesso.data.yaml_input import (
    DEFAULT_CONFORMER_SEED,
    parse_yaml,
    validate_schema,
)
from nesso.model.models.nesso1 import Nesso1

from nesso.data.esm import (
    DEFAULT_ESM2_MODEL,
    extract_esm_embedding,
    setup_esm_model,
)

# Default source for the model weights, config, and CCD dictionary on the
# HuggingFace Hub. Custom weights can be passed with the --checkpoint option.
HF_REPO_ID = "recursionpharma/nesso"
FALLBACK_REVISION = "main"
MODEL_WEIGHTS_NAME = "model.safetensors"
MODEL_HPARAMS_NAME = "hparams.json"
# Root query file used by the Hub for download counting (GET/HEAD on this path).
# See https://huggingface.co/docs/hub/models-download-stats
MODEL_CONFIG_NAME = "config.json"


def get_default_model_revision() -> str:
    """Default Hub revision: ``v{installed nesso version}``.

    Ties the model weights/config pin to the installed package version, so an
    ``nesso==X.Y.Z`` install always resolves the ``vX.Y.Z`` tag on the Hub
    repo by default -- avoiding silent code/weights drift from a moving
    "latest" pointer. Override with ``--model_revision`` or
    ``$NESSO_MODEL_REVISION`` (e.g. to pin an older/newer snapshot, or to
    fall back to a branch like "main" if a matching tag doesn't exist yet).
    """
    try:
        return f"v{pkg_version('nesso')}"
    except PackageNotFoundError:
        return FALLBACK_REVISION


def resolve_model_revision(explicit: str | None) -> str:
    """Resolve the Hub revision from (in priority order) an explicit value,
    the ``NESSO_MODEL_REVISION`` env var, or the installed package version.
    """
    return (
        explicit
        or os.environ.get("NESSO_MODEL_REVISION")
        or get_default_model_revision()
    )


def get_cache_path() -> Path:
    """Determine the cache path, prioritizing the NESSO_CACHE environment variable, defaulting to .cache."""
    env_cache = os.environ.get("NESSO_CACHE")
    if env_cache:
        return Path(env_cache).expanduser().resolve()
    return Path(".cache").resolve()


def ensure_cache(
    cache: Path,
    *,
    checkpoint: Path | None = None,
    ccd: Path | None = None,
    revision: str,
) -> tuple[Path, Path]:
    """Resolve CCD and model paths, downloading from the Hub when not provided.

    Explicit ``checkpoint``/``ccd`` take priority; otherwise assets are pulled
    (and cached) via ``huggingface_hub`` and the returned cache paths are used
    directly.
    """
    cache.mkdir(parents=True, exist_ok=True)
    hf_cache_dir = cache / "huggingface"

    from huggingface_hub import hf_hub_download

    if ccd is not None:
        ccd_pkl = ccd
    else:
        ccd_pkl = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename="ccd.pkl",
                revision=revision,
                cache_dir=hf_cache_dir,
            )
        )

    if checkpoint is not None:
        model_dir = checkpoint
    else:
        # Fetch the Hub query file so this load is counted in download stats.
        # Always from ``main`` so version tags need not re-copy this file.
        try:
            from huggingface_hub.errors import EntryNotFoundError

            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=MODEL_CONFIG_NAME,
                revision="main",
                cache_dir=hf_cache_dir,
            )
        except EntryNotFoundError as exc:
            warnings.warn(
                f"Could not download Hub query file {MODEL_CONFIG_NAME!r} "
                f"(download stats will not increment): {exc}",
                stacklevel=2,
            )
        weights = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=f"{revision}/{MODEL_WEIGHTS_NAME}",
                revision=revision,
                cache_dir=hf_cache_dir,
            )
        )
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"{revision}/{MODEL_HPARAMS_NAME}",
            revision=revision,
            cache_dir=hf_cache_dir,
        )
        model_dir = weights.parent

    return ccd_pkl, model_dir


_worker_ccd_dict: dict | None = None


def _init_worker(ccd_pkl: Path | None) -> None:
    global _worker_ccd_dict
    if ccd_pkl is not None:
        from nesso.data.yaml_input import load_ccd_mol_dict

        _worker_ccd_dict = load_ccd_mol_dict(ccd_pkl)


def _process_single_yaml(
    yp: Path,
    mol_dir: Path,
    structures_dir: Path,
    records_dir: Path,
    seed: int,
) -> Record:
    struct, rec, _, _ = parse_yaml(
        yp, mol_dir, ccd_dict=_worker_ccd_dict, base_seed=seed
    )
    struct.dump(structures_dir / f"{rec.id}.npz")
    rec.dump(records_dir / f"{rec.id}.json")
    return rec


def preprocess_yamls(
    yaml_paths: list[Path],
    mol_dir: Path,
    ccd_pkl: Path | None,
    structures_dir: Path,
    records_dir: Path,
    num_workers: int = 2,
    seed: int = DEFAULT_CONFORMER_SEED,
) -> tuple[Manifest, list[str]]:
    """Parse YAMLs into a Manifest, reporting which inputs failed.

    Returns ``(manifest, failed)`` where ``failed`` lists the stems of inputs
    that could not be parsed. Failures are dropped from the manifest (as
    before) but surfaced here so callers can report them instead of silently
    losing them.
    """
    structures_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []

    # Results are collected by submission index, not completion order:
    # `as_completed` yields whichever worker finishes first, so appending here
    # would make the manifest order depend on scheduling and hence on
    # `--num_workers`. Downstream that order is visible as the dataset index.
    by_index: dict[int, Record] = {}

    with ProcessPoolExecutor(
        max_workers=max(1, num_workers), initializer=_init_worker, initargs=(ccd_pkl,)
    ) as executor:
        futures = {
            executor.submit(
                _process_single_yaml, yp, mol_dir, structures_dir, records_dir, seed
            ): (i, yp)
            for i, yp in enumerate(yaml_paths)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="YAML"):
            i, yp = futures[future]
            try:
                by_index[i] = future.result()
            except Exception as e:
                failed.append(yp.stem)
                print(f"Error processing YAML {yp.name}: {e}", file=sys.stderr)

    records = [by_index[i] for i in sorted(by_index)]
    return Manifest(records), failed


def collect_esm_from_yamls(
    yaml_paths: list[Path],
) -> tuple[dict[str, str], dict[str, str]]:
    """Scan YAMLs for protein ``sequence`` / ``esm`` entries.

    Returns ``(seq_by_md5, esm_paths_by_md5)``.
    """
    seq_by_msa: dict[str, str] = {}
    esm_paths: dict[str, str] = {}
    for yp in yaml_paths:
        schema = pyyaml.safe_load(yp.read_text())
        if not isinstance(schema, dict):
            continue
        for item in schema.get("sequences", []):
            block = item.get("protein")
            if not block or "sequence" not in block:
                continue
            seq = str(block["sequence"])
            mid = hashlib.md5(seq.encode("utf-8")).hexdigest()
            seq_by_msa.setdefault(mid, seq)
            if block.get("esm"):
                esm_paths.setdefault(mid, str(block["esm"]))
    return seq_by_msa, esm_paths


def run_esm(
    seq_by_msa: dict[str, str],
    esm_out: Path,
    model_name: str,
    hf_cache: Path | None,
) -> None:
    esm_out.mkdir(parents=True, exist_ok=True)
    missing = {
        m: s
        for m, s in seq_by_msa.items()
        if not (esm_out / f"{m}.safetensors").exists()
    }
    if not missing:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = setup_esm_model(model_name, device, cache_dir=hf_cache)
    for mid, seq in tqdm(sorted(missing.items()), desc="ESM"):
        emb = extract_esm_embedding(seq, model, tokenizer)
        save_file({"embeddings": emb}, esm_out / f"{mid}.safetensors")


@dataclass
class Paths:
    processed: Path
    mol_dir: Path
    esm_dir: Path
    manifest_path: Path
    structures_dir: Path
    records_dir: Path
    predictions_dir: Path


def resolve_paths(out_dir: Path) -> Paths:
    processed = out_dir / "processed"
    mol_dir = processed / "rdkit_conformers"
    mol_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = out_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    return Paths(
        processed=processed,
        mol_dir=mol_dir,
        esm_dir=processed / "esm_embeddings",
        manifest_path=processed / "manifest.json",
        structures_dir=processed / "structures",
        records_dir=processed / "records",
        predictions_dir=predictions_dir,
    )


def filter_records(
    records: list[Record], predictions_dir: Path, override: bool
) -> list[Record]:
    """Filters records that already have predictions stored."""
    if override:
        return records

    return [
        r for r in records if not (predictions_dir / r.id / "affinity.json").exists()
    ]


def check_inputs(data: Path) -> list[Path]:
    """Check the input data and return list of YAML files."""
    if data.is_dir():
        yaml_paths = sorted(
            q for q in data.iterdir() if q.suffix.lower() in (".yaml", ".yml")
        )
        if not yaml_paths:
            raise RuntimeError(f"No .yaml/.yml files found in directory: {data}")
        return yaml_paths
    else:
        if data.suffix.lower() not in (".yaml", ".yml"):
            raise RuntimeError(
                f"Unable to parse filetype {data.suffix}, please provide a .yaml or .yml file."
            )
        return [data]


@click.group()
def cli() -> None:
    """Nesso: fast binding affinity prediction from protein sequence and ligand SMILES."""
    pass


@cli.command()
@click.argument("data", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--out_dir",
    type=click.Path(path_type=Path),
    default=Path("./output/"),
    help="The path where to save the predictions. Default is ./output/",
)
@click.option(
    "--cache",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Cache directory for downloaded assets (CCD, checkpoint, HuggingFace weights). "
        "When omitted, uses $NESSO_CACHE if set, otherwise defaults to .cache/."
    ),
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help=(
        "Path to a directory containing model.safetensors and hparams.json. "
        "Uses the cached default model by default."
    ),
)
@click.option(
    "--ccd",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to a pre-downloaded CCD pickle file. When provided together with --checkpoint, skips HuggingFace downloads.",
)
@click.option(
    "--model_revision",
    type=str,
    default=None,
    help=(
        "Hub revision (tag/branch/commit) to pull default weights/CCD from. "
        "Defaults to $NESSO_MODEL_REVISION if set, otherwise 'v{installed nesso version}'. "
        "Ignored if --checkpoint is provided."
    ),
)
@click.option(
    "--override",
    is_flag=True,
    help="Whether to override existing predictions.",
)
@click.option(
    "--devices",
    type=int,
    default=1,
    help="The number of devices to use for prediction. Default is 1.",
)
@click.option(
    "--accelerator",
    type=click.Choice(["gpu", "cpu", "auto"]),
    default="auto",
    help="The accelerator to use for prediction. Default is auto.",
)
@click.option(
    "--precision",
    type=str,
    default="bf16-mixed",
    help="The precision to use for prediction. Default is bf16-mixed.",
)
@click.option(
    "--recycling_steps",
    type=int,
    default=5,
    help="The number of recycling steps to use for prediction. Default is 5.",
)
@click.option(
    "--num_workers",
    type=int,
    default=2,
    help="The number of dataloader workers to use. Default is 2.",
)
@click.option(
    "--require_affinity",
    is_flag=True,
    help="Exit if checkpoint lacks an affinity head.",
)
@click.option(
    "--save_metadata",
    is_flag=True,
    help="Save pairwise trunk representations and metadata as predictions.safetensors.",
)
@click.option(
    "--refine_protein_inference/--no_refine_protein_inference",
    default=True,
    help="Two-stage pocket cropping (nesso-pocket). Enabled by default.",
)
@click.option(
    "--refine_protein_cutoff",
    type=float,
    default=22.0,
    help="Cutoff distance (in Å) for selecting pocket tokens in the refined pass. Default is 22.0.",
)
@click.option(
    "--refine_protein_tokens_budget",
    type=int,
    default=256,
    help="Max tokens when refine_protein_inference is on. Default is 256.",
)
@click.option(
    "--affinity_protein_cutoff",
    type=float,
    default=15.0,
    help="Distance cutoff (in Å) around the ligand for selecting pocket tokens.",
)
@click.option(
    "--no_kernels",
    is_flag=True,
    help="Whether to disable cuEquivariance kernels.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducible predictions. When omitted, no seed is set.",
)
def predict(
    data: Path,
    out_dir: Path,
    cache: Path | None,
    checkpoint: Path | None,
    ccd: Path | None,
    model_revision: str | None,
    override: bool,
    devices: int,
    accelerator: str,
    precision: str,
    recycling_steps: int,
    num_workers: int,
    require_affinity: bool,
    save_metadata: bool,
    refine_protein_inference: bool,
    refine_protein_cutoff: float,
    refine_protein_tokens_budget: int,
    affinity_protein_cutoff: float,
    no_kernels: bool,
    seed: int,
) -> None:
    """Run predictions using Nesso."""
    pl.seed_everything(seed, workers=True)

    for _, val, default, msg in [
        (
            "refine_protein_inference",
            refine_protein_inference,
            True,
            "refine_protein_inference is set to False, which deviates from the default setting (True).",
        ),
        (
            "refine_protein_cutoff",
            refine_protein_cutoff,
            22.0,
            f"refine_protein_cutoff is set to {refine_protein_cutoff}, which deviates from the default setting (22.0).",
        ),
        (
            "refine_protein_tokens_budget",
            refine_protein_tokens_budget,
            256,
            f"refine_protein_tokens_budget is set to {refine_protein_tokens_budget}, which deviates from the default setting (256).",
        ),
    ]:
        if val != default:
            warnings.warn(msg, UserWarning, stacklevel=2)

    warnings.filterwarnings(
        "ignore", ".*that has Tensor Cores. To properly utilize them.*"
    )
    torch.set_float32_matmul_precision("highest")

    cache = get_cache_path() if cache is None else cache.expanduser().resolve()

    # Avoid overriding user-configured HuggingFace cache locations.
    hf_cache_dir = str(cache / "huggingface")
    os.environ.setdefault("HF_HOME", hf_cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", hf_cache_dir)

    try:
        revision = resolve_model_revision(model_revision)
        ccd_pkl, checkpoint_path = ensure_cache(
            cache, checkpoint=checkpoint, ccd=ccd, revision=revision
        )
        yaml_paths = check_inputs(data)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    paths = resolve_paths(out_dir)

    # Validate input schemas up front (per-record)
    valid_yaml_paths: list[Path] = []
    invalid_schemas: list[str] = []
    for yp in yaml_paths:
        try:
            validate_schema(pyyaml.safe_load(yp.read_text()))
        except Exception as e:
            invalid_schemas.append(yp.stem)
            click.echo(f"Invalid input {yp.name}: {e}", err=True)
        else:
            valid_yaml_paths.append(yp)

    if invalid_schemas:
        click.echo(
            f"Skipping {len(invalid_schemas)} of {len(yaml_paths)} input(s) with "
            f"invalid schema: {', '.join(sorted(invalid_schemas))}",
            err=True,
        )
    if not valid_yaml_paths:
        click.echo(
            "Error: no valid inputs to predict; all inputs had invalid schemas.",
            err=True,
        )
        sys.exit(1)

    # 2. Preprocess YAMLs to Manifest
    click.echo("Preprocessing inputs...")
    manifest, failed_preprocessing = preprocess_yamls(
        valid_yaml_paths,
        paths.mol_dir,
        ccd_pkl,
        paths.structures_dir,
        paths.records_dir,
        num_workers=num_workers,
        seed=seed,
    )
    manifest.dump(paths.manifest_path)
    if failed_preprocessing:
        click.echo(
            f"Failed to preprocess {len(failed_preprocessing)} of "
            f"{len(valid_yaml_paths)} input(s): {', '.join(sorted(failed_preprocessing))}",
            err=True,
        )
    if not manifest.records:
        click.echo(
            "Error: no inputs could be preprocessed; nothing to predict.", err=True
        )
        sys.exit(1)

    # 3. Run ESM embedding generation
    click.echo("Generating ESM embeddings...")
    seq_by_msa, esm_paths_by_msa = collect_esm_from_yamls(yaml_paths)
    paths.esm_dir.mkdir(parents=True, exist_ok=True)
    for msa_id, src in esm_paths_by_msa.items():
        dst = paths.esm_dir / f"{msa_id}.safetensors"
        if not dst.exists() and Path(src).exists():
            try:
                os.symlink(Path(src).resolve(), dst)
            except Exception:
                pass
    run_esm(seq_by_msa, paths.esm_dir, DEFAULT_ESM2_MODEL, cache / "huggingface")

    # 4. Filter records
    records = filter_records(manifest.records, paths.predictions_dir, override=override)
    if not records:
        click.echo(
            f"Nothing to predict (all outputs already exist in {paths.predictions_dir}). Use --override to re-run."
        )
        return

    # 5. Load model
    click.echo(f"Loading model weights from {checkpoint_path}...")
    try:
        model_module = Nesso1.from_pretrained(checkpoint_path)
    except Exception as e:
        click.echo(f"Error loading checkpoint: {e}", err=True)
        sys.exit(1)

    if require_affinity and not model_module.affinity_prediction:
        click.echo(
            "Error: Checkpoint lacks affinity head (affinity_prediction=False).",
            err=True,
        )
        sys.exit(1)

    # 6. Apply no_kernels if requested
    if no_kernels:
        model_module.use_kernels = False

    model_module.predict_args.update(
        {
            "pose_protein_cutoff": 15.0,
            "recycling_steps": recycling_steps,
            "affinity_protein_cutoff": affinity_protein_cutoff,
            "refine_protein_inference": refine_protein_inference,
            "refine_protein_cutoff": refine_protein_cutoff,
            "refine_protein_tokens_budget": refine_protein_tokens_budget,
            "save_metadata": save_metadata,
        }
    )
    model_module.eval()

    # 7. Setup DataModule & Trainer
    precision_to_use = (
        "32" if accelerator == "cpu" and precision == "bf16-mixed" else precision
    )
    torch.set_grad_enabled(False)

    datamodule = NessoInferenceDataModule(
        manifest=replace(manifest, records=records),
        target_dir=paths.processed,
        esm_emb_dir=paths.esm_dir,
        ligand_dir=paths.mol_dir,
        ccd_pkl=ccd_pkl,
        num_workers=num_workers,
        use_esm_all_layers=False,
        esm_emb_dim=1280,
        esm_num_layers=33,
    )

    writer = NessoWriter(
        output_dir=paths.predictions_dir,
        save_metadata=save_metadata,
    )

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision_to_use,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        callbacks=[writer],
    )

    click.echo("Running predictions...")
    trainer.predict(model_module, datamodule=datamodule, return_predictions=False)
    click.echo(f"Done! Predictions saved to {paths.predictions_dir}")

    total_failed = len(invalid_schemas) + len(failed_preprocessing) + writer.failed
    if total_failed:
        click.echo(
            f"Failed examples: {total_failed} "
            f"({len(invalid_schemas)} invalid schema, "
            f"{len(failed_preprocessing)} preprocessing, {writer.failed} prediction).",
            err=True,
        )


if __name__ == "__main__":
    cli()
