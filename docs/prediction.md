# Prediction

Once `nesso` is installed, run predictions with:

`nesso predict <INPUT_PATH> [OPTIONS]`

* `<INPUT_PATH>` can be a single `.yaml` / `.yml` file, or a directory (all `.yaml`/`.yml` files inside will be processed).
* On first run, Nesso-1 automatically downloads and caches the standard CCD dictionary and model checkpoint. All assets (CCD, model checkpoint, and ESM-2/HuggingFace weights) are saved under the Hugging Face cache in `.cache/huggingface/` by default — set `NESSO_CACHE` to change the base cache dir.

> [!WARNING]
> `ccd.pkl` is a publisher-trusted pickle file downloaded from the Hugging Face model repo.
> Do not load untrusted pickle files. The optional YAML `conformer:` key also accepts a
> pickle path — only use conformers from sources you trust.

## Running Predictions

```bash
nesso predict complex.yaml --out_dir /path/to/output
```

### Key Options

| Option | Default | Description |
|:---|:---:|:---|
| `DATA` | *positional* | Path to a single `.yaml`/`.yml` file or a directory of YAML files |
| `--out_dir` | `./output/` | Output directory for logs, processed features, and predictions |
| `--cache` | *(unset)* | Cache directory. When omitted, uses `$NESSO_CACHE` if set, otherwise defaults to `.cache/` |
| `--checkpoint` | None | Path to a directory containing `hparams.json` and `model.safetensors` (bypasses default cache) |
| `--model_revision` | *(unset)* | Hub revision (tag/branch/commit) to pull default weights/CCD from. Defaults to `$NESSO_MODEL_REVISION` if set, otherwise `v{installed nesso version}`. Ignored if `--checkpoint` is set |
| `--override` | off | Re-run predictions and overwrite existing outputs in the output directory |
| `--devices` | 1 | Number of devices/GPUs to use for prediction |
| `--accelerator` | `auto` | Accelerator to use: `auto`, `gpu`, or `cpu` |
| `--precision` | `bf16-mixed`| Precision for PyTorch Lightning inference. Common choices: [`32`, `bf16`, `16`]. For more options, refer to [Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html#precision) |
| `--recycling_steps` | 5 | Number of trunk recycling iterations (v10 models typically use 5) |
| `--num_workers` | 2 | Number of dataloader workers to use |
| `--require_affinity`| off | Exit if the checkpoint does not contain an affinity head |
| `--save_metadata` | off | Save the pairwise trunk representations and metadata as `predictions.safetensors` |
| `--refine_protein_inference` / `--no_refine_protein_inference` | `on` | Enable two-stage pocket cropping during inference (nesso-pocket) |
| `--refine_protein_cutoff` | `22.0` | Cutoff distance (in Å) for selecting pocket tokens in the refined pass |
| `--refine_protein_tokens_budget` | `256` | Max tokens when refine_protein_inference is on |
| `--affinity_protein_cutoff` | `15.0` | Distance cutoff (in Å) around the ligand for selecting pocket tokens |
| `--no_kernels` | off | Disable cuEquivariance GPU kernels (useful for older GPUs or CPU runs) |
| `--seed` | 42 | Random seed for reproducible predictions (`pl.seed_everything(seed, workers=True)`). |

---

## Input Format

Each complex is described by a single `.yaml` file. The input format is designed to be simple without any MSA requirements.

### Full YAML Schema

```yaml
sequences:
  - protein:
      id: CHAIN_ID             # e.g., A, or a list like [A, B] for identical chains
      sequence: SEQUENCE       # Single-letter amino acid sequence
      esm: ESM_PATH            # Optional, path to pre-computed ESM embeddings
  - ligand:
      id: CHAIN_ID             # e.g., C, or a list like [C, D] for identical molecules
      smiles: "SMILES"         # SMILES string (exclusive with ccd and sdf)
      ccd: CCD                 # CCD component code (exclusive with smiles and sdf)
      sdf: SDF_PATH            # Path to an SDF file (exclusive with smiles and ccd)
properties:
  - affinity:
      binder: CHAIN_ID         # Chain ID of the ligand to predict affinity against
```

### Ligand Input Formats

Each ligand entry must specify exactly one of `smiles`, `ccd`, or `sdf`:

| Key | Description |
|-----|-------------|
| `smiles` | SMILES string. Nesso-1 generates a 3D conformer with RDKit ETKDG. |
| `ccd` | Chemical Component Dictionary code (e.g. `TYR`, `ATP`). Loaded from the cached `ccd.pkl`. |
| `sdf` | Path to an SDF file with 3D coordinates. Relative paths are resolved from the working directory. |

### Tutorial Examples

The [`tutorial/`](../tutorial/) directory contains ready-to-run examples for each ligand format. All three use the same protein sequence and tyrosine as the ligand:

| File | Ligand input |
|------|-------------|
| [`tutorial/smiles.yaml`](../tutorial/smiles.yaml) | `smiles: 'N[C@@H](Cc1ccc(O)cc1)C(=O)O'` |
| [`tutorial/ccd.yaml`](../tutorial/ccd.yaml) | `ccd: TYR` |
| [`tutorial/sdf.yaml`](../tutorial/sdf.yaml) | `sdf: tutorial/tyrosine.sdf` |
| [`tutorial/multi_ligand.yaml`](../tutorial/multi_ligand.yaml) | Two ligands; `binder` selects which gets affinity |

Run from the repository root:

```bash
nesso predict tutorial/smiles.yaml --out_dir ./output/tutorial_smiles
nesso predict tutorial/ccd.yaml --out_dir ./output/tutorial_ccd
nesso predict tutorial/sdf.yaml --out_dir ./output/tutorial_sdf
nesso predict tutorial/multi_ligand.yaml --out_dir ./output/tutorial_multi_ligand
```

Output is written to `{out_dir}/predictions/{record_id}/affinity.json`.

### Field Descriptions

- **`sequences`**: List of protein and ligand chains in the complex.
  - **`protein`**:
    - **`id`**: Unique identifier for the protein chain(s). Multiple identical chains can be specified as a list (e.g. `id: [A, B]`).
    - **`sequence`**: Standard single-letter amino acid sequence.
    - **`esm`** *(optional)*: Path to a pre-computed ESM `.safetensors` file of shape `[num_layers, seq_len+2, embed_dim]`. If omitted, ESM-2 embeddings are generated on-the-fly and cached.
  - **`ligand`**:
    - **`id`**: Unique identifier for the ligand chain(s). Multiple identical molecules can be specified as a list (e.g. `id: [C, D]`).
    - **`smiles`**: Standard SMILES string. Nesso-1 generates 3D conformers from SMILES automatically.
    - **`ccd`**: CCD component code (e.g. `TYR`). Looked up from the cached standard CCD dictionary.
    - **`sdf`**: Path to an SDF file containing the ligand with 3D coordinates.
    - Provide exactly one of `smiles`, `ccd`, or `sdf` per ligand entry.

- **`properties`** *(optional)*: Specifies auxiliary tasks like affinity prediction.
  - **`affinity`**:
    - **`binder`**: The `id` of the ligand against which binding affinity to the protein target will be predicted.

---

## Output Structure

Predictions are written to `out_dir/predictions/{record_id}/`:

```
out_dir/
└── predictions/
    └── {record_id}/
        ├── affinity.json           # Scalar predictions and metadata
        └── predictions.safetensors # Optional tensor outputs (requires --save_metadata)
```

### Output Files

#### 1. `affinity.json`
Contains ensemble affinity scalars, binary scores, and distogram entropies:

```json
{
  "entropy_pp": 0.24,
  "entropy_pl": 0.47,
  "entropy_ll": 0.23,
  "entropy_crop_pp": 0.59,
  "entropy_crop_pl": 0.82,
  "entropy_crop_ll": 0.23,
  "affinity_pred_value": 1.25,
  "affinity_pred_value1": 1.49,
  "affinity_pred_value2": 1.01,
  "affinity_logits_binary": 0.20,
  "affinity_probability_binary": 0.55
}
```

* **`affinity_pred_value`**: Ensemble mean predicted binding affinity as $\log_{10}(\text{IC}_{50} / \mu\text{M})$. Lower values indicate stronger binding.
  * **-3.0**: $\sim$ 1 nM (strong binder)
  * **0.0**: $\sim$ 1 $\mu\text{M}$ (moderate binder)
  * **2.0**: $\sim$ 100 $\mu\text{M}$ (weak/non-binder)
* **`affinity_pred_value1` / `affinity_pred_value2`**: Per-ensemble member affinity predictions.
* **`affinity_logits_binary`**: Raw logit for binary classification (binder vs. non-binder).
* **`affinity_probability_binary`**: Sigmoid probability from `affinity_logits_binary` (0.0 to 1.0).
* **`entropy_pp` / `entropy_pl` / `entropy_ll`**: Entropies of the predicted distance distributions for protein-protein (pp), protein-ligand (pl), and ligand-ligand (ll) pairs.
* **`entropy_crop_pp` / `entropy_crop_pl` / `entropy_crop_ll`**: Entropies computed on cropped distograms (if cropping is enabled).

> [!IMPORTANT]  
> Since we always do cropping or refinement after the first recycling step (by passing `--refine_protein_inference`), we crop after the first recyling step to protein tokens in the vicinity of the ligand tokens. Therefore, the value of `entropy_crop_{pp/pl/ll}` is the entropy term thats calculated on the protein-ligand interface. This is the value users should be looking at to get a measure of model confidence on structural predictions. If the value of `entropy_crop_{pl}` is 0.0, then it means the model wasn't able to confidently place the ligand, the predictions in those cases should not be trusted.

#### 2. `predictions.safetensors` *(optional)*
Only written when `--save_metadata` is set. Contains the following tensors:
* **`z`**: Squeezed `z_full` Pairformer representation.
* **`pdistogram`**: Squeezed `pdistogram` tensor from the distogram head.
* **`refine_mask`**: Sequence/complex tokens mask for refinement pass.
* **`pocket_mask`**: Pocket token mask.
* **`token_pad_mask`**: Original padding mask for tokens.

---

## Feature Extraction

With `--save_metadata`, the `predictions.safetensors` file includes `z` and `mol_type`, which can be used to extract protein/ligand single and pairwise features. The `tutorial/extract_features.py` helper script slices `z` into protein-protein, protein-ligand, and ligand-ligand blocks and computes per-token averages.

```bash
nesso predict tutorial/smiles.yaml --out_dir ./output/tutorial_smiles --save_metadata
python tutorial/extract_features.py \
  ./output/tutorial_smiles/predictions/smiles/predictions.safetensors
```

The script prints the shapes for:

* `z_pp`, `z_pl`, `z_ll` (pairwise blocks)
* `s_prot`, `s_lig` (single-token features from row means)
