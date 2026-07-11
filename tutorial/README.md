# Tutorials

Runnable examples and helper scripts for Nesso-1. Run all commands from the repository root.

## Prediction Examples

Three YAML inputs demonstrate the supported ligand formats (SMILES, CCD, SDF). Each uses the same protein sequence with tyrosine as the ligand.

```bash
nesso predict tutorial/smiles.yaml --out_dir ./output/tutorial_smiles
nesso predict tutorial/ccd.yaml --out_dir ./output/tutorial_ccd
nesso predict tutorial/sdf.yaml --out_dir ./output/tutorial_sdf
nesso predict tutorial/multi_ligand.yaml --out_dir ./output/tutorial_multi_ligand  # two ligands; binder set in YAML
```

## Feature Extraction

`tutorial/extract_features.py` reads `predictions.safetensors` produced when running with `--save_metadata`. It slices the Pairformer `z` tensor into protein/ligand blocks and prints tensor shapes for quick inspection.

```bash
nesso predict tutorial/smiles.yaml --out_dir ./output/tutorial_smiles --save_metadata
python tutorial/extract_features.py \
  ./output/tutorial_smiles/predictions/smiles/predictions.safetensors
```

Outputs:

* `z_pp`, `z_pl`, `z_ll`: pairwise protein/ligand blocks from `z`
* `s_prot`, `s_lig`: per-token features from row means of `z`
