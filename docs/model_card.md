---
license: apache-2.0
library_name: nesso
tags:
  - structure-based-drug-design
  - binding-affinity
  - protein-ligand
  - drug-discovery
pipeline_tag: other
---

# Model Card for Nesso-1

Nesso-1 is a fast, structure-based protein–ligand binding-affinity model. Given a
protein sequence and a ligand (SMILES / CCD code / SDF), it predicts a binding
affinity scalar along with a binder/non-binder score.

- **Developed by:** [Valence Labs](https://valencelabs.com) ([Recursion](https://recursion.com))
- **Input modality:** Protein Amino-Acid sequence + Ligand (SMILES, CCD code, or SDF)
- **License:** [Apache License 2.0](https://huggingface.co/recursionpharma/nesso/blob/main/LICENSE)
- **Paper:** [Technical Report](https://www.valencelabs.com/wp-content/uploads/2026/07/nesso1.pdf)

For full method details and evaluations, see the [Technical Report](https://www.valencelabs.com/wp-content/uploads/2026/07/nesso1.pdf) and [Github](https://github.com/recursionpharma/nesso/)

## Documentation

Install the package, run predictions, and explore examples from the GitHub repository:

- [README](https://github.com/recursionpharma/nesso) — installation, quick start, and development setup
- [docs/prediction.md](https://github.com/recursionpharma/nesso/blob/main/docs/prediction.md) — CLI options, input YAML schema, and output format
- [tutorial/](https://github.com/recursionpharma/nesso/tree/main/tutorial) — runnable YAML examples and feature extraction
- [Issues](https://github.com/recursionpharma/nesso/issues) - For questions related to weights, model code, please raise issues here.
