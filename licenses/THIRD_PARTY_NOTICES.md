# Third-Party Notices

Nesso-1 — Copyright Recursion Pharmaceuticals

Licensed under Apache License 2.0 (see [../LICENSE](../LICENSE))

## Adapted source code

The following codebases contributed adapted source included in this
distribution. Copyright notices are also retained in the corresponding
source files.

- [Boltz2](https://github.com/jwohlwend/boltz) — MIT — see [boltz.LICENSE](boltz.LICENSE)
- [OpenFold](https://github.com/aqlaboratory/openfold) — Apache 2.0 — see [openfold.LICENSE](openfold.LICENSE)
- [AlphaFold3 PyTorch](https://github.com/lucidrains/alphafold3-pytorch) — MIT — see [alphafold3-pytorch.LICENSE](alphafold3-pytorch.LICENSE)
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d) — BSD 3-Clause — see [pytorch3d.LICENSE](pytorch3d.LICENSE) (rotation utilities in `nesso/model/modules/utils.py`)

## Bundled runtime components

When installed with the `[kernels]` extra or included in a bundled image:

- [cuEquivariance](https://github.com/NVIDIA/cuEquivariance) — Apache 2.0 — see [cuequivariance.LICENSE](cuequivariance.LICENSE)

## Runtime dependencies (not redistributed as source)

- ESM-2 weights loaded via HuggingFace Transformers ([facebookresearch/esm](https://github.com/facebookresearch/esm)) — MIT
