"""Failure reporting of the predict CLI (nesso/main.py).

A YAML that fails to parse is dropped from the manifest. These tests check that
the drop is *reported* rather than swallowed:

1. ``preprocess_yamls`` returns the failed input stems alongside the manifest,
   so a batch can account for the invalid pairs instead of losing them
   silently.

2. When *every* input fails preprocessing the manifest is empty, and the CLI
   exits non-zero — it must not take the "nothing to predict / all outputs
   already exist" branch, which is for inputs that parsed but were already
   predicted.

Both paths run CPU-only and need a CCD pickle to build the (valid) protein
chain, so they skip without one (see the ``ccd_pkl_path`` fixture).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from nesso.main import preprocess_yamls

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

# Valid YAML, invalid schema: a ligand may have exactly one of smiles/ccd/sdf.
_BAD_YAML = textwrap.dedent(
    """\
    version: 1
    sequences:
      - ligand:
          id: B
          smiles: "C"
          ccd: "TYR"
    properties:
      - affinity:
          binder: B
    """
)


def _write(dirpath: Path, name: str, body: str) -> Path:
    p = dirpath / name
    p.write_text(body)
    return p


def test_preprocess_reports_dropped_inputs(tmp_path: Path, ccd_pkl_path: Path):
    """A mixed batch tells the caller which inputs failed to parse.

    ``preprocess_yamls`` returns ``(manifest, failed)``: the valid input is in
    the manifest, the invalid one is named in ``failed``. Without this the bad
    input vanishes with no signal (the root cause of a bogus
    "Number of failed examples: 0" on mixed batches).
    """
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _write(inputs, "good_example.yaml", _GOOD_YAML)
    _write(inputs, "bad_example.yaml", _BAD_YAML)

    mol_dir = tmp_path / "mol"  # caller-created in production (resolve_paths)
    mol_dir.mkdir()

    result = preprocess_yamls(
        sorted(inputs.glob("*.yaml")),
        mol_dir=mol_dir,
        ccd_pkl=ccd_pkl_path,
        structures_dir=tmp_path / "structures",
        records_dir=tmp_path / "records",
        num_workers=1,
    )

    # Contract: (manifest, failed_stems).
    assert isinstance(result, tuple), (
        "preprocess_yamls should report failures alongside the manifest"
    )
    manifest, failed = result
    assert {r.id for r in manifest.records} == {"good_example"}
    assert [Path(f).stem if isinstance(f, str) else f for f in failed] == [
        "bad_example"
    ]


def test_single_bad_input_is_not_reported_as_already_done(
    tmp_path: Path, ccd_pkl_path: Path
):
    """A single bad input must not be reported as "all outputs already exist".

    The whole batch fails preprocessing -> empty manifest. The CLI signals the
    failure (non-zero exit) instead of taking the "nothing to predict / all
    outputs already exist" branch, which is only for already-predicted inputs.
    """
    from click.testing import CliRunner

    from nesso.main import cli

    bad = _write(tmp_path, "bad_example.yaml", _BAD_YAML)
    ckpt = tmp_path / "dummy.ckpt"  # never loaded: we return before model load
    ckpt.touch()
    out = tmp_path / "out"

    res = CliRunner().invoke(
        cli,
        [
            "predict",
            str(bad),
            "--out_dir",
            str(out),
            "--checkpoint",
            str(ckpt),
            "--ccd",
            str(ccd_pkl_path),
            "--accelerator",
            "cpu",
            "--num_workers",
            "1",
        ],
    )

    assert "all outputs already exist" not in res.output, (
        "a failed-preprocessing input should not be reported as already predicted"
    )
    assert res.exit_code != 0, "every input failing preprocessing should be an error"
