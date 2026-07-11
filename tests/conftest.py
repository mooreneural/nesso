"""Shared fixtures for the nesso test suite.

These tests are intentionally CPU-only and require neither the model checkpoint
nor a GPU — they cover the input/output plumbing around the model (YAML parsing,
feature slicing, padding, cropping, safetensor I/O), which is where regressions
tend to hide and what every user touches first.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest

from nesso.data.yaml_input import load_ccd_mol_dict

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


def _find_ccd_pkl() -> Path | None:
    """Locate a CCD pickle without hardcoding any private path.

    Search order mirrors how the CLI resolves it: an explicit ``NESSO_CCD``,
    then ``$NESSO_CACHE/ccd.pkl`` (plus its Hugging Face cache), then the
    documented default ``./.cache/ccd.pkl`` (plus its Hugging Face cache).
    """
    candidates: list[Path] = []
    if os.environ.get("NESSO_CCD"):
        candidates.append(Path(os.environ["NESSO_CCD"]))
    cache_dirs: list[Path] = []
    if os.environ.get("NESSO_CACHE"):
        cache_dirs.append(Path(os.environ["NESSO_CACHE"]))
    cache_dirs.append(REPO_ROOT / ".cache")
    for cache_dir in cache_dirs:
        candidates.append(cache_dir / "ccd.pkl")
        candidates.extend(sorted(cache_dir.glob("huggingface/**/ccd.pkl")))
    return next((p for p in candidates if p.is_file()), None)


@pytest.fixture(scope="session")
def ccd_pkl_path() -> Path:
    """Path to a CCD pickle (not the loaded dict).

    Needed by tests that exercise ``preprocess_yamls``, which takes a pickle
    path and loads it inside worker processes. Skips when none is available.
    """
    path = _find_ccd_pkl()
    if path is None:
        pytest.skip(
            "CCD pickle not found; set NESSO_CCD or cache ccd.pkl to run "
            "protein-parsing tests"
        )
    return path


@pytest.fixture(scope="session")
def ccd_dict(ccd_pkl_path: Path) -> dict:
    """Standard-residue RDKit mols, needed to build protein chains.

    Skips when no CCD pickle is available (e.g. plain CI without the gated
    asset), so the rest of the suite still runs.
    """
    return load_ccd_mol_dict(ccd_pkl_path)


@pytest.fixture(scope="session")
def extract_features() -> ModuleType:
    """Import ``tutorial/extract_features.py`` (a script, not a package module)."""
    path = REPO_ROOT / "tutorial" / "extract_features.py"
    spec = importlib.util.spec_from_file_location(
        "nesso_tutorial_extract_features", path
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
