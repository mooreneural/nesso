"""Tests for the safetensor I/O helper (nesso/data/io.py)."""

from __future__ import annotations

import torch
from safetensors.torch import save_file

from nesso.data.io import load_safetensor


def test_load_safetensor_default_key(tmp_path):
    t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    path = tmp_path / "x.safetensors"
    save_file({"embeddings": t}, str(path))

    torch.testing.assert_close(load_safetensor(path), t)


def test_load_safetensor_all_keys(tmp_path):
    a = torch.ones(2, 2)
    b = torch.zeros(3)
    path = tmp_path / "x.safetensors"
    save_file({"a": a, "b": b}, str(path))

    out = load_safetensor(path, key=None)
    assert set(out) == {"a", "b"}
    torch.testing.assert_close(out["a"], a)
    torch.testing.assert_close(out["b"], b)
