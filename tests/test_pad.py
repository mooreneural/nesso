"""Tests for the padding helpers (nesso/data/pad.py)."""

from __future__ import annotations

import torch

from nesso.data.pad import pad_dim, pad_to_max


def test_pad_dim_extends_and_fills():
    x = torch.ones(2, 3)
    out = pad_dim(x, dim=1, pad_len=2, value=0)
    assert out.shape == (2, 5)
    assert torch.equal(out[:, :3], x)
    assert torch.all(out[:, 3:] == 0)


def test_pad_dim_zero_is_noop():
    x = torch.ones(2, 3)
    out = pad_dim(x, dim=1, pad_len=0)
    assert out is x


def test_pad_to_max_equal_shapes_just_stacks():
    items = [torch.ones(2, 3), torch.zeros(2, 3)]
    padded, mask = pad_to_max(items)
    assert padded.shape == (2, 2, 3)
    assert mask == 0  # no padding needed


def test_pad_to_max_ragged_shapes_and_mask():
    a = torch.ones(2, 3)
    b = torch.ones(1, 5)
    padded, mask = pad_to_max([a, b], value=0)

    assert padded.shape == (2, 2, 5)
    assert mask.shape == (2, 2, 5)
    # mask marks the real (unpadded) region of each item
    assert mask[0, :2, :3].all() and not mask[0, :, 3:].any()
    assert mask[1, :1, :5].all() and not mask[1, 1:, :].any()
