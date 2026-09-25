"""The compact retrieval bank (scripts/slim_assets.py) must give generation exactly what the LMDB
bank gives: the same keyframe batches, built by signspark_render's own keyframe_batch."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "auslan_smplx"))
R = pytest.importorskip("signspark_render")

from signchat.text2sign import STREAMS, compact_bank, expand_entry, load_compact_bank  # noqa: E402

DIMS = {"hand": 180, "body": 60, "face": 56}


def fake_bank(n=40, seed=0):
    rng = np.random.default_rng(seed)
    bank = []
    for i in range(n):
        T = int(rng.choice([1, 2, 3] + list(range(10, 200)))) if i < 6 else int(rng.integers(10, 250))
        kf = sorted({int(f) for f in rng.integers(0, T, size=int(rng.integers(1, 20)))})
        bank.append({"name": f"c{i}", "text": f"<Auslan> sentence {i} .", "T": T, "keyframes": kf,
                     **{s: rng.standard_normal((T, d)).astype(np.float32) for s, d in DIMS.items()}})
    return bank


def test_compact_bank_gives_identical_keyframe_batches(tmp_path):
    bank = fake_bank()
    path = str(tmp_path / "bank.npz")
    np.savez(path, **compact_bank(bank), emb=np.zeros((len(bank), 4), np.float32))
    small = load_compact_bank(path)
    rng = np.random.default_rng(1)
    for full, c in zip(bank, small):
        e = expand_entry(c)
        assert (e["text"], e["T"], e["keyframes"]) == (full["text"], full["T"], full["keyframes"])
        for n in (20, int(rng.integers(21, 300)), 300):
            for s in STREAMS:
                xa, ya = R.keyframe_batch(s, [full["text"]], [n], [full])
                xb, yb = R.keyframe_batch(s, [e["text"]], [n], [e])
                assert torch.equal(xa, xb) and ya["keyframes"] == yb["keyframes"]


def test_lmdb_entries_pass_through():
    e = fake_bank(1)[0]
    assert expand_entry(e) is e
