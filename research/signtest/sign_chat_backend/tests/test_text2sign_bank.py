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


def _stub_generator(mode, fail=False):
    """A SignGenerator whose model parts are stubbed: tests generate()'s file handling only."""
    import threading
    import types
    from signchat.text2sign import SignGenerator
    g = SignGenerator.__new__(SignGenerator)
    g.cfg = {"render_video": mode, "video_size": 64}
    g.lock = threading.Lock()
    T = 30
    g.features = lambda s: {"T": T, "retrieved": "hi .", "seen": False}
    g.smplx = lambda f: ({"body_pose": np.zeros((T, 63), np.float32)}, np.zeros((T, 127, 3), np.float32))
    g.skeleton = types.SimpleNamespace(parents=np.full(127, -1))

    def write_video(path, clips, titles, parents, size, caption):
        import time
        time.sleep(0.3)
        if fail:
            raise RuntimeError("ffmpeg broke")
        open(path, "wb").write(b"mp4")
    g.R = types.SimpleNamespace(FPS=25, write_video=write_video)
    return g


def test_generate_async_video_appears_after_reply(tmp_path):
    import json
    import time
    g = _stub_generator("async")
    r = g.generate("Hi.", str(tmp_path), "x")
    assert r["video_status"] == "rendering" and r["video_file"] == "x.mp4"
    assert json.load(open(tmp_path / "x.json"))["frames"] == 30
    assert not (tmp_path / "x.mp4").exists()                  # reply came back before the video
    for _ in range(50):
        if (tmp_path / "x.mp4").exists():
            break
        time.sleep(0.05)
    assert (tmp_path / "x.mp4").read_bytes() == b"mp4"
    assert not [p for p in os.listdir(tmp_path) if p.startswith(".")]   # no temporary files left


def test_generate_sync_off_and_failure(tmp_path):
    import time
    assert _stub_generator(True).generate("Hi.", str(tmp_path), "a")["video_status"] == "ready"
    assert (tmp_path / "a.mp4").exists()
    off = _stub_generator(False).generate("Hi.", str(tmp_path), "b")
    assert off["video_status"] == "off" and off["video_file"] is None and not (tmp_path / "b.mp4").exists()
    bad = _stub_generator("async", fail=True)
    assert bad.generate("Hi.", str(tmp_path), "c")["video_status"] == "rendering"
    time.sleep(0.6)
    assert not (tmp_path / "c.mp4").exists()                  # failure logged, server unaffected
