"""The avatar's stage (stage.py): transitions keep joint rotations valid and land exactly on the
poses they join; the idle loop starts at rest; the drawing has the right size."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from signchat import stage  # noqa: E402


def test_slerp_ends_and_middle():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(20, 3)), rng.normal(size=(20, 3))
    q = lambda x: stage._aa_to_quat(x)
    same = lambda x, y: np.allclose(np.abs((q(x) * q(y)).sum(-1)), 1, atol=1e-5)    # same rotation, either sign
    assert same(stage.slerp_aa(a, b, 0.0), a) and same(stage.slerp_aa(a, b, 1.0), b)
    z = np.zeros((1, 3))
    half = stage.slerp_aa(z, np.array([[0, 0, 1.0]]), 0.5)
    assert np.allclose(half, [[0, 0, 0.5]], atol=1e-6)


def test_with_transitions():
    T, a, b = 12, 8, 10
    rng = np.random.default_rng(1)
    clip = {"body_pose": rng.normal(0, 0.3, (T, 63)).astype(np.float32),
            "expression": rng.normal(size=(T, 50)).astype(np.float32),
            "betas": np.zeros((1, 10), np.float32)}                      # not per frame: left alone
    rest = {"body_pose": stage.arms_down(np.zeros((1, 63))), "expression": np.zeros((1, 50), np.float32)}
    out = stage.with_transitions(clip, rest, a, b)
    assert out["body_pose"].shape == (a + T + b, 63) and out["betas"].shape == (1, 10)
    assert np.array_equal(out["body_pose"][a:a + T], clip["body_pose"])
    w = stage.ease(a)
    assert np.all(np.diff(w) > 0) and 0 < w[0] and w[-1] < 1
    # the expression eases linearly: the first added frame is mostly rest, the last mostly the clip
    assert np.allclose(out["expression"][0], w[0] * clip["expression"][0])
    assert np.allclose(out["expression"][-1], stage.ease(b)[-1] * 0 + (1 - stage.ease(b)[-1]) * clip["expression"][-1])


def test_breathing_loop_starts_at_rest():
    rest = {"body_pose": stage.arms_down(np.zeros((1, 63)))}
    idle = stage.breathing(rest, 100)
    assert idle["body_pose"].shape == (100, 63)
    assert np.allclose(idle["body_pose"][0], rest["body_pose"][0])
    assert not np.allclose(idle["body_pose"][50], rest["body_pose"][0])


def test_stage_projection_and_drawing():
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15] + \
              [20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53]
    J = np.zeros((127, 3), np.float32)
    J[stage.NECK] = [0, 0.1, 0]
    J[stage.L_SHOULDER], J[stage.R_SHOULDER] = [0.16, 0.08, 0], [-0.16, 0.08, 0]
    st = stage.Stage(J, J[None], parents, width_m=1.1, neck_at=0.3)
    P = st.project(J)
    assert np.allclose(P[stage.NECK], [0.5, 0.3])
    assert P[stage.L_SHOULDER][0] > 0.5 > P[stage.R_SHOULDER][0]      # front view: the signer's left on the right
    img = st.draw(P, 200)
    assert img.shape == (200, 200, 3) and img.min() < 255
