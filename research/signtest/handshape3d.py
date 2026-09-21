#!/usr/bin/env python3
"""Canonicalise 3D hand poses so that only finger configuration remains.

A handshape is the configuration of the fingers relative to the palm. Where the
hand is in space, and which way it points, are separate parameters of a sign.
Clustering handshapes therefore means removing position, size and orientation
first.

In 2D this is impossible, and the earlier attempt failed on exactly that: only
in-plane rotation can be undone, so rolling the hand about its own axis (palm
towards the camera vs edge-on) changes the projected shape completely. Measured
on 2D landmarks, the same held handshape inside a single clip was assigned a
median of 3 different clusters, and only 9.1% of holds stayed in one.

World landmarks are metric 3D, so the full rotation can be removed by building
a frame from the palm itself. That is the whole reason for re-extracting.
"""

import numpy as np

# MediaPipe hand landmark indices
WRIST, THUMB_CMC = 0, 1
INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 5, 9, 13, 17
TIPS = [4, 8, 12, 16, 20]


def _unit(v, axis=-1, eps=1e-9):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def canonicalise(hand):
    """(N, 21, 3) metric world landmarks -> (N, 21, 3) in a palm-fixed frame.

    Frame construction, from the palm rather than from any single finger, so a
    curled finger cannot rotate the whole hand:

        origin  wrist
        y       wrist -> middle-finger MCP    ("up" the hand)
        x       pinky MCP -> index MCP, orthogonalised against y ("across")
        z       y x x                          (palm normal)

    Scale is the wrist-to-middle-MCP length, which is a bone and so is stable
    whatever the fingers are doing.

    Returns NaN rows where the frame is degenerate (a hand seen so edge-on that
    the palm axes are collinear) rather than silently emitting garbage.
    """
    h = np.asarray(hand, np.float64)
    out = np.full(h.shape, np.nan)
    ok = np.isfinite(h).all(axis=(1, 2))
    if not ok.any():
        return out

    p = h[ok] - h[ok][:, WRIST:WRIST + 1, :]          # wrist at origin

    y = p[:, MIDDLE_MCP]                               # (n,3)
    scale = np.linalg.norm(y, axis=1)
    good = scale > 1e-6
    if not good.any():
        return out

    p, y, scale = p[good], y[good], scale[good]
    p = p / scale[:, None, None]
    y = _unit(y)

    across = p[:, INDEX_MCP] - p[:, PINKY_MCP]
    x = across - (across * y).sum(1, keepdims=True) * y    # orthogonalise
    nx = np.linalg.norm(x, axis=1)
    solid = nx > 1e-3                                      # palm not collinear
    if not solid.any():
        return out

    p, y, x = p[solid], y[solid], x[solid] / nx[solid, None]
    z = np.cross(x, y)
    R = np.stack([x, y, z], axis=1)                        # world -> palm frame
    canon = np.einsum("nij,nkj->nki", R, p)

    idx = np.flatnonzero(ok)[good][solid]
    out[idx] = canon
    return out


def features(hand):
    """(N,21,3) -> (N,63) flat canonical coordinates, NaN rows dropped by caller."""
    c = canonicalise(hand)
    return c.reshape(len(c), -1)


def stability(feat, keep=0.4):
    """Boolean mask of the most static frames -- the holds rather than the
    transitions between them. A handshape is a held configuration; the frames
    in between are not handshapes and should not become their own clusters."""
    ok = np.isfinite(feat).all(1)
    v = np.full(len(feat), np.inf)
    idx = np.flatnonzero(ok)
    if len(idx) < 2:
        return np.zeros(len(feat), bool)
    d = np.linalg.norm(np.diff(feat[idx], axis=0), axis=1)
    v[idx] = np.r_[d, d[-1]]
    thr = np.percentile(v[idx], keep * 100)
    return ok & (v <= thr)
