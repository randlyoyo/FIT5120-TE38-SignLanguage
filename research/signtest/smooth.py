#!/usr/bin/env python3
"""Temporal smoothing for retargeted rotations.

Sign language is the awkward case for a filter. It alternates between HOLDS,
where the hand is genuinely still and every wobble is measurement noise, and
TRANSITIONS, where the hand moves as fast as it ever will and any lag reads as
mushiness. A fixed cutoff has to pick one to serve badly.

So the filter is One Euro: an exponential smoother whose cutoff rises with the
measured angular speed. Still hands get smoothed hard, moving hands get barely
touched, and there is no lag where lag would be visible. The smoothing step
itself is a slerp rather than a component-wise blend, so the result is a
rotation by construction and never needs renormalising back onto the sphere.

Judging it needs two numbers, not one, because a filter that flattens the
motion also flattens the noise:

    hold jitter      median angular acceleration where the joint is still
    motion retained  p95 angular speed after / before
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def align_signs(q):
    """q and -q are the same rotation; make the sequence continuous.

    Without this a sign flip between consecutive frames looks like a 360-degree
    spin to any filter, and the smoother will faithfully average it into
    garbage.
    """
    q = q.copy()
    for t in range(1, q.shape[0]):
        flip = (q[t] * q[t - 1]).sum(-1) < 0
        q[t][flip] *= -1
    return q


def angular_speed(q, fps):
    """(T,J) radians per second between consecutive frames."""
    r = Rot.from_quat(q.reshape(-1, 4))
    T, J = q.shape[:2]
    a = r[:-J] if False else None
    d = (Rot.from_quat(q[:-1].reshape(-1, 4)).inv() *
         Rot.from_quat(q[1:].reshape(-1, 4))).magnitude().reshape(T - 1, J)
    return np.vstack([d[:1], d]) * fps


def rot_medoid(q, window=3):
    """Replace each rotation with the medoid of its neighbourhood.

    The finger spikes are single-frame impulses, and no exponential smoother
    removes an impulse -- it spreads it over neighbouring frames instead. That
    is what a speed cap on One Euro's adaptation failed to fix: capping changed
    which frames the filter trusted, but the spike was still averaged in.
    Impulse noise needs an order statistic. The medoid -- the member of the
    window closest to all the others in geodesic distance -- discards an
    outlier outright rather than mixing it in, and unlike a mean of rotations
    it needs no averaging on the sphere at all.
    """
    T, J, _ = q.shape
    h = window // 2
    out = q.copy()
    for t in range(T):
        lo, hi = max(0, t - h), min(T, t + h + 1)
        w = q[lo:hi]                                   # (n,J,4)
        n = w.shape[0]
        if n < 3:
            continue
        for j in range(J):
            r = Rot.from_quat(w[:, j])
            d = np.array([[(r[a].inv() * r[b]).magnitude() for b in range(n)]
                          for a in range(n)])
            out[t, j] = w[d.sum(1).argmin(), j]
    return out


def _alpha(cutoff, dt):
    tau = 1.0 / (2 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def one_euro(q, fps, mincutoff=2.5, beta=1.0, dcutoff=1.0, speed_cap=8.0):
    """One Euro filter over a (T,J,4) quaternion track.

    `speed_cap` exists because One Euro's core assumption -- that high measured
    speed means real motion, so the filter should get out of the way -- is
    false for finger joints. Their landmark noise produces apparent local
    rotations of 50-80 rad/s, which is 3000-4600 deg/s; a finger cannot exceed
    roughly 15 rad/s even at full speed. Left uncapped, the filter opens
    exactly where the signal is worst and passes the noise through. Capping the
    speed that drives adaptation keeps the responsiveness for real motion and
    denies it to spikes that physiology rules out.
    """
    q = align_signs(q)
    T, J, _ = q.shape
    dt = 1.0 / float(fps)
    out = np.empty_like(q)
    out[0] = q[0]
    prev = Rot.from_quat(q[0])
    speed = np.zeros(J)
    for t in range(1, T):
        cur = Rot.from_quat(q[t])
        raw = (Rot.from_quat(out[t - 1]).inv() * cur).magnitude() / dt
        # Low-pass the speed estimate too, or a single noisy frame opens the
        # cutoff and lets that same noise straight through.
        speed = speed + _alpha(dcutoff, dt) * (raw - speed)
        a = _alpha(mincutoff + beta * np.minimum(speed, speed_cap), dt)
        prevr = Rot.from_quat(out[t - 1])
        out[t] = np.stack([
            (prevr[j] * Rot.from_rotvec(
                a[j] * (prevr[j].inv() * cur[j]).as_rotvec())).as_quat()
            for j in range(J)])
    return out


def report(q_raw, q_smooth, fps, hold_quantile=0.3):
    """-> dict of the two numbers that matter, per the module docstring."""
    s_raw = angular_speed(align_signs(q_raw), fps)
    s_new = angular_speed(q_smooth, fps)
    acc = lambda s: np.abs(np.diff(s, axis=0)) * fps
    a_raw, a_new = acc(s_raw), acc(s_new)
    hold = s_raw[1:] <= np.quantile(s_raw, hold_quantile)
    return {
        "hold_jitter_raw": float(np.median(a_raw[hold])),
        "hold_jitter_smooth": float(np.median(a_new[hold])),
        "peak_speed_raw": float(np.percentile(s_raw, 95)),
        "peak_speed_smooth": float(np.percentile(s_new, 95)),
    }
