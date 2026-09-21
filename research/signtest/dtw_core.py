#!/usr/bin/env python3
"""Banded DTW, JIT-compiled. The inner loop of the whole baseline."""

import numpy as np
from numba import njit, prange

BAND_FRAC = 0.15
BAND_MIN = 8


@njit(cache=True, fastmath=True, inline="always")
def _frame_dist(A, B, i, j):
    s = 0.0
    for k in range(A.shape[1]):
        d = A[i, k] - B[j, k]
        s += d * d
    return np.sqrt(s)


@njit(cache=True, fastmath=True)
def dtw(A, B, band_frac=BAND_FRAC, band_min=BAND_MIN):
    """Length-normalised banded DTW distance between (n,D) and (m,D).

    The band is centred on the *scaled* diagonal j = i*m/n, not on i == j.
    That matters: even after resampling to a common frame rate, a slow signer
    and a fast one differ in length by ~1.4x here, and a band around i == j
    would put the correct alignment out of reach for the longer pairs.

    Returns accumulated cost divided by the warping path length. Dividing is
    not cosmetic -- raw DTW cost grows with path length, so without it every
    long clip looks dissimilar to everything.
    """
    n, m = A.shape[0], B.shape[0]
    r = max(band_min, int(band_frac * max(n, m)))
    ratio = m / n

    INF = 1e18
    cost = np.full((2, m + 1), INF)     # rolling rows
    plen = np.zeros((2, m + 1), np.int32)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        cur, prev = i & 1, (i - 1) & 1
        centre = (i - 1) * ratio
        lo = max(1, int(centre - r) + 1)
        hi = min(m, int(centre + r) + 1)
        for j in range(m + 1):
            cost[cur, j] = INF
            plen[cur, j] = 0
        for j in range(lo, hi + 1):
            d = _frame_dist(A, B, i - 1, j - 1)
            # pick the cheapest predecessor, carrying its path length along
            c0, c1, c2 = cost[prev, j - 1], cost[prev, j], cost[cur, j - 1]
            if c0 <= c1 and c0 <= c2:
                best, bl = c0, plen[prev, j - 1]
            elif c1 <= c2:
                best, bl = c1, plen[prev, j]
            else:
                best, bl = c2, plen[cur, j - 1]
            if best >= INF:
                continue
            cost[cur, j] = best + d
            plen[cur, j] = bl + 1

    c = cost[n & 1, m]
    L = plen[n & 1, m]
    if L == 0 or c >= INF:
        return np.inf
    return c / L


@njit(cache=True, parallel=True, fastmath=True)
def dtw_one_to_many(q, refs, offsets, out, band_frac=BAND_FRAC, band_min=BAND_MIN):
    """One query against many templates packed into a single flat array.

    refs is every template stacked on axis 0; offsets[k]..offsets[k+1] slices
    out template k. Packing avoids a Python-level loop over a reflected list,
    which is where numba object mode would otherwise cost more than the DTW.
    """
    for k in prange(offsets.shape[0] - 1):
        out[k] = dtw(q, refs[offsets[k]:offsets[k + 1]], band_frac, band_min)


def pack(seqs):
    """list of (T,D) -> (sum_T, D) array plus an offsets index."""
    offs = np.zeros(len(seqs) + 1, np.int64)
    for i, s in enumerate(seqs):
        offs[i + 1] = offs[i] + s.shape[0]
    buf = np.empty((offs[-1], seqs[0].shape[1]), np.float32)
    for i, s in enumerate(seqs):
        buf[offs[i]:offs[i + 1]] = s
    return buf, offs


def score_against(q, packed, offsets, n_per_word):
    """Distance from q to each word, as the min over that word's templates.

    Min rather than mean: verification asks whether the query matches *any*
    stored example of W, and averaging lets one atypical template drag a good
    match down.
    """
    out = np.empty(offsets.shape[0] - 1, np.float64)
    dtw_one_to_many(q, packed, offsets, out)
    return out.reshape(-1, n_per_word).min(axis=1)
