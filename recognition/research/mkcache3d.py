#!/usr/bin/env python3
"""Build the 3D feature cache, mirroring dtw_score.build_split.

    python mkcache3d.py Train Valid Test_MTV

Writes cache3d/<split>.npz in exactly the format metric_data.load_split reads,
so the trainer needs a cache directory switch and nothing else.
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import dtw_core as K
import dtw_features3d as F3

OUT = Path("cache3d")


def _one(p):
    try:
        return p, F3.build(p), None
    except Exception as e:                       # Unusable, or a corrupt file
        return p, None, f"{type(e).__name__}: {e}"


def build(split, workers=4):
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / f"{split}.npz"
    if cache.exists():
        print(f"  {split}: exists, skipped")
        return
    paths = sorted(Path(f"keypoints3d/{split}").glob("*.npz"))
    with Pool(workers) as pool:
        res = pool.map(_one, [str(p) for p in paths], chunksize=64)
    seqs, stems, dropped = [], [], {}
    for path, seq, err in res:
        if err:
            k = err.split(":")[0]
            dropped[k] = dropped.get(k, 0) + 1
            continue
        seqs.append(seq)
        stems.append(Path(path).stem)
    print(f"  {split}: {len(seqs)}/{len(paths)} usable, dropped {dropped}", flush=True)
    buf, offs = K.pack(seqs)
    np.savez_compressed(cache, buf=buf, offsets=offs, stems=np.array(stems))


if __name__ == "__main__":
    for s in sys.argv[1:]:
        build(s, workers=3)      # extraction still owns 6 cores; stay polite
