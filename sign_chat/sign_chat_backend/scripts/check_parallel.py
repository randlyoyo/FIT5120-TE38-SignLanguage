#!/usr/bin/env python3
"""Is parallel sampling of the three streams the same generation as the sequential reference, and
how much faster is it?

    SIGNCHAT_CONFIG=/content/signchat.yaml python scripts/check_parallel.py

Loads the generator the server uses (~20 GB of GPU memory: stop the server first) and generates each
sentence twice: parallel=False runs signspark_render's own sample_* functions one stream after the
other, parallel=True the threaded version. Prints the largest feature difference per stream (0 =
identical) and the time of each.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SENTENCES = ["Hello, how are you?", "I am good, thank you.", "What is your name?",
             "Let us go now.", "I am hungry, let us eat something together."]


def main():
    import torch
    from signchat.config import load_config
    from signchat.text2sign import STREAMS, SignGenerator

    gen = SignGenerator(load_config(), torch.device("cuda"))
    gen.features("Good morning.", parallel=False)          # warm-up, both paths
    gen.features("Good morning.", parallel=True)
    worst, ts = 0.0, {False: [], True: []}
    for sent in SENTENCES:
        f = {}
        for par in (False, True):
            torch.cuda.synchronize()
            t0 = time.time()
            f[par] = gen.features(sent, parallel=par)
            ts[par].append(time.time() - t0)
        d = {s: float(np.abs(f[True][s] - f[False][s]).max()) for s in STREAMS}
        worst = max(worst, *d.values())
        print(f"{sent!r:<48} T={f[True]['T']:<4} sequential {ts[False][-1]:.2f}s | parallel {ts[True][-1]:.2f}s | "
              + " ".join(f"{s} {v:.2e}" for s, v in d.items()), flush=True)
    print(f"\nmean: sequential {np.mean(ts[False]):.2f}s, parallel {np.mean(ts[True]):.2f}s "
          f"({np.mean(ts[False]) / np.mean(ts[True]):.2f}x); largest feature difference {worst:.2e}"
          + ("  -> identical" if worst == 0 else ""))
    print("peak GPU memory", f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
