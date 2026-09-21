#!/usr/bin/env python3
"""Pre-build the cached feature arrays for the evaluation splits."""
import sys
import dtw_score as S

if __name__ == "__main__":          # required: macOS spawns, and without the
    for s in sys.argv[1:]:          # guard each child re-runs the module body
        S.build_split(s, "cache", workers=6)
