#!/bin/bash
cd "$(dirname "$0")"
PY=./venv/bin/python
for s in Valid Test_STU Test_ITW Test_TED Test_SYN Test_MTV; do
  echo "=== $s $(date '+%T') ==="
  "$PY" dtw_score.py --split $s --out scores/$s.npz
done
echo "=== SCORING DONE $(date '+%T') ==="
