#!/bin/bash
cd "$(dirname "$0")"
./venv/bin/python metric_sweep.py --trials 24 --epochs 30 --out runs/sweep
echo "=== SWEEP DONE $(date '+%F %T') ==="
./venv/bin/python metric_sweep.py --final --final-epochs 120 --out runs/sweep
echo "=== FINAL DONE $(date '+%F %T') ==="
