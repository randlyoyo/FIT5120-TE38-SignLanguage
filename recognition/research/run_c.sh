#!/bin/bash
# Arm C: the same encoder and the same hyper-parameters as runs/final, trained
# on canonicalised 3D features instead of 2D image-plane ones.
#
# Every number below is copied from runs/final/result.json. Re-tuning for the
# 3D features would confound the comparison: a win could then be the new
# features or the new hyper-parameters, and there would be no way to tell.
set -e
SIGN_CACHE=cache3d SIGN_NDIM=3 SIGN_SPLITS=Valid,Test_MTV \
venv/bin/python metric_train.py \
  --T 96 --hidden 256 --emb 256 --depth 3 --drop-p 0.1 \
  --lr 0.00548854262206015 --wd 0.01 --bs 128 --epochs 120 --ls 0.0 \
  --scale-s 20.0 --margin 0.2 --eval-every 5 --seed 0 \
  --speed 0.1 --scale 0.1 --rot 10.0 --shift 0.03 --drop 0.0 --noise 0.02 \
  --out runs/mtv3d --ckpt runs/mtv3d/ckpt.pt
