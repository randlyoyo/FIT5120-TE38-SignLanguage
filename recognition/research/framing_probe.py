#!/usr/bin/env python3
"""Does apparent size (i.e. distance) actually predict recognition failure?

The feature pipeline divides by shoulder width, so scale is normalised away and
distance should NOT matter geometrically. What distance can still break is
detection: a hand far from the camera is a few pixels across and MediaPipe
stops finding it. This measures which of the two stories the data supports,
because the guidance shown to a user should come from the failure that is real.
"""
import glob, json, os, random
import numpy as np
import slr_common as C

P = {v: i for i, v in enumerate(C.POSE_SUBSET)}
SPLIT = "Test_ITW"

import mcnemar as MC                     # reuse its loader
correct = MC.predictions("cache", 2, "runs/final/encoder.pt", SPLIT)

rows = []
for f in glob.glob(f"keypoints/{SPLIT}/*.npz"):
    stem = os.path.basename(f)[:-4]
    if stem not in correct:
        continue
    d = np.load(f, allow_pickle=False)
    m = json.loads(str(d["meta"]))
    w, h = m.get("width", 0), m.get("height", 0)
    if not (w and h):
        continue
    p = d["pose"][:, :, :2].astype(np.float32).copy()
    p[:, :, 0] *= w / h
    sw = np.nanmedian(np.linalg.norm(p[:, P[11]] - p[:, P[12]], axis=-1))
    lh = ~np.isnan(d["left_hand"][:, 0, 0]); rh = ~np.isnan(d["right_hand"][:, 0, 0])
    rows.append((sw, (lh & rh).mean(), (lh | rh).mean(), float(correct[stem]),
                 min(w, h)))
a = np.array(rows)
print(f"\n{SPLIT}  n={len(a)}")
qs = np.nanpercentile(a[:, 0], [0, 10, 30, 50, 70, 90, 100])
print(f"\n{'肩宽区间':>16s}{'n':>7s}{'双手检出':>10s}{'至少一手':>10s}{'rank-1':>9s}")
print("-" * 54)
for lo, hi in zip(qs[:-1], qs[1:]):
    m = (a[:, 0] >= lo) & (a[:, 0] <= hi if hi == qs[-1] else a[:, 0] < hi)
    if m.sum() == 0:
        continue
    print(f"{lo:7.3f}-{hi:6.3f}{int(m.sum()):7d}{a[m,1].mean():10.3f}"
          f"{a[m,2].mean():10.3f}{a[m,3].mean()*100:8.2f}%")
r = np.corrcoef(a[:, 0], a[:, 1])[0, 1]
r2 = np.corrcoef(a[:, 0], a[:, 3])[0, 1]
print(f"\n相关: 肩宽~双手检出 r={r:+.3f}   肩宽~正确 r={r2:+.3f}")
print(f"分辨率(短边)与检出 r={np.corrcoef(a[:,4],a[:,1])[0,1]:+.3f}")
np.save("/tmp/itw_framing.npy", a)
