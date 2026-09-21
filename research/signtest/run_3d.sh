#!/bin/bash
# 3D keypoint extraction, all splits, straight from the zips.
# Resumable: finished .npz files are skipped, so re-run after any stop.
cd "$(dirname "$0")"
PY=./venv/bin/python
W="${W:-4}"

run () {
  [ -f "$2" ] || { echo "SKIP $1 (no $2)"; return; }
  echo "=== $1  $(date '+%F %T') ==="
  "$PY" extract3d.py --zip "$2" --out "keypoints3d/$1" \
        --model ./holistic_landmarker.task --workers "$W"
}

# Valid first: smallest, so anything wrong surfaces in minutes not hours.
run Valid    "MM-WLAuslan/Valid/rgb.zip"
run Train    "MM-WLAuslan/Train/rgb.zip"
run Test_STU "MM-WLAuslan/Test_STU/rgb.zip"
run Test_ITW "MM-WLAuslan/Test_ITW/rgb.zip"
run Test_TED "MM-WLAuslan/Test_TED/rgb.zip"
run Test_SYN "MM-WLAuslan/Test_SYN/rgb (1).zip"
run Test_MTV "MM-WLAuslan/Test_MTV/Test_MTV_RGB.zip"
echo "=== ALL DONE $(date '+%F %T') ==="
