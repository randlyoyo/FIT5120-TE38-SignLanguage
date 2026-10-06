"""Convert per-sign SMPL-X .npz fits into gzipped pose JSON for importPoses.js.

Usage (needs only numpy -- the sign_chat_backend venv has it):
    python convert_poses.py <npz_dir> <out_dir>

Each <GLOSS>.npz becomes <out_dir>/<GLOSS>.json.gz in the same shape as the
Sign Chat backend's pose JSON ({fps, frames, smplx: {...}}), so the client
plays it with the existing parseSmplxPose + SmplxAvatar.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

KEYS = ("global_orient", "body_pose", "left_hand_pose", "right_hand_pose")


def convert(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    return {
        "fps": float(meta.get("fps", 30.0)),
        "frames": int(d["body_pose"].shape[0]),
        # 4 decimals of a radian is ~0.006 degrees -- invisible on screen.
        # Rounded in float64: rounding float32 then calling tolist() turns
        # 0.6049 back into 0.6049000024795532 and undoes the size saving.
        "smplx": {k: np.round(d[k].astype(np.float64), 4).tolist() for k in KEYS},
    }


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(src.glob("*.npz"))
    for p in paths:
        data = json.dumps(convert(p), separators=(",", ":")).encode()
        (out / f"{p.stem}.json.gz").write_bytes(gzip.compress(data))
    print(f"converted {len(paths)} file(s) into {out}")


if __name__ == "__main__":
    main()
