#!/usr/bin/env python3
"""A killed-and-resumed training run must end exactly where an uninterrupted one does.

Builds a tiny synthetic corpus, trains the smoke model with checkpoints every
few steps, and for each arm compares three runs:

  reference   one uninterrupted run
  killed      the same run, hard-killed (os._exit) mid-epoch -- the steps
              since the last checkpoint are lost, as when Colab reclaims a
              runtime -- then resumed with --resume
  stopped     the same run, stopped as if by Ctrl-C / Stop mid-epoch (it saves
              on the way out), then resumed

Every parameter of the final model must be bit-identical (CPU, num_workers=0).
Arm C exercises the auxiliary stream crossing an aux-epoch boundary; Arm B
stage 1 exercises per-group learning rates. Then the guards: refusing to start
over existing checkpoints without --resume, refusing to resume under a changed
config, and a --resume of a finished run doing no training.

    python test_resume.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spec  # noqa: E402

SENT = ["the meeting starts at nine", "it is raining in melbourne today",
        "i went to the shops yesterday", "the prime minister spoke this morning",
        "how are you feeling now", "my sister works at the hospital"]
GLOSS = ["WHALE", "POLICE", "SHAVE (UNDERARM)", "YOUR (PLURAL)", "ZOO",
         "ACCIDENT", "HOSPITAL", "RAIN"]


def make_corpus(root: Path) -> None:
    rng = np.random.default_rng(1)
    pose = root / "pose"
    pose.mkdir(parents=True)
    W, H, rows = 640, 480, []

    def make(uid, ds, sub, split, text, gloss, T):
        kp = rng.random((T, 133, 2)).astype(np.float32) * 0.4 + 0.3
        kp[:, 5, 0], kp[:, 6, 0] = 0.62, 0.38
        kp[:, spec.PART_INDICES["left"], 0] += (sum(map(ord, text)) % 7) * 0.02
        sc = np.full((T, 133), 0.9, np.float32)
        meta = dict(uid=uid, dataset=ds, subset=sub, split=split, text=text, gloss=gloss,
                    camera=None, source="synthetic", spec_version=spec.SPEC_VERSION,
                    schema_fingerprint=spec.SCHEMA_FINGERPRINT,
                    pose_model="synthetic", person_select="largest",
                    coordinate_space="frame_normalised", width=W, height=H,
                    fps=25.0, n_frames=int(T))
        with (pose / f"{uid}.npz").open("wb") as fh:
            np.savez_compressed(fh, keypoints=kp, scores=sc, meta=json.dumps(meta))
        rows.append(dict(uid=uid, dataset=ds, subset=sub, split=split,
                         video="synthetic", text=text, gloss=gloss, camera=None))

    i = 0
    for split, n in (("train", 48), ("val", 12)):
        for k in range(n):
            make(f"ad-{split}-{i}", "auslandaily", "news" if k % 2 else "communication",
                 split, SENT[k % len(SENT)], None, int(rng.integers(20, 40)))
            i += 1
    for k in range(64):
        g = GLOSS[k % len(GLOSS)]
        make(f"mmwl-{k}", "mmwlauslan", "studio", "train", g, g, int(rng.integers(12, 24)))
    (root / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def run(root: Path, cfg: str, out: Path, *extra: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(HERE / "train.py"), "--config", str(HERE / "configs" / cfg),
           "--out", str(out), "--device", "cpu", "--no-eval", *extra, "--set",
           f"data.manifest={root / 'manifest.jsonl'}", f"data.npz_dir={root / 'pose'}",
           "num_workers=0", "log_every=1000", "optim.epochs=3", "backend.d=32",
           "checkpoint.every_steps=5", "checkpoint.every_minutes=null", "checkpoint.keep=1"]
    return subprocess.run(cmd, capture_output=True, text=True)


def final_weights(out: Path) -> dict:
    return torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=True)["model"]


def compare(a: dict, b: dict) -> tuple[bool, float]:
    if a.keys() != b.keys():
        return False, float("inf")
    worst = max(float((a[k].double() - b[k].double()).abs().max()) if a[k].numel() else 0.0
                for k in a)
    return all(torch.equal(a[k], b[k]) for k in a), worst


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="test-resume-"))
    failures = 0
    try:
        make_corpus(tmp / "data")
        data = tmp / "data"
        print(f"{'arm':14} {'scenario':10} {'resumed from':>13}  result")
        for cfg in ("arm_a.yaml", "arm_c.yaml", "arm_b_stage1.yaml"):
            ref = run(data, cfg, tmp / cfg / "ref")
            if ref.returncode != 0:
                print(ref.stdout[-1500:], ref.stderr[-1500:])
                raise SystemExit(f"{cfg}: reference run failed")
            w_ref = final_weights(tmp / cfg / "ref")
            for name, hook, code in (("killed", ("--crash-after-steps", "11"), 137),
                                     ("stopped", ("--interrupt-after-steps", "7"), 130),
                                     ("paused", ("--stop-after-epochs", "1"), 130)):
                out = tmp / cfg / name
                first = run(data, cfg, out, *hook)
                if first.returncode != code:
                    print(first.stdout[-1500:], first.stderr[-1500:])
                    print(f"{cfg:14} {name:10}  FAIL: first leg exited {first.returncode}, expected {code}")
                    failures += 1
                    continue
                second = run(data, cfg, out, "--resume")
                line = next((l for l in second.stdout.splitlines() if l.startswith("resumed from")), "")
                step = line.split("step ")[1].split("/")[0] if "step " in line else "?"
                if second.returncode != 0:
                    print(second.stdout[-1500:], second.stderr[-1500:])
                    print(f"{cfg:14} {name:10} {step:>13}  FAIL: resume exited {second.returncode}")
                    failures += 1
                    continue
                same, worst = compare(w_ref, final_weights(out))
                failures += not same
                print(f"{cfg:14} {name:10} {'step ' + step:>13}  "
                      + ("PASS  bit-identical to the uninterrupted run" if same
                         else f"FAIL  max |diff| {worst:.3e}"))

        # --- guards, on a finished run -----------------------------------------
        out = tmp / "arm_a.yaml" / "killed"
        g1 = run(data, "arm_a.yaml", out)
        ok1 = g1.returncode != 0 and "--resume" in (g1.stderr + g1.stdout)
        g2 = run(data, "arm_a.yaml", out, "--resume", "--set", "optim.lr=0.001")
        # note: the second --set is appended after the harness defaults
        ok2 = g2.returncode != 0 and "different run" in (g2.stderr + g2.stdout)
        g3 = run(data, "arm_a.yaml", out, "--resume")
        ok3 = g3.returncode == 0 and "run already finished" in g3.stdout and "[checkpoint]" not in g3.stdout
        for ok, what in ((ok1, "start over existing checkpoints without --resume -> refused"),
                         (ok2, "--resume with a changed config -> refused"),
                         (ok3, "--resume of a finished run -> no training, exits cleanly")):
            failures += not ok
            print(f"{'guard':14} {'':10} {'':>13}  {'PASS' if ok else 'FAIL'}  {what}")
            if not ok:
                for g in (g1, g2, g3):
                    pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL PASS" if failures == 0 else f"\n{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
