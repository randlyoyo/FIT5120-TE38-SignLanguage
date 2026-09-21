#!/usr/bin/env python3
"""Evaluate a fine-tuned checkpoint on a split with free-running generation.

    python evaluate.py --checkpoint checkpoints/best.pt --split test

Reports (units: shoulder widths), per sample and aggregated:
  dtw_distance           mean frame cost along the DTW path
  mpjpe_dtw (+ body / hands / face), hand_shape_error_dtw (wrist-relative)
  mpjpe_resampled        prediction linearly resampled to the reference length
  velocity_error         per-frame velocity difference after resampling
  pck@0.1, pck@0.2       fraction of joints within 0.1 / 0.2 shoulder widths (DTW-aligned)
  motion_energy_ratio    mean speed generated / reference (<< 1 = collapsed to a static pose)
  length_ratio           generated / reference frame count
and the same metrics for a STATIC MEAN-POSE baseline (training mean pose held
for the reference duration). A model that does not beat that baseline has not
learned anything visible, whatever its absolute numbers look like.

Optional back-translation (--slr-checkpoint): generated AND reference poses go
through a Uni-Sign SLT model; BLEU/chrF of both are reported. See
slp_utils/back_translation.py for the leakage caveats.

LIMITS. None of these metrics establish that the output is correct Auslan.
Coordinate error rewards being near the reference signer's trajectory; it does
not know which differences change meaning (handshape, location, movement,
orientation, non-manuals) and which do not (signer size, speed, style). A low
score with the wrong handshape is still the wrong sign. Treat the numbers as
regression tests and model-selection signals, and use back-translation plus
review by Deaf Auslan signers for any claim about linguistic quality.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer import load_trained  # noqa: E402
from slp_data.auslan_dataset import AuslanPoseDataset, Collator, read_split  # noqa: E402
from slp_utils.build import pretrained_paths  # noqa: E402
from slp_utils.config import resolve_path  # noqa: E402
from slp_utils.metrics import aggregate, sequence_metrics  # noqa: E402
from slp_utils.visualization import camera_for, render_video  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="evaluate generated Auslan poses")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=None, help="default: <checkpoint dir>/eval_<split>")
    ap.add_argument("--render-n", type=int, default=4, help="side-by-side videos for the first N samples")
    ap.add_argument("--mt5-dir", default=None)
    ap.add_argument("--unisign-repo", default=None)
    ap.add_argument("--slr-checkpoint", default=None, help="Uni-Sign SLT checkpoint for back-translation")
    ap.add_argument("--slt-fps", type=float, default=None, help="fps the SLT model was trained at (default: source median)")
    args = ap.parse_args(argv)

    model, tok, cfg, stats, ck = load_trained(Path(args.checkpoint), args.mt5_dir, args.unisign_repo, args.device)
    d = cfg["data"]
    rows = read_split(resolve_path(cfg, d["split_dir"]) / f"{args.split}.csv")
    if args.limit:
        rows = rows[: args.limit]
    ds = AuslanPoseDataset(rows, resolve_path(cfg, d["pose_dir"]), stats, d.get("text_column", "text"), train=False,
                           max_frames=int(cfg.get("normalization", {}).get("max_frames", 400)))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=Collator(tok, cfg.get("model", {}).get("text_prompt", ""), int(d.get("max_text_tokens", 64))))
    icfg = cfg.get("inference", {})
    fps = float(ck["pose_representation"]["fps"])
    mean_pose = np.asarray(stats["mean"], dtype=np.float32)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).resolve().parent / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    per, base, gens_all, refs_all, texts, uids = [], [], [], [], [], []
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if args.device.startswith("cuda") else torch.autocast("cpu", enabled=False)
    for batch in loader:
        if batch is None:
            continue
        with amp:
            gens = model.generate(batch["input_ids"].to(args.device), batch["attention_mask"].to(args.device),
                                  max_frames=int(icfg.get("max_frames", 300)), min_frames=int(icfg.get("min_frames", 8)),
                                  stop_counter=float(icfg.get("stop_counter", 0.97)))
        for i, g in enumerate(gens):
            L = int(batch["lengths"][i])
            ref = batch["pose"][i, :L].numpy()
            val = batch["joint_valid"][i, :L].numpy()
            gp = g.numpy()
            m = sequence_metrics(gp, ref, val)
            m.update(uid=batch["uid"][i], text=batch["text"][i], gen_frames=gp.shape[0], ref_frames=L)
            per.append(m)
            base.append(sequence_metrics(np.repeat(mean_pose[None], L, 0), ref, val))
            gens_all.append(gp); refs_all.append(ref); texts.append(batch["text"][i]); uids.append(batch["uid"][i])
            np.save(out_dir / f"{batch['uid'][i]}.npy", gp)
        print(f"  evaluated {len(per)}/{len(ds)}", flush=True)

    summary = {"split": args.split, "n": len(per), "checkpoint": str(Path(args.checkpoint).resolve()),
               "epoch": ck.get("epoch"), "model": aggregate(per), "mean_pose_baseline": aggregate(base)}
    summary["beats_mean_pose_baseline"] = {
        k: summary["model"][k] < summary["mean_pose_baseline"][k]
        for k in ("dtw_distance", "mpjpe_dtw", "mpjpe_dtw_hands", "hand_shape_error_dtw") if k in summary["model"]}

    if args.slr_checkpoint:
        from slp_utils.back_translation import UniSignBackTranslator, text_scores
        pp = pretrained_paths(cfg)
        slt_fps = args.slt_fps or float(stats.get("source_fps_median", fps))
        bt = UniSignBackTranslator(Path(args.slr_checkpoint), pp["unisign_repo"], pp["mt5_dir"],
                                   stats["canonical_camera"], stats["mean_confidence"], slt_fps, device=args.device)
        hyp_gen = bt.translate(gens_all, fps)
        hyp_ref = bt.translate(refs_all, fps)
        summary["back_translation"] = {"generated": text_scores(hyp_gen, texts),
                                       "reference_poses": text_scores(hyp_ref, texts)}
        for m, hg, hr in zip(per, hyp_gen, hyp_ref):
            m["bt_generated"], m["bt_reference"] = hg, hr

    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=1))
    keys = sorted({k for r in per for k in r})
    with (out_dir / "per_sample.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(per)
    cam = camera_for(gens_all[:1] + refs_all[:1], stats.get("view_bounds"))
    for i in range(min(args.render_n, len(gens_all))):
        render_video(out_dir / f"{uids[i]}_compare.mp4", [(gens_all[i], None, "generated"), (refs_all[i], None, "reference")],
                     fps, cam, texts[i])

    print(json.dumps({k: summary[k] for k in ("n", "model", "mean_pose_baseline", "beats_mean_pose_baseline")}, indent=1))
    if "back_translation" in summary:
        print(json.dumps(summary["back_translation"], indent=1))
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
