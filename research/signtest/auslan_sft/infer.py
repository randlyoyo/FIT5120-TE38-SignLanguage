#!/usr/bin/env python3
"""English sentence -> Auslan pose sequence -> stick-figure MP4.

    python infer.py --checkpoint checkpoints/best.pt \
        --text "I am going to university today." --output outputs/demo.mp4

Writes outputs/demo.npy (T, 79, 2), signer space (shoulder widths, origin at
the shoulder centre, image y down), outputs/demo.json (fps, layout, text,
view bounds) and outputs/demo.mp4.

Pipeline: mT5 SentencePiece tokenizer -> fine-tuned SignT5TextToPose ->
autoregressive generation until the progress counter reaches stop_counter ->
de-standardisation (inside the model) -> optional pixel denormalisation with
the dataset's canonical camera (--pixels) -> renderer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slp_data.auslan_dataset import build_tokenizer  # noqa: E402
from slp_utils.build import build_model, pretrained_paths  # noqa: E402
from slp_utils.checkpoint import load_checkpoint  # noqa: E402
from slp_utils.config import load_config, set_by_path  # noqa: E402
from slp_utils.visualization import camera_for, render_video  # noqa: E402


def load_trained(checkpoint: Path, mt5_dir: str | None, unisign_repo: str | None, device: str):
    ck = load_checkpoint(checkpoint)
    cfg = ck["config"]
    if mt5_dir:
        set_by_path(cfg, "pretrained.mt5_dir", str(Path(mt5_dir).resolve()))
    if unisign_repo:
        set_by_path(cfg, "pretrained.unisign_repo", str(Path(unisign_repo).resolve()))
    stats = ck["pose_representation"]["norm_stats"]
    # Weights come from the fine-tuned checkpoint; no pretrained loading here.
    model, _, _, _ = build_model(cfg, stats, load_weights=False, with_feature_encoder=False, verbose=False)
    model.load_state_dict(ck["model"], strict=True)
    model.to(device).eval()
    tok = build_tokenizer(pretrained_paths(cfg)["mt5_dir"])
    return model, tok, cfg, stats, ck


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="English -> Auslan stick-figure video")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--text", required=True)
    ap.add_argument("--output", default="outputs/demo.mp4")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mt5-dir", default=None, help="override the tokenizer dir stored in the checkpoint")
    ap.add_argument("--unisign-repo", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--pixels", action="store_true", help="also save pose in canonical-camera pixels (_px.npy)")
    args = ap.parse_args(argv)

    text = args.text.strip()
    if not text:
        raise SystemExit("empty --text")
    model, tok, cfg, stats, ck = load_trained(Path(args.checkpoint), args.mt5_dir, args.unisign_repo, args.device)
    icfg = cfg.get("inference", {})
    prompt = cfg.get("model", {}).get("text_prompt", "")
    enc = tok([prompt + text], return_tensors="pt", truncation=True,
              max_length=int(cfg["data"].get("max_text_tokens", 64)))
    gen = model.generate(enc["input_ids"].to(args.device), enc["attention_mask"].to(args.device),
                         max_frames=args.max_frames or int(icfg.get("max_frames", 300)),
                         min_frames=int(icfg.get("min_frames", 8)),
                         stop_counter=float(icfg.get("stop_counter", 0.97)))[0].numpy().astype(np.float32)
    fps = float(ck["pose_representation"]["fps"])
    hit_max = gen.shape[0] >= (args.max_frames or int(icfg.get("max_frames", 300)))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    npy = out.with_suffix(".npy")
    np.save(npy, gen)
    meta = {"text": text, "fps": fps, "frames": int(gen.shape[0]), "stopped_by_counter": not hit_max,
            "layout": ck["pose_representation"]["layout_version"],
            "units": ck["pose_representation"]["units"], "view_bounds": stats.get("view_bounds"),
            "checkpoint": str(Path(args.checkpoint).resolve()), "epoch": ck.get("epoch"),
            "note": "Generated skeleton motion. Not verified as correct Auslan."}
    if args.pixels:
        cx, cy, s = stats["canonical_camera"][:3]
        np.save(out.with_name(out.stem + "_px.npy"), gen * s + np.asarray([cx, cy], dtype=np.float32))
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"generated {gen.shape[0]} frames ({gen.shape[0]/fps:.2f}s) -> {npy}"
          + ("  [WARNING: hit max_frames; counter never reached stop threshold]" if hit_max else ""))
    if not args.no_render:
        cam = camera_for([gen], stats.get("view_bounds"), int(icfg.get("render_size", 720)))
        render_video(out, [(gen, None, None)], fps, cam, text)
        print(f"rendered -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
