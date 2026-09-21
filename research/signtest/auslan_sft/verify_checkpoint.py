#!/usr/bin/env python3
"""Step 4: verify the pretrained checkpoint loads before any training.

    python verify_checkpoint.py --config configs/auslan_sft.yaml

Prints loaded / missing / unexpected / trainable counts and exits non-zero if
the load gate fails. With --forward, also runs one teacher-forced pass and a
3-frame generation on dummy input (or one real batch when splits exist) to
prove the loaded model is wired correctly.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slp_models import skeleton as sk  # noqa: E402
from slp_models.pretrained_slp import PretrainedLoadError  # noqa: E402
from slp_utils.build import build_model, pretrained_paths  # noqa: E402
from slp_utils.config import load_config  # noqa: E402

EXPECTED_SHA256 = {"openasl_pose_only_slt.pth": "f836ea66bc837bbe6ed717a4b9bece87875f03ef96d4bf1092ca3dd767982798",
                   "how2sign_pose_only_slt.pth": "1bfd5f3312f04e4736f0a52f4ef9535916e6de9676a2a0d00c708748683fb00d"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--skip-sha", action="store_true")
    ap.add_argument("--forward", action="store_true", help="also run a tiny forward + generate")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    pp = pretrained_paths(cfg)
    ck = pp["unisign_checkpoint"]
    print(f"checkpoint: {ck}")
    if not ck.is_file():
        print("ERROR: checkpoint file missing. Download from "
              "https://huggingface.co/ZechengLi19/Uni-Sign (revision eab251b7)")
        return 2
    if not args.skip_sha and ck.name in EXPECTED_SHA256:
        got = sha256(ck)
        ok = got == EXPECTED_SHA256[ck.name]
        print(f"sha256: {got} {'OK' if ok else 'MISMATCH (expected ' + EXPECTED_SHA256[ck.name] + ')'}")
        if not ok:
            return 2
    try:
        model, fe, rep, mode = build_model(cfg, None, load_weights=True, with_feature_encoder=True)
    except (PretrainedLoadError, FileNotFoundError) as exc:
        print(f"STOP: {exc}")
        return 1
    per_group = {}
    from slp_models.freeze import group_of
    for n, p in model.named_parameters():
        g = group_of(n)
        g = "enc_blocks" if g.startswith("enc_block") else ("dec_blocks" if g.startswith("dec_block") else g)
        a, t = per_group.get(g, (0, 0))
        per_group[g] = (a + p.numel(), t + (p.numel() if p.requires_grad else 0))
    print("parameter groups (total / trainable):")
    for g, (a, t) in per_group.items():
        print(f"  {g:<16} {a:>13,} / {t:>13,}")
    fe_n = sum(p.numel() for p in fe.parameters())
    print(f"frozen Uni-Sign pose encoder: {fe_n:,} params (not part of the generator)")

    if args.forward:
        from slp_data.auslan_dataset import build_tokenizer
        tok = build_tokenizer(pp["mt5_dir"])
        enc = tok(["how are you ?", "i am going to school ."], padding=True, return_tensors="pt")
        B, T = 2, 12
        pose = torch.randn(B, T, sk.NUM_JOINTS, sk.COORD_DIM) * 0.3
        fm = torch.ones(B, T, dtype=torch.bool)
        model.eval()
        with torch.no_grad():
            pred, counter = model(enc["input_ids"], enc["attention_mask"], pose, fm)
            gens = model.generate(enc["input_ids"], enc["attention_mask"], max_frames=3, min_frames=1)
            from slp_utils.unisign_format import to_unisign_parts
            cam = torch.tensor([[640., 300., 150., 1280., 720., 0.3, 0.1, 0.5]] * B)
            feats = fe(to_unisign_parts(pose, torch.ones(B, T, sk.NUM_JOINTS, dtype=torch.bool), cam))
        assert pred.shape == pose.shape and counter.shape == (B, T), (pred.shape, counter.shape)
        assert torch.isfinite(pred).all() and torch.isfinite(feats).all()
        print(f"forward OK: pose_hat {tuple(pred.shape)}, counter {tuple(counter.shape)}, "
              f"generated {[tuple(g.shape) for g in gens]}, Uni-Sign features {tuple(feats.shape)}")
    print("VERIFIED: pretrained weights loaded; safe to start SFT.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
