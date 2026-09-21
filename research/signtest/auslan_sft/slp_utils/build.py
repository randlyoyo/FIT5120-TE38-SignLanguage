"""Shared construction of model, feature encoder and data for all entry points."""

from __future__ import annotations

from pathlib import Path

import torch

from slp_models import skeleton as sk
from slp_models.freeze import apply_mode
from slp_models.pretrained_slp import (SignT5TextToPose, UniSignPoseEncoder, check_load_gate,
                                       load_pretrained, mt5_config_from)
from slp_utils.config import resolve_path


def pretrained_paths(cfg: dict) -> dict:
    p = dict(cfg.get("pretrained", {}))
    p["mt5_dir"] = resolve_path(cfg, p.get("mt5_dir"))
    p["unisign_checkpoint"] = resolve_path(cfg, p.get("unisign_checkpoint"))
    p["unisign_repo"] = resolve_path(cfg, p.get("unisign_repo"))
    return p


def build_model(cfg: dict, stats: dict | None, *, load_weights: bool, with_feature_encoder: bool | None = None,
                verbose: bool = True):
    """Returns (model, feature_encoder_or_None, load_report_or_None, mode_summary)."""
    pp = pretrained_paths(cfg)
    config = mt5_config_from(cfg.get("model", {}), pp["mt5_dir"])
    model = SignT5TextToPose(config, sk.NUM_JOINTS, sk.COORD_DIM)
    if stats is not None:
        model.set_pose_stats(torch.tensor(stats["mean"], dtype=torch.float32),
                             torch.tensor(stats["std"], dtype=torch.float32))
    if with_feature_encoder is None:
        with_feature_encoder = float(cfg.get("loss", {}).get("lambda_feat", 0.0)) > 0
    fe = UniSignPoseEncoder(pp["unisign_repo"]) if with_feature_encoder else None
    report = None
    if load_weights:
        report = load_pretrained(model, pp, fe)
    mode = apply_mode(model, cfg.get("training", {}))
    if report is not None:
        report.trainable_params = mode["trainable_params"]
        if verbose:
            report.print()
            print(f"Training mode {mode['mode']}: trainable groups {mode['trainable_groups']}")
        check_load_gate(report, pp)
    return model, fe, report, mode
