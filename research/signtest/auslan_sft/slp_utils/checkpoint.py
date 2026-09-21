"""Checkpoint save/load (best.pt, latest.pt)."""

from __future__ import annotations

from pathlib import Path

import torch

from slp_models import skeleton as sk


def save_checkpoint(path: Path, *, model, optimizer, scheduler, scaler, epoch: int, step: int,
                    val_metric: float, best_metric: float, config: dict, norm_stats: dict,
                    load_report: dict, bad_epochs: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch, "step": step,
        "val_metric": val_metric, "best_metric": best_metric, "bad_epochs": bad_epochs,
        "config": config,
        "pose_representation": {
            **sk.layout_metadata(),
            "units": "shoulder widths, origin at clip-median shoulder centre, image y down",
            "fps": config.get("normalization", {}).get("target_fps", 25.0),
            "norm_stats": norm_stats,
        },
        "pretrained_load_report": load_report,
        "torch_rng": torch.get_rng_state(),
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, map_location="cpu") -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ck = torch.load(str(path), map_location=map_location, weights_only=False)
    for key in ("model", "config", "pose_representation"):
        if key not in ck:
            raise KeyError(f"{path} is not an auslan_sft checkpoint (no {key!r})")
    rep = ck["pose_representation"]
    if rep.get("fingerprint") != sk.layout_metadata()["fingerprint"]:
        raise RuntimeError(f"{path}: pose layout {rep.get('layout_version')} differs from this code "
                           f"({sk.LAYOUT_VERSION}); refusing to load.")
    return ck
