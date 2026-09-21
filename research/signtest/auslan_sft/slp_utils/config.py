"""YAML config loading, dotted overrides, path resolution and seeding."""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_value(text: str) -> Any:
    # YAML parsing gives ints, floats, bools, null and lists for free.
    return yaml.safe_load(text)


def set_by_path(cfg: dict, dotted: str, value: Any) -> None:
    node = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def get_by_path(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> dict:
    """Load YAML; apply `key.sub=value` overrides; remember where it came from.

    Relative paths inside the config are resolved against the config file's
    directory's parent (the project root layout: configs/<file>.yaml), so the
    same config works from any working directory.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open() as fh:
        cfg = yaml.safe_load(fh) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        k, v = item.split("=", 1)
        set_by_path(cfg, k.strip(), _parse_value(v))
    cfg.setdefault("_meta", {})["config_path"] = str(path.resolve())
    cfg["_meta"]["base_dir"] = str(path.resolve().parent.parent)
    return cfg


def resolve_path(cfg: dict, p: str | None) -> Path | None:
    """Absolute paths pass through; relative ones are relative to the project root."""
    if p is None or p == "":
        return None
    q = Path(os.path.expanduser(str(p)))
    if q.is_absolute():
        return q
    base = Path(cfg.get("_meta", {}).get("base_dir", PROJECT_ROOT))
    return (base / q).resolve()


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clean_for_save(cfg: dict) -> dict:
    return copy.deepcopy(cfg)
