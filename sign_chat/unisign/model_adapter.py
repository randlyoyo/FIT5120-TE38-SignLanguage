#!/usr/bin/env python3
"""Bridge to the Uni-Sign checkpoint, plus the parameter grouping the arms need.

This file does not reimplement Uni-Sign. It loads the real model from the real
repo, because a reimplementation that is 95% right loads the pre-trained
weights without complaint and is worthless in a way no error message reveals.

What it *does* own is the thing the experiment design actually turns on:
deciding which parameters are the four per-part pose encoders, which are the
temporal stack, and which are the mT5 decoder. Arm B stage 1 unfreezes only the
first group (plan section 4). Getting that grouping silently wrong -- matching
zero parameters, or matching everything -- turns Arm B into either "no training
happened" or "a naive two-stage run", the exact thing the plan rules out. Both
produce a plausible-looking loss curve.

So grouping is by explicit regex over parameter names, the patterns live in the
run config where they can be corrected, and `python model_adapter.py --report`
prints what matched before any GPU time is spent. Check it once against the
checkpoint you actually downloaded.
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

# Name patterns, taken from Uni_Sign.__init__ in the repo (models.py), not
# guessed. The distinction that matters for Arm B is spatial vs temporal:
#
#   proj_linear          nn.Linear(3, 64) per part          -- spatial
#   gcn_modules          get_stgcn_chain(..., 'spatial')    -- spatial
#   fusion_gcn_modules   get_stgcn_chain(..., 'temporal')   -- temporal
#   part_para, pose_proj  fuse the parts and project to mT5 -- temporal side
#   mt5_model            the decoder
#   rgb_*, fusion_pose_rgb_*, fusion_gate  RGB branch, unused in pose-only
#
# Arm B stage 1 unfreezes `pose_encoder` only: that is exactly "the four
# per-part pose encoders (spatial branch)" from the plan.
#
# Order matters -- first match wins, so fusion_gcn_modules must be tested
# before the looser `gcn` pattern would catch it.
DEFAULT_GROUP_PATTERNS: dict[str, list[str]] = {
    "rgb": [r"rgb_support_backbone", r"rgb_proj", r"fusion_pose_rgb",
            r"fusion_gate"],
    # The trailing entries in each list are smoke_model.py's names, so one set
    # of patterns covers both the real model and the harness. They cannot
    # collide: the real model has no module by those names.
    "decoder": [r"^mt5_model", r"^decoder_embed", r"^decoder_stack", r"^lm_head",
                r"^decoder_pos"],
    "temporal": [r"fusion_gcn_modules", r"^pose_proj", r"^part_para",
                 r"^temporal_encoder", r"^temporal_pos"],
    "pose_encoder": [r"^proj_linear", r"^gcn_modules"],
}


@dataclass
class ParamGroups:
    assignment: dict[str, str]                     # param name -> group
    counts: dict[str, int] = field(default_factory=dict)
    params: dict[str, int] = field(default_factory=dict)

    def names(self, group: str) -> list[str]:
        return [n for n, g in self.assignment.items() if g == group]


def group_parameters(model: nn.Module,
                     patterns: dict[str, list[str]] | None = None) -> ParamGroups:
    """Assign every parameter to exactly one group, or to 'unmatched'.

    First matching group in declaration order wins, so the patterns are ordered
    most-specific first. 'unmatched' is not an error but it is never silent:
    the report prints it, and the freezing policies refuse to run if too much
    of the model landed there.
    """
    pats = patterns or DEFAULT_GROUP_PATTERNS
    compiled = {g: [re.compile(p, re.I) for p in ps] for g, ps in pats.items()}
    assignment: dict[str, str] = {}
    counts: dict[str, int] = {}
    nparams: dict[str, int] = {}
    for name, p in model.named_parameters():
        group = "unmatched"
        for g, regs in compiled.items():
            if any(r.search(name) for r in regs):
                group = g
                break
        assignment[name] = group
        counts[group] = counts.get(group, 0) + 1
        nparams[group] = nparams.get(group, 0) + p.numel()
    return ParamGroups(assignment, counts, nparams)


def apply_freeze_policy(model: nn.Module, groups: ParamGroups,
                        trainable: list[str], *,
                        max_unmatched_frac: float = 0.10) -> dict:
    """Freeze everything outside `trainable`. Returns a summary to be logged.

    Refuses to proceed when a trainable group matched nothing, or when a large
    share of the model is unmatched. A stage-1 run that trains zero parameters
    still emits a loss curve -- a flat one that is easy to mistake for
    convergence -- so this has to be an exception, not a warning.
    """
    total = sum(groups.params.values())
    unmatched = groups.params.get("unmatched", 0)
    if total and unmatched / total > max_unmatched_frac:
        raise RuntimeError(
            f"{unmatched/total:.1%} of parameters matched no group pattern. "
            "The patterns do not fit this checkpoint -- run "
            "`python model_adapter.py --report` and fix `group_patterns` in the "
            "config before training."
        )
    for g in trainable:
        if not groups.params.get(g):
            raise RuntimeError(
                f"group {g!r} is marked trainable but matched 0 parameters; "
                "the freeze policy would train nothing."
            )

    live = set(trainable)
    n_train = 0
    for name, p in model.named_parameters():
        p.requires_grad_(groups.assignment[name] in live)
        if p.requires_grad:
            n_train += p.numel()
    return {
        "trainable_groups": sorted(live),
        "trainable_params": n_train,
        "total_params": total,
        "trainable_frac": n_train / max(total, 1),
    }


def param_groups_for_optimiser(model: nn.Module, groups: ParamGroups,
                               base_lr: float,
                               lr_scale: dict[str, float] | None = None) -> list[dict]:
    """Per-group learning rates.

    Arm B stage 1 runs the temporal stack and mT5 at a fraction of the base
    rate rather than hard-frozen -- the plan allows either. A scale of 0.0
    means frozen and the group is dropped from the optimiser entirely, which
    is not the same as passing lr=0 (weight decay and momentum would still
    move the weights).
    """
    scale = lr_scale or {}
    buckets: dict[float, list[nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lr = base_lr * float(scale.get(groups.assignment[name], 1.0))
        if lr == 0.0:
            p.requires_grad_(False)
            continue
        buckets.setdefault(lr, []).append(p)
    return [{"params": ps, "lr": lr} for lr, ps in sorted(buckets.items())]


# --- loading the real model -------------------------------------------------

# Uni_Sign takes an argparse Namespace. These are utils.get_args_parser's
# defaults for the fields the constructor and forward actually read.
UNISIGN_ARGS = {
    "hidden_dim": 256,
    "rgb_support": False,      # pose-only
    "label_smoothing": 0.2,
    "max_length": 256,
    "dataset": "How2Sign",     # only used to pick the prompt language;
                               # anything without "CSL" in it gives English
    "task": "SLT",
}


def build_args(**overrides) -> SimpleNamespace:
    cfg = dict(UNISIGN_ARGS)
    cfg.update(overrides)
    if "CSL" in str(cfg["dataset"]):
        raise ValueError(
            f"dataset={cfg['dataset']!r} contains 'CSL', which makes Uni-Sign "
            "prompt mT5 for Chinese output. Auslan translation targets are "
            "English."
        )
    return SimpleNamespace(**cfg)


def load_unisign(checkpoint: str | Path, repo_path: str | Path,
                 mt5_path: str | Path | None = None, device: str = "cpu",
                 **arg_overrides):
    """Instantiate Uni_Sign from its own repo and load the pose-only weights.

    Entry point verified against the repo: `models.Uni_Sign(args=...)`, with
    mT5 loaded from `config.mt5_path`. That path is patched here before models
    is imported, so the repo's checkout does not have to be edited.
    """
    repo = Path(repo_path).expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(
            f"Uni-Sign repo not found at {repo}. Clone it and pass --repo; the "
            "model definition must come from the repo the checkpoint was "
            "trained with."
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    config = importlib.import_module("config")
    if mt5_path:
        config.mt5_path = str(Path(mt5_path).expanduser().resolve())
    if not Path(config.mt5_path).is_dir():
        raise FileNotFoundError(
            f"mt5-base not found at {config.mt5_path}. Uni-Sign loads it with "
            "MT5ForConditionalGeneration.from_pretrained. Fetch it once:\n"
            "  huggingface-cli download google/mt5-base --local-dir "
            "<repo>/pretrained_weight/mt5-base\n"
            "then pass --mt5-path, or set mt5_path in the repo's config.py."
        )

    models = importlib.import_module("models")
    args = build_args(**arg_overrides)
    model = models.Uni_Sign(args=args)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    for key in ("model", "state_dict", "module"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)

    # A pose-only checkpoint legitimately lacks the RGB branch, so strict=False
    # is required -- which means the load can no longer fail loudly. Report it
    # instead, and let the caller decide what is acceptable.
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("       first missing:", missing[:8])
    if unexpected:
        print("       first unexpected:", unexpected[:8])
    if len(missing) > 0.5 * len(list(model.state_dict())):
        raise RuntimeError(
            "over half the model's parameters were not in the checkpoint. "
            "This is a wrong-checkpoint or wrong-constructor mismatch, not a "
            "pose-only/RGB difference."
        )
    return model.to(device)


def report(model: nn.Module, patterns: dict[str, list[str]] | None = None) -> None:
    groups = group_parameters(model, patterns)
    total = sum(groups.params.values())
    print(f"{total/1e6:.1f}M parameters over {len(groups.assignment)} tensors\n")
    for g in sorted(groups.params, key=lambda k: -groups.params[k]):
        n = groups.params[g]
        print(f"  {g:<14} {groups.counts[g]:>5} tensors  {n/1e6:>8.2f}M  "
              f"({100*n/max(total,1):.1f}%)")
        for name in groups.names(g)[:4]:
            print(f"      {name}")
        if groups.counts[g] > 4:
            print(f"      ... {groups.counts[g]-4} more")
    if groups.params.get("unmatched"):
        print("\n  'unmatched' is where the patterns failed. Fix "
              "group_patterns in your config until it is near zero, or the "
              "freeze policies are not doing what the arm specifies.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="inspect Uni-Sign parameter groups")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo", required=True, help="path to the Uni-Sign clone")
    ap.add_argument("--mt5-path", default=None,
                    help="local google/mt5-base directory")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)

    model = load_unisign(args.checkpoint, args.repo, args.mt5_path)
    if args.report:
        report(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
