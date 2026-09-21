"""Training modes (A/B/C), freezing and differential learning rates.

Parameter groups, by name:
    text_embed         text_encoder.embed.*                  (192M, SentencePiece vocab)
    enc_block_<i>      text_encoder.stack.block.<i>.*
    enc_final_norm     text_encoder.stack.final_layer_norm.*
    dec_block_<i>      pose_decoder.stack.block.<i>.*        (block 0 owns the relative position bias)
    dec_final_norm     pose_decoder.stack.final_layer_norm.*
    new                pose_decoder.pose_in / pose_out / counter_out

MODE A  (head-only)   new [+ last `train_last_decoder_layers` decoder blocks + final norm]
MODE B  (partial)     new + last N decoder blocks + dec final norm
                          + last M encoder blocks (+ enc final norm if M > 0)
MODE C  (full)        everything except text_embed (unless freeze_embeddings: false),
                      pretrained lr multiplied by lr_scale (default 0.1)

"new" parameters get lr_new; loaded parameters get lr_pretrained.
No weight decay on norms or the counter/pose biases.
"""

from __future__ import annotations

import re

import torch.nn as nn

_ENC = re.compile(r"^text_encoder\.stack\.block\.(\d+)\.")
_DEC = re.compile(r"^pose_decoder\.stack\.block\.(\d+)\.")


def group_of(name: str) -> str:
    if name.startswith(("pose_decoder.pose_in.", "pose_decoder.pose_out.", "pose_decoder.counter_out.")):
        return "new"
    if name.startswith("text_encoder.embed."):
        return "text_embed"
    if m := _ENC.match(name):
        return f"enc_block_{int(m.group(1))}"
    if m := _DEC.match(name):
        return f"dec_block_{int(m.group(1))}"
    if name.startswith("text_encoder.stack.final_layer_norm"):
        return "enc_final_norm"
    if name.startswith("pose_decoder.stack.final_layer_norm"):
        return "dec_final_norm"
    return "unknown"


def trainable_groups(mode: str, tcfg: dict, n_enc: int, n_dec: int) -> set[str]:
    mode = mode.upper()
    g: set[str] = {"new"}
    if mode == "A":
        k = int(tcfg.get("mode_a", {}).get("train_last_decoder_layers", 0))
        g |= {f"dec_block_{i}" for i in range(n_dec - k, n_dec)}
        if k > 0:
            g.add("dec_final_norm")
    elif mode == "B":
        b = tcfg.get("mode_b", {})
        nd = int(b.get("train_last_decoder_layers", 4))
        ne = int(b.get("train_last_encoder_layers", 2))
        g |= {f"dec_block_{i}" for i in range(max(0, n_dec - nd), n_dec)} | {"dec_final_norm"}
        g |= {f"enc_block_{i}" for i in range(max(0, n_enc - ne), n_enc)}
        if ne > 0:
            g.add("enc_final_norm")
    elif mode == "C":
        g |= {f"dec_block_{i}" for i in range(n_dec)} | {f"enc_block_{i}" for i in range(n_enc)}
        g |= {"enc_final_norm", "dec_final_norm"}
        if not tcfg.get("mode_c", {}).get("freeze_embeddings", True):
            g.add("text_embed")
    else:
        raise ValueError(f"training.mode must be A, B or C, got {mode!r}")
    return g


def apply_mode(model: nn.Module, tcfg: dict) -> dict:
    n_enc = len(model.text_encoder.stack.block)
    n_dec = len(model.pose_decoder.stack.block)
    live = trainable_groups(tcfg.get("mode", "B"), tcfg, n_enc, n_dec)
    unknown = [n for n, _ in model.named_parameters() if group_of(n) == "unknown"]
    if unknown:
        raise RuntimeError(f"parameters not covered by any group: {unknown[:5]}")
    n_train = n_total = 0
    for name, p in model.named_parameters():
        p.requires_grad_(group_of(name) in live)
        n_total += p.numel()
        n_train += p.numel() if p.requires_grad else 0
    if n_train == 0:
        raise RuntimeError("training mode leaves zero trainable parameters")
    return {"mode": tcfg.get("mode", "B"), "trainable_groups": sorted(live),
            "trainable_params": n_train, "total_params": n_total}


def optimizer_param_groups(model: nn.Module, tcfg: dict) -> list[dict]:
    mode = str(tcfg.get("mode", "B")).upper()
    lr_new = float(tcfg.get("lr_new", 1e-4))
    lr_pre = float(tcfg.get("lr_pretrained", 1e-5))
    if mode == "C":
        lr_pre *= float(tcfg.get("mode_c", {}).get("lr_scale", 0.1))
    wd = float(tcfg.get("weight_decay", 0.01))
    buckets: dict[tuple[str, bool], list] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        kind = "new" if group_of(name) == "new" else "pretrained"
        no_decay = p.dim() < 2 or "layer_norm" in name
        buckets.setdefault((kind, no_decay), []).append(p)
    groups = []
    for (kind, no_decay), ps in sorted(buckets.items()):
        groups.append({"params": ps, "lr": lr_new if kind == "new" else lr_pre,
                       "weight_decay": 0.0 if no_decay else wd, "name": f"{kind}{'_nodecay' if no_decay else ''}"})
    return groups
