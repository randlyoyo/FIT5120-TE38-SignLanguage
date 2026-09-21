#!/usr/bin/env python3
"""Training driver for Arms A / B / C.

The arms exist because the isolated-sign data cannot simply be poured in first.
MM-WLAuslan clips are citation forms: they start from rest, end at rest, carry
no coarticulation, and run long. Fine-tuning the whole model on them teaches it
that one clip is one gloss, which is precisely the sentence-level temporal
structure the Uni-Sign pre-training provides and this project cannot rebuild
from ~30 hours. So:

  arm_a           Auslan-Daily only. The control. Run it first, and do not
                  read B or C without it.
  arm_b_stage1    MM-WLAuslan, only the four per-part pose encoders training.
                  The temporal stack and mT5 are frozen or run at a fraction of
                  the base rate. This adapts the spatial front end to Auslan
                  handshapes without letting the isolated timing reach the
                  sequence model.
  arm_b_stage2    Auslan-Daily, everything unfrozen, initialised from stage 1.
                  The order is fixed. Running MM-WLAuslan after Auslan-Daily
                  would end training on citation-form timing.
  arm_c           One stage. Auslan-Daily is the primary loss and MM-WLAuslan
                  is mixed in as a low-weight auxiliary. No forgetting to
                  manage, so this is usually the steadier of the two.

Everything else -- the pose spec, the preprocessing, the eval -- is shared, so
that a difference between arms is a difference between arms.

Checkpoints and resuming
------------------------
Colab reclaims runtimes without warning. In this project one was removed after
about eight hours. A run therefore has to survive being killed at any instant.
Every `checkpoint.every_minutes` (and every `checkpoint.every_steps`, if set),
at each epoch boundary, and on Stop, the driver saves everything needed to
continue as if nothing had happened:

  * model and optimiser state
  * the step, the position inside the current epoch, and the position in the
    auxiliary stream
  * every RNG: Python, NumPy, torch CPU and CUDA

`--resume` picks up from the newest checkpoint. On CPU with num_workers=0, a
killed-and-resumed run ends bit-identical to an uninterrupted one;
test_resume.py checks exactly that. With worker processes, the data order is
still exact, but the random temporal subsampling of long clips draws
differently after a resume. That is a harmless augmentation difference.

Two things had to change to make that possible:

  * Batch order is derived from (seed, epoch), not from the global RNG, so a
    resumed run can skip exactly the batches the killed one had already
    consumed.
  * The auxiliary stream used itertools.cycle, which caches the first pass and
    replays it. The aux data was therefore never reshuffled after the first
    pass, and every aux batch was held in memory. It is now reshuffled every
    aux epoch, and its position is checkpointed.

Guards, because every one of these failure modes is silent otherwise:

  * starting over existing checkpoints without --resume refuses (use --fresh
    to move them aside)
  * resuming with a different config refuses (a fingerprint covers the config,
    the dataset sizes, --init-from and --smoke-steps)
  * checkpoints are written to a temp file and renamed, so a kill during a
    save never leaves a truncated "latest" behind

Decoding
--------
The config's `decode` block adds generation options on top of Uni-Sign's own
generate(), which passes only max_new_tokens and num_beams: for example
no_repeat_ngram_size and repetition_penalty, against the loops an
under-trained decoder falls into. It changes how a model is decoded, never what
it learns, so it is left out of the resume fingerprint. Empty means exactly
Uni-Sign's call.

Precision
---------
Training accepts ``precision: fp32`` (the default) or ``precision: bf16``.
BF16 uses CUDA autocast while keeping model parameters and AdamW state in
FP32; validation decoding remains FP32 so metrics stay directly comparable.
The precision is part of the run fingerprint, so a run cannot accidentally
resume under a different numerical training mode.

`--eval-only` decodes the validation split with a finished run's weights
(<out>/checkpoint.pt, or --weights) and writes <out>/eval_<tag>/ -- the run's
own predictions and metrics are left alone -- so one model can be scored under
several decoding settings without training again.

A Uni-Sign checkpoint is large: 588M parameters plus AdamW's two moment
buffers, about 7 GB. Put output_dir on Drive so it outlives the runtime.
checkpoint.keep defaults to 2: the newest, plus one to fall back to if the
newest was saved after training had already gone wrong (a loss gone to NaN, a
bad learning rate). To fall back, delete the newest ckpt_step*.pt and --resume
continues from the other.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec  # noqa: E402
from dataset import SignPoseDataset, collate, target_text_for  # noqa: E402
from evaluate import evaluate_records, format_report  # noqa: E402
from model_adapter import (DEFAULT_GROUP_PATTERNS, apply_freeze_policy,  # noqa: E402
                           group_parameters, param_groups_for_optimiser)

CKPT_DEFAULTS = {"every_minutes": 30.0, "every_steps": None, "keep": 2, "dir": None}
# Config keys that may differ between a run and its resumption. `decode` only
# changes how the finished model is decoded, never what it learns.
_VOLATILE_CFG = ("output_dir", "device", "num_workers", "log_every", "checkpoint",
                 "decode")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Match Uni-Sign's reproducibility settings for dynamic-length pose input.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(device: torch.device, precision: str):
    """Return the autocast context for a training forward/backward pass.

    Parameters stay in FP32 and AdamW keeps FP32 state; only eligible CUDA
    operations run in BF16. This makes a BF16 run load the same model-only
    checkpoint as an FP32 run and avoids the numerical fragility of converting
    the whole model with ``model.to(torch.bfloat16)``. BF16 is intentionally
    strict about the device: silently falling back to FP32 would make a run
    labelled BF16 misleading.
    """
    name = str(precision or "fp32").strip().lower()
    if name in {"fp32", "float32", "none"}:
        return nullcontext()
    if name not in {"bf16", "bfloat16"}:
        raise ValueError(
            f"unsupported precision={precision!r}; use 'fp32' or 'bf16'"
        )
    if device.type != "cuda":
        raise RuntimeError("precision=bf16 requires a CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("precision=bf16 requires a CUDA device with BF16 support")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def move(batch: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# --- backends ---------------------------------------------------------------
#
# A backend owns the two things that differ between the smoke model and the
# real one: how a batch becomes a loss, and how a batch becomes text.


class SmokeBackend:
    name = "smoke"

    def __init__(self, train_texts: list[str], device, **kw):
        from smoke_model import SmokeSLT, WordVocab, pad_targets

        self.vocab = WordVocab(train_texts)
        self.pad_targets = pad_targets
        self.model = SmokeSLT(len(self.vocab), **kw).to(device)
        self.device = device
        self.crit = nn.CrossEntropyLoss(ignore_index=0)

    def loss(self, batch: dict) -> torch.Tensor:
        tgt = self.pad_targets([self.vocab.encode(t) for t in batch["text"]],
                               self.device)
        logits = self.model(batch, tgt)
        return self.crit(logits.reshape(-1, logits.shape[-1]),
                         tgt[:, 1:].reshape(-1))

    def generate(self, batch: dict) -> list[str]:
        return [self.vocab.decode(ids) for ids in self.model.generate(batch)]


class UniSignBackend:
    """The real model.

    Verified against models.py: `Uni_Sign.forward(src_input, tgt_input)` returns
    a dict carrying 'loss' plus the 'inputs_embeds'/'attention_mask' its own
    generate() consumes, and `tgt_input['gt_sentence']` is a list of plain
    strings that the model tokenises internally. Our collate already emits the
    four part tensors and the attention mask under the exact keys forward reads,
    so nothing is remapped here.
    """

    name = "unisign"

    def __init__(self, device, checkpoint: str, repo: str,
                 mt5_path: str | None = None, max_new_tokens: int = 100,
                 num_beams: int = 5, **arg_overrides):
        from model_adapter import load_unisign

        self.model = load_unisign(checkpoint, repo, mt5_path,
                                  device=str(device), **arg_overrides)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.decode: dict = {}          # the config's `decode` block, see build_backend

    @staticmethod
    def _split(batch: dict) -> tuple[dict, dict]:
        src = {k: v for k, v in batch.items()
               if k in spec.PART_ORDER or k in ("attention_mask",
                                                "src_length_batch", "name_batch")}
        # gt_gloss is read by the repo's collate but not by forward; pass the
        # texts through as sentences, which is what SLT trains against.
        tgt = {"gt_sentence": batch["text"], "gt_gloss": batch["text"]}
        return src, tgt

    def loss(self, batch: dict) -> torch.Tensor:
        src, tgt = self._split(batch)
        return self.model(src, tgt)["loss"]

    def generate(self, batch: dict) -> list[str]:
        src, tgt = self._split(batch)
        out = self.model(src, tgt)
        # The same call as Uni_Sign.generate (models.py), plus the `decode`
        # options, which that method has no way to pass on.
        ids = self.model.mt5_model.generate(
            inputs_embeds=out["inputs_embeds"], attention_mask=out["attention_mask"],
            max_new_tokens=self.max_new_tokens, num_beams=self.num_beams, **self.decode)
        return self.model.mt5_tokenizer.batch_decode(ids, skip_special_tokens=True)


BACKENDS = {"smoke": SmokeBackend, "unisign": UniSignBackend}


# --- data -------------------------------------------------------------------

def read_exclude(path) -> frozenset[str]:
    """uids to leave out, one per line -- verify_pose.py --exclude-out."""
    if not path:
        return frozenset()
    lines = (s.strip() for s in Path(path).read_text().splitlines())
    return frozenset(s for s in lines if s and not s.startswith("#"))


def build_dataset(cfg: dict, split_key: str, common: dict) -> SignPoseDataset | None:
    sel = cfg.get(split_key)
    if not sel:
        return None
    # Validation subsamples long clips deterministically. Upstream uses a random
    # subsample everywhere; keeping that at eval time makes the score depend on
    # the seed, and a metric that moves when re-run cannot compare arms.
    return SignPoseDataset(
        common["manifest"], common["npz_dir"],
        datasets=sel.get("datasets"), subsets=sel.get("subsets"),
        splits=sel.get("splits"),
        max_length=common.get("max_length", 256),
        gloss_text_mode=common.get("gloss_text_mode", "strip_paren"),
        deterministic=(split_key != "train"),
        require_npz=common.get("require_npz", True),
        exclude=read_exclude(common.get("exclude")),
    )


def loader(ds, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    """Unshuffled loader, for evaluation."""
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, collate_fn=collate,
                      drop_last=shuffle)


def epoch_order(n: int, bs: int, seed: int, stream: int, epoch: int,
                skip_batches: int = 0) -> list[int]:
    """This epoch's sample order, from (seed, stream, epoch) alone.

    Independent of the global RNG, so a resumed run reproduces the permutation
    of the epoch it was killed in and skips exactly the batches already used.
    Truncated to whole batches, which is what drop_last did before.
    """
    full = (n // bs) * bs
    rng = np.random.default_rng([seed, stream, epoch])
    return rng.permutation(n)[skip_batches * bs: full].tolist()


def ordered_loader(ds, bs: int, workers: int, indices: list[int],
                   seed: int, stream: int, epoch: int) -> DataLoader:
    # A private generator seeds the worker processes, so building a loader
    # does not consume the global RNG -- a resumed run and an uninterrupted one
    # see the same global RNG sequence however many loaders were built.
    g = torch.Generator()
    g.manual_seed(seed * 1_000_003 + stream * 10_007 + epoch)
    return DataLoader(ds, batch_size=bs, sampler=indices, num_workers=workers,
                      collate_fn=collate, drop_last=True, generator=g)


def aux_batches(ds, bs: int, workers: int, seed: int, consumed: int):
    """Endless auxiliary stream, reshuffled every aux epoch, resumable."""
    per_epoch = len(ds) // bs
    if per_epoch == 0:
        raise ValueError("the aux set is smaller than one batch")
    ep, off = divmod(consumed, per_epoch)
    while True:
        idx = epoch_order(len(ds), bs, seed, 1, ep, off)
        yield from ordered_loader(ds, bs, workers, idx, seed, 1, ep)
        ep, off = ep + 1, 0


# --- checkpoints ------------------------------------------------------------

def run_fingerprint(cfg: dict, args, n_train: int, n_aux: int) -> str:
    """What must match for a resume to be a continuation of the same run."""
    payload = {"cfg": {k: v for k, v in cfg.items() if k not in _VOLATILE_CFG},
               "smoke_steps": args.smoke_steps,
               "init_from": str(args.init_from) if args.init_from else None,
               "n_train": n_train, "n_aux": n_aux,
               "spec": spec.SCHEMA_FINGERPRINT}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def rng_state() -> dict:
    """Every RNG, in plain types so the checkpoint loads with weights_only=True."""
    py = random.getstate()
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    st = {"python": {"version": py[0], "state": list(py[1]), "gauss": py[2]},
          "numpy": {"kind": kind, "keys": torch.from_numpy(keys.astype(np.int64)),
                    "pos": int(pos), "has_gauss": int(has_gauss),
                    "cached": float(cached)},
          "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict) -> None:
    p = st["python"]
    random.setstate((p["version"], tuple(p["state"]), p["gauss"]))
    n = st["numpy"]
    np.random.set_state((n["kind"], n["keys"].numpy().astype(np.uint32),
                         n["pos"], n["has_gauss"], n["cached"]))
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    found = sorted(ckpt_dir.glob("ckpt_step*.pt"))    # zero-padded: sorts by step
    return found[-1] if found else None


def save_checkpoint(ckpt_dir: Path, keep: int, payload: dict, why: str) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final = ckpt_dir / f"ckpt_step{payload['state']['step']:09d}.pt"
    tmp = final.with_name(final.name + ".part")
    t = time.time()
    with tmp.open("wb") as fh:
        torch.save(payload, fh)
    os.replace(tmp, final)                 # atomic: never a half-written "latest"
    for old in sorted(ckpt_dir.glob("ckpt_step*.pt"))[:-max(int(keep), 1)]:
        old.unlink(missing_ok=True)
    print(f"  [checkpoint] step {payload['state']['step']} ({why}) -> {final.name}, "
          f"{final.stat().st_size / 1e6:.1f} MB in {time.time() - t:.1f}s", flush=True)


# --- loop -------------------------------------------------------------------

@torch.no_grad()
def run_eval(backend, ds, batch_size: int, workers: int, device,
             limit: int | None = None) -> list[dict]:
    backend.model.eval()
    dl = loader(ds, batch_size, False, workers)
    records: list[dict] = []
    for batch in dl:
        b = move(batch, device)
        hyps = backend.generate(b)
        for uid, dset, sub, ref, hyp in zip(batch["uid"], batch["dataset"],
                                            batch["subset"], batch["text"], hyps):
            records.append({"uid": uid, "dataset": dset, "subset": sub,
                            "ref": ref, "hyp": hyp})
        if limit and len(records) >= limit:
            break
    backend.model.train()
    return records


def training_texts(train_ds, aux_ds) -> list[str]:
    texts = [train_ds.target_text(r) for r in train_ds.rows]
    if aux_ds:
        texts += [aux_ds.target_text(r) for r in aux_ds.rows]
    return texts


def build_backend(cfg: dict, device):
    common = cfg["data"]
    # The smoke backend builds its own word vocabulary, and it must be built
    # over the WHOLE manifest rather than the current stage's split. Arm B
    # stage 2 loads stage 1's checkpoint, and a vocabulary derived per stage
    # gives the two stages different embedding shapes -- the load then fails,
    # or worse, silently maps the same row to a different word. The real
    # backend has no such problem: mT5's tokenizer is fixed and shared.
    from manifest import read_manifest
    corpus_texts = [target_text_for(r, common.get("gloss_text_mode", "strip_paren"))
                    for r in read_manifest(Path(common["manifest"]))]

    bcfg = dict(cfg["backend"])
    backend_cls = BACKENDS[bcfg.pop("name")]
    backend = (backend_cls(corpus_texts, device, **bcfg)
               if backend_cls is SmokeBackend
               else backend_cls(device=device, **bcfg))
    decode = dict(cfg.get("decode") or {})
    if decode and backend_cls is SmokeBackend:
        print(f"decode {decode} ignored: the smoke backend decodes greedily")
    elif decode:
        backend.decode = decode
    return backend


def evaluate_only(cfg: dict, args, device, out_dir: Path,
                  train_ds, val_ds, aux_ds) -> int:
    """Decode the val split with a finished run's weights, into <out>/eval_<tag>/.

    Touches neither the checkpoints nor the run's own predictions and metrics.
    metrics.json is written last, so its presence means the evaluation is
    complete.
    """
    if not val_ds:
        raise SystemExit("--eval-only needs a val split in the config")
    weights = Path(args.weights) if args.weights else out_dir / "checkpoint.pt"
    if not weights.exists():
        raise SystemExit(f"{weights} not found: the run has not finished yet, "
                         "or pass --weights")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if state.get("spec_fingerprint") != spec.SCHEMA_FINGERPRINT:
        raise SystemExit(f"{weights} was trained on poses with spec fingerprint "
                         f"{state.get('spec_fingerprint')}, these are "
                         f"{spec.SCHEMA_FINGERPRINT}")
    backend = build_backend(cfg, device)
    backend.model.load_state_dict(state["model"])
    decode = dict(cfg.get("decode") or {})
    print(f"--eval-only: weights {weights} (arm={state.get('arm')} "
          f"step={state.get('step')}), decode {decode or 'plain'}")

    records = run_eval(backend, val_ds, int(cfg["optim"]["batch_size"]),
                       int(cfg.get("num_workers", 0)), device, args.eval_limit)
    metrics = evaluate_records(records, train_texts=training_texts(train_ds, aux_ds))
    dest = out_dir / f"eval_{args.eval_tag}"
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "predictions.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    settings = {"weights": str(weights), "step": state.get("step"),
                "training_precision": cfg.get("precision", "fp32"),
                "eval_precision": "fp32", "decode": decode,
                **{k: cfg["backend"].get(k) for k in ("num_beams", "max_new_tokens")
                   if k in cfg["backend"]}}
    (dest / "settings.json").write_text(json.dumps(settings, indent=2))
    (dest / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(format_report(metrics))
    print(f"-> {dest}")
    return 0


def train(cfg: dict, args) -> int:
    common = cfg["data"]
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    device = pick_device(args.device or cfg.get("device"))
    precision = str(cfg.get("precision", "fp32")).strip().lower()
    if precision == "bfloat16":
        precision = "bf16"
    # Validate before constructing the model or starting workers. The returned
    # context is deliberately not reused: each forward gets a fresh context.
    autocast_context(device, precision)
    out_dir = Path(args.out or cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"arm={cfg['arm']} device={device} precision={precision} out={out_dir}")

    train_ds = build_dataset(cfg, "train", common)
    val_ds = build_dataset(cfg, "val", common)
    aux_ds = build_dataset(cfg, "aux", common)
    print(f"train={len(train_ds)}" + (f" aux={len(aux_ds)}" if aux_ds else "")
          + (f" val={len(val_ds)}" if val_ds else ""))
    if common.get("exclude"):
        n_ex = sum(d.excluded for d in (train_ds, val_ds, aux_ds) if d)
        print(f"left out {n_ex} clips listed in {common['exclude']}")
    if args.eval_only:
        return evaluate_only(cfg, args, device, out_dir, train_ds, val_ds, aux_ds)

    # --- checkpoint bookkeeping first, so a bad invocation fails before any
    # --- model is built
    ck_cfg = {**CKPT_DEFAULTS, **(cfg.get("checkpoint") or {})}
    ckpt_dir = Path(args.ckpt_dir or ck_cfg.get("dir") or out_dir / "checkpoints")
    for stray in ckpt_dir.glob("*.part"):              # a save the kill interrupted
        stray.unlink()
    existing = latest_checkpoint(ckpt_dir) if ckpt_dir.exists() else None
    if existing and not args.resume:
        if args.fresh:
            aside = ckpt_dir.with_name(
                f"{ckpt_dir.name}.old-{time.strftime('%Y%m%d-%H%M%S')}")
            ckpt_dir.rename(aside)
            print(f"--fresh: moved the previous checkpoints aside to {aside}")
            existing = None
        else:
            raise SystemExit(
                f"{ckpt_dir} already holds {existing.name}. Pass --resume to "
                "continue that run, or --fresh to move it aside and start over.")
    fingerprint = run_fingerprint(cfg, args, len(train_ds), len(aux_ds) if aux_ds else 0)

    train_texts = training_texts(train_ds, aux_ds)
    backend = build_backend(cfg, device)
    model = backend.model

    resuming = bool(args.resume and existing)
    if args.init_from and not resuming:
        state = torch.load(args.init_from, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        print(f"initialised from {args.init_from} "
              f"(arm={state.get('arm')} step={state.get('step')})")

    # Freezing. Reported before the first step, always -- a stage that trains
    # nothing still draws a loss curve.
    patterns = cfg.get("group_patterns") or DEFAULT_GROUP_PATTERNS
    groups = group_parameters(model, patterns)
    trainable = cfg["optim"].get("trainable_groups") or sorted(
        g for g in groups.params if g != "unmatched")
    summary = apply_freeze_policy(model, groups, trainable)
    print(f"trainable groups={summary['trainable_groups']} "
          f"{summary['trainable_params']/1e6:.2f}M / "
          f"{summary['total_params']/1e6:.2f}M "
          f"({summary['trainable_frac']:.1%})")

    base_lr = float(cfg["optim"]["lr"])
    pgroups = param_groups_for_optimiser(model, groups, base_lr,
                                         cfg["optim"].get("lr_scale"))
    # Snapshot the per-group rates BEFORE constructing the optimiser. AdamW
    # keeps the dicts it is handed, so opt.param_groups[i] IS pgroups[i]; using
    # pgroups as the "base" while writing into opt.param_groups multiplies the
    # rate by the schedule every step instead of setting it, and the learning
    # rate decays to zero within a few dozen steps. The loss curve just flattens.
    base_lrs = [float(g["lr"]) for g in pgroups]
    # Print what each group will actually train at. "trainable" above is a
    # yes/no; a group held at 0.1x is the difference between Arm B stage 1 and
    # a naive two-stage run, and it is not visible anywhere else.
    scale_cfg = cfg["optim"].get("lr_scale") or {}
    for g in sorted(groups.params):
        if g == "unmatched" and not groups.params.get("unmatched"):
            continue
        if g not in summary["trainable_groups"]:
            eff = "FROZEN"
        else:
            eff = f"{base_lr * float(scale_cfg.get(g, 1.0)):.2e}"
        print(f"    {g:<14} {groups.params[g]/1e6:>7.2f}M  lr={eff}")

    opt = torch.optim.AdamW(pgroups, lr=base_lr,
                            weight_decay=float(cfg["optim"].get("weight_decay", 0.01)),
                            betas=tuple(cfg["optim"].get("betas", (0.9, 0.999))),
                            eps=float(cfg["optim"].get("eps", 1.0e-8)))

    bs = int(cfg["optim"]["batch_size"])
    grad_accum = int(cfg["optim"].get("gradient_accumulation_steps", 1))
    if grad_accum < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    workers = int(cfg.get("num_workers", 0))
    epochs = int(cfg["optim"]["epochs"])
    aux_weight = float(cfg["optim"].get("aux_weight", 0.0))
    clip = float(cfg["optim"].get("grad_clip", 1.0))
    if aux_ds and aux_weight <= 0:
        raise ValueError("an aux set is configured but aux_weight is 0")

    per_epoch = len(train_ds) // bs
    if per_epoch == 0:
        raise ValueError("the training set is smaller than one batch")
    # The official recipe uses batch 8 per GPU on four GPUs. On one A100,
    # accumulate four micro-batches so the optimiser sees the same effective
    # batch of 32. Drop a partial accumulation group just as DataLoader drops
    # a partial batch; checkpoints are only written after an optimiser step.
    updates_per_epoch = per_epoch // grad_accum
    if updates_per_epoch == 0:
        raise ValueError("the training set is smaller than one accumulation group")
    microbatches_per_epoch = updates_per_epoch * grad_accum
    total_steps = args.smoke_steps or epochs * updates_per_epoch
    warmup = int(cfg["optim"].get("warmup_frac", 0.05) * total_steps)
    print(f"batch={bs} gradient_accumulation={grad_accum} "
          f"effective_batch={bs * grad_accum} updates/epoch={updates_per_epoch}")

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    # --- resume -------------------------------------------------------------
    state = {"step": 0, "epoch": 0, "batch_in_epoch": 0, "aux_consumed": 0}
    done = False
    log_path = out_dir / "train_log.jsonl"
    if resuming:
        ck = torch.load(existing, map_location="cpu", weights_only=True)
        if ck["fingerprint"] != fingerprint and not args.force_resume:
            raise SystemExit(
                f"{existing.name} belongs to a different run: the config, the "
                "dataset sizes, --init-from or --smoke-steps changed "
                f"(fingerprint {ck['fingerprint']} vs {fingerprint}). Resuming "
                "would splice two runs together. Use --fresh to start over, or "
                "--force-resume if the change is deliberate.")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        state = dict(ck["state"])
        done = bool(ck.get("done"))
        set_rng_state(ck["rng"])          # last: model and optimiser are built
        if log_path.exists():             # drop log lines past the checkpoint
            kept = [line for line in log_path.read_text().splitlines()
                    if line.strip() and json.loads(line).get("step", 0) <= state["step"]]
            log_path.write_text("".join(line + "\n" for line in kept))
        print(f"resumed from {existing.name}: step {state['step']}/{total_steps}, "
              f"epoch {state['epoch']} batch {state['batch_in_epoch']}"
              + (", run already finished" if done else ""))
    elif args.resume:
        print(f"--resume: no checkpoint in {ckpt_dir}; starting fresh")

    last_save = {"time": time.time(), "step": state["step"] if resuming else -1}

    def save(why: str, finished: bool = False) -> None:
        payload = {"model": model.state_dict(), "optimizer": opt.state_dict(),
                   "state": dict(state), "rng": rng_state(),
                   "fingerprint": fingerprint, "arm": cfg["arm"], "done": finished,
                   "spec_fingerprint": spec.SCHEMA_FINGERPRINT}
        save_checkpoint(ckpt_dir, int(ck_cfg["keep"]), payload, why)
        last_save["time"], last_save["step"] = time.time(), state["step"]

    every_steps = ck_cfg.get("every_steps")
    every_min = ck_cfg.get("every_minutes")
    aux_it = (aux_batches(aux_ds, bs, workers, seed, state["aux_consumed"])
              if aux_ds else None)

    t0 = time.time()
    model.train()
    try:
        while not done and state["epoch"] < epochs:
            idx = epoch_order(len(train_ds), bs, seed, 0, state["epoch"],
                              state["batch_in_epoch"])
            stop = False
            opt.zero_grad(set_to_none=True)
            primary_sum = 0.0
            aux_sum = 0.0
            for batch in ordered_loader(train_ds, bs, workers, idx, seed, 0,
                                        state["epoch"]):
                if state["batch_in_epoch"] >= microbatches_per_epoch:
                    break
                scale = lr_at(state["step"])
                for gp, base in zip(opt.param_groups, base_lrs):
                    gp["lr"] = base * scale

                b = move(batch, device)
                with autocast_context(device, precision):
                    loss = backend.loss(b)
                    primary = float(loss.detach())
                    primary_sum += primary
                    aux_val = None
                    if aux_it is not None:
                        ab = move(next(aux_it), device)
                        state["aux_consumed"] += 1
                        aux_loss = backend.loss(ab)
                        aux_val = float(aux_loss.detach())
                        aux_sum += aux_val
                        loss = loss + aux_weight * aux_loss

                # Divide before backward so accumulation is the mean of the
                # micro-batch gradients, matching a single larger batch.
                (loss / grad_accum).backward()
                state["batch_in_epoch"] += 1
                if state["batch_in_epoch"] % grad_accum:
                    continue

                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], clip)
                opt.step()
                state["step"] += 1
                opt.zero_grad(set_to_none=True)

                if state["step"] % int(cfg.get("log_every", 50)) == 0:
                    rec = {"step": state["step"], "epoch": state["epoch"],
                           "loss": round(primary_sum / grad_accum, 4),
                           "lr": round(opt.param_groups[0]["lr"], 7),
                           "sec": round(time.time() - t0, 1)}
                    if aux_it is not None:
                        rec["aux_loss"] = round(aux_sum / grad_accum, 4)
                    print("  " + json.dumps(rec))
                    with log_path.open("a") as fh:
                        fh.write(json.dumps(rec) + "\n")

                primary_sum = 0.0
                aux_sum = 0.0

                # Test hooks (test_resume.py). A kill loses everything since the
                # last checkpoint, as when Colab reclaims the runtime; Stop
                # saves on the way out.
                if args.crash_after_steps and state["step"] >= args.crash_after_steps:
                    print(f"  [test] simulated kill after step {state['step']}", flush=True)
                    os._exit(137)
                if args.interrupt_after_steps and state["step"] >= args.interrupt_after_steps:
                    raise KeyboardInterrupt

                due = ((every_steps and state["step"] % int(every_steps) == 0)
                       or (every_min and time.time() - last_save["time"]
                           >= 60 * float(every_min)))
                if due:
                    save("periodic")
                if args.smoke_steps and state["step"] >= args.smoke_steps:
                    stop = True
                    break
                if state["batch_in_epoch"] >= microbatches_per_epoch:
                    break
            if stop:
                break
            state["epoch"] += 1
            state["batch_in_epoch"] = 0
            if state["epoch"] < epochs and last_save["step"] != state["step"]:
                save("epoch end")
            # Deliberate pause at an epoch boundary, so a first epoch can be
            # inspected before the remaining ones are paid for. The learning
            # rate schedule is unaffected: total_steps still comes from
            # optim.epochs, and this flag is not part of the resume
            # fingerprint, so `--resume` without it continues the same run.
            if (args.stop_after_epochs and state["epoch"] >= args.stop_after_epochs
                    and state["epoch"] < epochs):
                if last_save["step"] != state["step"]:
                    save("epoch pause")
                print(f"paused after epoch {state['epoch'] - 1}; "
                      f"continue with --resume", flush=True)
                return 130
        if not done:
            done = True
            save("finished", finished=True)
    except KeyboardInterrupt:
        # A Stop can arrive between two micro-batches of an accumulation
        # group. Those gradients have not changed the model yet, so rewind the
        # consumed group and resume it from the next checkpoint boundary.
        remainder = state["batch_in_epoch"] % grad_accum
        if remainder:
            state["batch_in_epoch"] -= remainder
            if aux_it is not None:
                state["aux_consumed"] -= remainder
            opt.zero_grad(set_to_none=True)
        save("interrupted")
        print("stopped; continue with --resume")
        return 130

    ckpt = out_dir / "checkpoint.pt"               # weights only, for --init-from
    torch.save({"model": model.state_dict(), "arm": cfg["arm"],
                "step": state["step"], "spec_fingerprint": spec.SCHEMA_FINGERPRINT},
               ckpt)
    print(f"saved {ckpt}")

    if val_ds and not args.no_eval:
        records = run_eval(backend, val_ds, bs, workers, device, args.eval_limit)
        pred_path = out_dir / "predictions.jsonl"
        with pred_path.open("w") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        metrics = evaluate_records(records, train_texts=train_texts)
        print(format_report(metrics))
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--init-from", default=None,
                    help="checkpoint to start from (Arm B stage 2)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the newest checkpoint if there is one; "
                         "safe to pass on the very first run too")
    ap.add_argument("--fresh", action="store_true",
                    help="move existing checkpoints aside and start over")
    ap.add_argument("--force-resume", action="store_true",
                    help="resume even though the config fingerprint changed")
    ap.add_argument("--ckpt-dir", type=Path, default=None,
                    help="where checkpoints go (default: <out>/checkpoints)")
    ap.add_argument("--no-eval", action="store_true",
                    help="skip the final validation decode")
    ap.add_argument("--smoke-steps", type=int, default=None,
                    help="stop after N steps -- harness check, not a result")
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--eval-only", action="store_true",
                    help="no training: decode the val split with a finished run's "
                         "weights and write <out>/eval_<tag>/")
    ap.add_argument("--weights", type=Path, default=None,
                    help="with --eval-only: the weights to load "
                         "(default <out>/checkpoint.pt)")
    ap.add_argument("--eval-tag", default="eval",
                    help="with --eval-only: names the output folder, eval_<tag>")
    ap.add_argument("--stop-after-epochs", type=int, default=None,
                    help="stop cleanly after N epochs and exit 130, leaving a "
                         "checkpoint to continue from with --resume. The LR "
                         "schedule still spans optim.epochs, so pausing does "
                         "not change what is trained")
    ap.add_argument("--crash-after-steps", type=int, default=None,
                    help="TESTING: hard-exit after N steps, no checkpoint")
    ap.add_argument("--interrupt-after-steps", type=int, default=None,
                    help="TESTING: behave as if Stop was pressed after N steps")
    ap.add_argument("--set", nargs="*", action="extend", default=[],
                    help="config overrides, e.g. optim.lr=1e-5. Repeatable: every "
                         "--set is applied, later values win")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    for override in args.set:
        key, _, val = override.partition("=")
        node = cfg
        *path, leaf = key.split(".")
        for p in path:
            node = node.setdefault(p, {})
        try:
            node[leaf] = yaml.safe_load(val)
        except Exception:
            node[leaf] = val
    return train(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
