#!/usr/bin/env python3
"""Supervised fine-tuning of the pretrained text->pose model on Auslan.

    python train.py --config configs/auslan_sft.yaml
    torchrun --nproc_per_node 2 train.py --config configs/auslan_sft.yaml     (optional DDP)

Order of operations (each step fails loudly):
  1. splits + normalisation stats (train split only)
  2. build model, load pretrained weights, print loaded/missing/unexpected/
     trainable counts, STOP if the pretrained load is incomplete
  3. apply training mode A/B/C, differential learning rates
  4. train with AMP, gradient clipping, warmup+cosine (or plateau) schedule
  5. validate each epoch: teacher-forced losses + free-running DTW on a fixed subset
  6. save latest.pt every epoch, best.pt on improvement; early stopping
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slp_data.auslan_dataset import AuslanPoseDataset, Collator, build_tokenizer, load_or_compute_stats, read_split  # noqa: E402
from slp_models.freeze import optimizer_param_groups  # noqa: E402
from slp_models.losses import PoseLoss  # noqa: E402
from slp_utils.build import build_model, pretrained_paths  # noqa: E402
from slp_utils.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from slp_utils.config import clean_for_save, load_config, resolve_path, set_seed  # noqa: E402
from slp_utils.metrics import aggregate, sequence_metrics  # noqa: E402
from slp_utils.unisign_format import pad_with_last_frame, to_unisign_parts  # noqa: E402


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def log(*a, **k):
    if is_main():
        print(*a, **k, flush=True)


def setup_device(tcfg: dict) -> tuple[torch.device, bool]:
    ddp = "LOCAL_RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1
    want = str(tcfg.get("device", "cuda"))
    if ddp:
        backend = "nccl" if torch.cuda.is_available() and want.startswith("cuda") else "gloo"
        dist.init_process_group(backend)
        rank = int(os.environ["LOCAL_RANK"])
        if backend == "nccl":
            torch.cuda.set_device(rank)
            return torch.device("cuda", rank), True
        return torch.device("cpu"), True
    if want.startswith("cuda") and not torch.cuda.is_available():
        print("[train] WARNING: CUDA requested but not available; using CPU", flush=True)
        return torch.device("cpu"), False
    return torch.device(want), False


def amp_settings(tcfg: dict, device: torch.device):
    prec = str(tcfg.get("precision", "bf16"))
    if device.type != "cuda" or prec == "fp32":
        return None, False
    if prec == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("precision: bf16 requested but this GPU does not support bfloat16")
        return torch.bfloat16, False
    if prec == "fp16":
        # T5 is known to overflow in fp16; allowed but warned.
        print("[train] WARNING: fp16 with T5 blocks can overflow to NaN; prefer bf16", flush=True)
        return torch.float16, True
    raise ValueError(f"training.precision must be fp32|bf16|fp16, got {prec!r}")


def make_scheduler(opt, tcfg: dict, steps_per_epoch: int):
    kind = tcfg.get("scheduler", "cosine")
    total = max(1, steps_per_epoch * int(tcfg.get("epochs", 50)))
    warm = int(float(tcfg.get("warmup_frac", 0.05)) * total)
    min_ratio = float(tcfg.get("min_lr_ratio", 0.05))
    if kind == "cosine":
        def f(step):
            if step < warm:
                return (step + 1) / max(1, warm)
            prog = (step - warm) / max(1, total - warm)
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
        return torch.optim.lr_scheduler.LambdaLR(opt, f), "step"
    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=float(tcfg.get("plateau_factor", 0.5)),
            patience=int(tcfg.get("plateau_patience", 3))), "epoch"
    raise ValueError(f"training.scheduler must be cosine|plateau, got {kind!r}")


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def compute_losses(model, fe, crit, batch, lcfg, noise, amp_dtype, device, conf_fill):
    core = model.module if hasattr(model, "module") else model
    with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
        pred, counter = model(batch["input_ids"], batch["attention_mask"], batch["pose"], batch["frame_mask"],
                              input_noise_std=noise)
    counter_gt = core.counter_target(batch["frame_mask"])
    feat_p = feat_g = None
    if fe is not None and float(lcfg.get("lambda_feat", 0)) > 0:
        lengths = batch["lengths"]
        # Uni-Sign semantics: both sequences padded with their last real frame,
        # the same joints valid, the same camera and confidence channel.
        pp = pad_with_last_frame(pred.float(), lengths)
        gp = pad_with_last_frame(batch["pose"], lengths)
        jv = pad_with_last_frame(batch["joint_valid"], lengths)
        conf = conf_fill.to(device)
        with torch.autocast(device_type=device.type, enabled=False):
            feat_p = fe(to_unisign_parts(pp, jv, batch["camera"], conf))
            with torch.no_grad():
                feat_g = fe(to_unisign_parts(gp, jv, batch["camera"], conf))
    return crit(pred, batch["pose"], batch["joint_valid"], batch["frame_mask"], counter, counter_gt, feat_p, feat_g)


@torch.no_grad()
def validate(model, fe, crit, loader, lcfg, vcfg, amp_dtype, device, conf_fill, gen_batches):
    model.eval()
    core = model.module if hasattr(model, "module") else model
    sums, n = {}, 0
    gen_rows = []
    for bi, batch in enumerate(loader):
        if batch is None:
            continue
        batch = to_device(batch, device)
        # Unwrapped model: a DDP forward on rank 0 alone would start a buffer
        # broadcast the other ranks never join.
        losses = compute_losses(core, fe, crit, batch, lcfg, 0.0, amp_dtype, device, conf_fill)
        for k, v in losses.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        n += 1
        if bi < gen_batches:
            with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                gens = core.generate(batch["input_ids"], batch["attention_mask"],
                                     max_frames=int(vcfg.get("max_frames", 300)),
                                     min_frames=int(vcfg.get("min_frames", 8)),
                                     stop_counter=float(vcfg.get("stop_counter", 0.97)))
            for i, g in enumerate(gens):
                L = int(batch["lengths"][i])
                gen_rows.append(sequence_metrics(g.numpy(), batch["pose"][i, :L].cpu().numpy(),
                                                 batch["joint_valid"][i, :L].cpu().numpy()))
    out = {f"val_{k}": v / max(1, n) for k, v in sums.items()}
    if gen_rows:
        out.update({f"val_gen_{k}": v for k, v in aggregate(gen_rows).items()})
    model.train()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. training.mode=A training.epochs=5")
    ap.add_argument("--resume", action="store_true", help="continue from <output_dir>/latest.pt")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    tcfg, lcfg, vcfg, dcfg = cfg["training"], cfg.get("loss", {}), cfg.get("validation", {}), cfg["data"]
    device, ddp = setup_device(tcfg)
    rank = dist.get_rank() if ddp else 0
    set_seed(int(cfg.get("seed", 42)) + rank, bool(tcfg.get("deterministic", False)))
    out_dir = resolve_path(cfg, tcfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    split_dir = resolve_path(cfg, dcfg["split_dir"])
    pose_dir = resolve_path(cfg, dcfg["pose_dir"])
    train_rows, val_rows = read_split(split_dir / "train.csv"), read_split(split_dir / "val.csv")
    stats = load_or_compute_stats(out_dir / "norm_stats.json", train_rows, pose_dir) if is_main() else None
    if ddp:
        dist.barrier()
        stats = json.loads((out_dir / "norm_stats.json").read_text())

    model, fe, report, mode = build_model(cfg, stats, load_weights=True, verbose=is_main())
    model.to(device)
    if fe is not None:
        fe.to(device).eval()
    if tcfg.get("gradient_checkpointing", False):
        for stack in (model.text_encoder.stack, model.pose_decoder.stack):
            stack.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    pp = pretrained_paths(cfg)
    tok = build_tokenizer(pp["mt5_dir"])
    collate = Collator(tok, cfg.get("model", {}).get("text_prompt", ""), int(dcfg.get("max_text_tokens", 64)))
    max_frames = int(cfg.get("normalization", {}).get("max_frames", 400))
    train_ds = AuslanPoseDataset(train_rows, pose_dir, stats, dcfg.get("text_column", "text"),
                                 cfg.get("augmentation", {}), train=True, max_frames=max_frames)
    val_ds = AuslanPoseDataset(val_rows, pose_dir, stats, dcfg.get("text_column", "text"), train=False,
                               max_frames=max_frames)
    sampler = DistributedSampler(train_ds, shuffle=True, seed=int(cfg.get("seed", 42))) if ddp else None
    nw = int(tcfg.get("num_workers", 4))
    g = torch.Generator(); g.manual_seed(int(cfg.get("seed", 42)))
    train_loader = DataLoader(train_ds, batch_size=int(tcfg.get("batch_size", 16)), shuffle=sampler is None,
                              sampler=sampler, num_workers=nw, collate_fn=collate, drop_last=len(train_ds) > int(tcfg.get("batch_size", 16)),
                              pin_memory=device.type == "cuda", generator=g, persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=int(vcfg.get("batch_size", tcfg.get("batch_size", 16))),
                            shuffle=False, num_workers=nw, collate_fn=collate)
    log(f"[data] train {len(train_ds)} clips, val {len(val_ds)} clips, device {device}")

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=False)
    core = model.module if ddp else model
    groups = optimizer_param_groups(core, tcfg)
    for gr in groups:
        log(f"[optim] {gr['name']:<22} {sum(p.numel() for p in gr['params']):>12,} params  lr={gr['lr']:.2e}  wd={gr['weight_decay']}")
    opt = torch.optim.AdamW([{k: v for k, v in gr.items() if k != "name"} for gr in groups],
                            betas=tuple(tcfg.get("betas", (0.9, 0.999))), eps=float(tcfg.get("eps", 1e-8)))
    accum = int(tcfg.get("grad_accum_steps", 1))
    steps_per_epoch = max(1, len(train_loader) // accum)
    sched, sched_unit = make_scheduler(opt, tcfg, steps_per_epoch)
    amp_dtype, use_scaler = amp_settings(tcfg, device)
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None
    crit = PoseLoss(lcfg).to(device)
    conf_fill = torch.tensor(stats["mean_confidence"], dtype=torch.float32)

    tb = None
    if tcfg.get("tensorboard", False) and is_main():
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(str(out_dir / "tb"))
        except ImportError:
            log("[train] tensorboard not installed; logging to metrics.jsonl only")

    start_epoch, step, bad, best = 0, 0, 0, float("inf")
    metric_name = vcfg.get("early_stopping_metric", "val_gen_dtw_distance")
    if args.resume and (out_dir / "latest.pt").is_file():
        ck = load_checkpoint(out_dir / "latest.pt")
        core.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if ck.get("scheduler"):
            sched.load_state_dict(ck["scheduler"])
        if scaler is not None and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start_epoch, step, best, bad = ck["epoch"] + 1, ck["step"], ck["best_metric"], ck.get("bad_epochs", 0)
        log(f"[resume] from epoch {ck['epoch']} (best {metric_name}={best:.4f})")

    load_rep = report.__dict__ if report is not None else {}
    saved_cfg = clean_for_save(cfg)
    epochs = int(tcfg.get("epochs", 50))
    patience = int(vcfg.get("patience", 8))
    clip = float(tcfg.get("grad_clip", 1.0))
    noise = float(tcfg.get("decoder_input_noise", 0.05))
    log_every = int(tcfg.get("log_every", 50))
    metrics_path = out_dir / "metrics.jsonl"
    model.train()
    if fe is not None:
        fe.eval()

    for epoch in range(start_epoch, epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        t0, run, nb = time.time(), {}, 0
        opt.zero_grad(set_to_none=True)
        for bi, batch in enumerate(train_loader):
            if batch is None:
                continue
            batch = to_device(batch, device)
            losses = compute_losses(model, fe, crit, batch, lcfg, noise, amp_dtype, device, conf_fill)
            loss = losses["total"] / accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} batch {bi}: "
                                         f"{ {k: float(v) for k, v in losses.items()} }")
            (scaler.scale(loss) if scaler else loss).backward()
            for k, v in losses.items():
                run[k] = run.get(k, 0.0) + float(v.detach())
            nb += 1
            if (bi + 1) % accum:
                continue
            if scaler:
                scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_([p for p in core.parameters() if p.requires_grad], clip)
            if scaler:
                scaler.step(opt); scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            if sched_unit == "step":
                sched.step()
            step += 1
            if step % log_every == 0:
                lr = {gr.get("name", i): opt.param_groups[i]["lr"] for i, gr in enumerate(groups)}
                msg = " ".join(f"{k}={run[k]/nb:.4f}" for k in ("total", "pose", "velocity", "acceleration", "bone", "counter", "feat") if k in run)
                log(f"[ep {epoch} step {step}] {msg} gnorm={float(gnorm):.2f} lr={ {k: f'{v:.2e}' for k, v in lr.items()} }")
                if tb:
                    for k in run:
                        tb.add_scalar(f"train/{k}", run[k] / nb, step)
                    for k, v in lr.items():
                        tb.add_scalar(f"lr/{k}", v, step)

        train_summary = {f"train_{k}": v / max(1, nb) for k, v in run.items()}
        gen_batches = int(vcfg.get("generate_batches", 4))
        val = validate(model, fe, crit, val_loader, lcfg, vcfg, amp_dtype, device, conf_fill, gen_batches) if is_main() else {}
        if ddp:
            obj = [val]; dist.broadcast_object_list(obj, src=0); val = obj[0]
        metric = val.get(metric_name)
        if metric is None:
            raise KeyError(f"early_stopping_metric {metric_name!r} not produced; available: {sorted(val)}")
        if sched_unit == "epoch":
            sched.step(metric)
        improved = metric < best - float(vcfg.get("min_delta", 0.0))
        if improved:
            best, bad = metric, 0
        else:
            bad += 1
        record = {"epoch": epoch, "step": step, "time_s": round(time.time() - t0, 1),
                  "lr": [gr["lr"] for gr in opt.param_groups], **train_summary, **val,
                  "best": best, "improved": improved}
        log(f"[epoch {epoch}] " + " ".join(f"{k}={v:.4f}" for k, v in record.items()
                                            if isinstance(v, float) and ("val" in k or k.startswith("train_total") or k.startswith("train_pose") or k.startswith("train_vel"))))
        if is_main():
            with metrics_path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
            if tb:
                for k, v in record.items():
                    if isinstance(v, float) and k.startswith("val"):
                        tb.add_scalar(f"val/{k[4:]}", v, epoch)
            common = dict(model=core, optimizer=opt, scheduler=sched, scaler=scaler, epoch=epoch, step=step,
                          val_metric=metric, best_metric=best, config=saved_cfg, norm_stats=stats,
                          load_report=load_rep, bad_epochs=bad)
            save_checkpoint(out_dir / "latest.pt", **common)
            if improved:
                save_checkpoint(out_dir / "best.pt", **common)
                log(f"[epoch {epoch}] new best {metric_name}={best:.4f} -> {out_dir / 'best.pt'}")
        if bad >= patience:
            log(f"[early stop] no improvement in {metric_name} for {patience} epochs")
            break
    if tb:
        tb.close()
    log(f"[done] best {metric_name}={best:.4f}; checkpoints in {out_dir}")
    if ddp:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
