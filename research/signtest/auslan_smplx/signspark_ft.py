"""Fine-tune one SignSparK stream (hand / body / face) on the Auslan-Daily LMDBs.

Uses SignSparK's own model, flow loss and dataset code (the repo is put on sys.path); what this
file replaces is the training loop, because the released one cannot fine-tune as it stands:

- resuming from the released ema_0.9999_200000.pt sets the step counter to 200000, so its linear
  LR anneal starts almost at zero, and its EMA at rate 0.9999 barely moves in a few thousand steps;
- it evaluates only the flow loss on one random batch, which says little about text-to-pose.

Here the released weights are the starting point and step 0, the LR warms up and decays by
cosine, several EMA rates are tracked in one run, and every eval measures the same fixed dev
clips three ways: the flow loss at fixed t and noise, keyframe-conditioned sampling (the paper's
setting) and text-only sampling with classifier-free guidance (our use case), scored by DTW and by
how much it moves.

Speed and memory, all numerically neutral except bf16, which section 5 of the notebook checks on
the eval metrics: every sentence is encoded once up front and the frozen text encoder (2.2 GB)
then leaves the GPU, so its memory goes to the batch instead; keyframe
masks are built on the CPU in one go instead of hundreds of tiny GPU writes, the loss's per-step
min/max logging (a GPU->CPU sync every step) is switched off, AdamW is fused, the EMA is one
foreach lerp over the trainable weights only (not the frozen 560M-parameter text encoder), and
losses are read back only every log interval.

    python signspark_ft.py probe --stream hand --amp 0 --check 64 --batches 32,48,64,96
    python signspark_ft.py train --run run.json
    python signspark_ft.py eval --stream hand --weights released|path.pt --split test --n 512
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
import types

import numpy as np
import torch

EPS = 1e-3
TEXT_KEY = 'text_enc_model.'          # the frozen M-CLIP encoder inside the SignSparK model


# --------------------------------------------------------------------------- setup
def setup_paths(ssk: str) -> None:
    """Put SignSparK on sys.path. Its modules import wandb, which we never use: the real package is
    used when installed (the notebook installs it, runs set WANDB_MODE=disabled), otherwise a stub.
    The stub needs a __spec__, because transformers probes packages with importlib.util.find_spec,
    which raises on a module whose __spec__ is None."""
    if ssk not in sys.path:
        sys.path.insert(0, ssk)
    try:
        import wandb  # noqa: F401
    except ImportError:
        import importlib.machinery
        stub = types.ModuleType('wandb')
        stub.__spec__ = importlib.machinery.ModuleSpec('wandb', None)
        sys.modules['wandb'] = stub
    os.environ.setdefault('TQDM_DISABLE', '1')          # the ODE sampler opens a progress bar per call
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def load_cfg(ssk: str, stream: str, overrides: list[str] | None = None):
    """The repo's own training config for this stream (configs/default.yaml + data_v2/<stream>)."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=os.path.join(ssk, 'configs'), version_base=None):
        return compose(config_name='default', overrides=[f'data_v2={stream}'] + list(overrides or []))


def trainable_state(model) -> dict:
    return {k: v for k, v in model.state_dict().items() if not k.startswith(TEXT_KEY)}


def load_weights(model, path: str) -> None:
    """Load either a released checkpoint (full state, text encoder included) or one of ours
    (trainable weights only). Anything missing must be the text encoder, which is loaded from
    Hugging Face when the model is built and never trained."""
    sd = torch.load(path, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith(TEXT_KEY)]
    assert not bad and not unexpected, f'{path}: missing {bad[:5]}, unexpected {unexpected[:5]}'


def build(cfg, weights: str, device, texts=()):
    """The model with the given weights. With `texts`, every sentence is encoded once here and the
    text encoder is moved back to the CPU: training then never runs it, and its 2.2 GB of weights
    (plus activations) are free for the batch. A sentence missing from `texts` still works, it is
    just encoded on the CPU when first seen."""
    from basic_utils import args_to_dict, create_model_and_flow
    model, flow = create_model_and_flow(**args_to_dict(cfg, cfg.keys()))
    load_weights(model, weights)
    model.to(device)
    flow.log_min_max = lambda a, b: {}                  # was a .cpu() of two full batches every step
    model.encode_text = TextCache(model)
    if texts:
        t0 = time.time()
        model.encode_text.prime(sorted(set(texts)))
        model.text_enc_model.to('cpu')
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f'text encoder: {len(model.encode_text.cache)} sentences encoded in {time.time() - t0:.0f}s, '
              f'encoder moved off the GPU', flush=True)
    return model, flow


def lmdb_texts(path: str) -> list[str]:
    """Every clip's conditioning text, exactly as PoseDataset_Lmdb builds it (specify_lang=True)."""
    import io
    import pickle
    import lmdb
    try:
        env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
    except lmdb.Error as e:
        # py-lmdb allows one open handle per file per process; a loader running in this process may
        # hold it. Then nothing is pre-encoded: sentences are encoded as first seen, same numbers.
        print(f'WARNING: {path} is already open in this process ({e}); text encoder stays on the GPU', flush=True)
        return []
    out = []
    with env.begin() as txn:
        for k in pickle.loads(txn.get(b'__meta__'))['clip_ids']:
            with np.load(io.BytesIO(txn.get(k.encode())), allow_pickle=True) as z:
                out.append(f"<{z['language'][0]}> {z['translation'][0]}")
    env.close()
    return out


class TextCache:
    """Frozen text encoder, memoised per sentence. Exact: the encoder is in eval mode and under
    no_grad, and each sentence is pooled over its own attention mask. Always run in fp32."""

    def __init__(self, model):
        self.model = model
        self.encode = model.encode_text
        self.cache = {}

    def __call__(self, texts):
        new = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if new:
            enc = next(iter(self.model.text_enc_model.parameters())).device
            with torch.autocast(enc.type, enabled=False):
                emb = self.encode(new)
            # stored where the rest of the model is, wherever the encoder happens to be
            self.cache.update(zip(new, emb.float().to(self.model.embed_text.weight.device)))
        return torch.stack([self.cache[t] for t in texts])

    def prime(self, texts, batch=256):
        texts = list(texts)
        for i in range(0, len(texts), batch):
            self(texts[i:i + batch])


def set_mode(model, train: bool) -> None:
    """train() on the whole model would also switch the frozen text encoder to train mode (dropout
    on, cache no longer exact); SignSparK's own loop never calls train(), so it stays in eval."""
    model.train(train)
    model.text_enc_model.eval()


class Amp(torch.nn.Module):
    """Runs the model under bf16 autocast (or not) and hands the flow loss fp32 outputs."""

    def __init__(self, model, amp: bool):
        super().__init__()
        self.model, self.amp = model, amp

    def forward(self, x, t, **kw):
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=self.amp):
            return self.model(x, t, **kw).float()


# --------------------------------------------------------------------------- data
def data_args(stream: str):
    return types.SimpleNamespace(dataset_feat=stream, specify_lang=True, flip_left_hand=True,
                                 keyframe_selection_mode=3, custom_keyframe_file='null')


def lmdb_path(data: str, split: str, name: str = 'AuslanDaily') -> str:
    return os.path.join(data, split, f'{name}_{split}.lmdb')


def _seed_worker(worker_id):
    # PoseDataset_Lmdb picks the hand side with np.random; numpy is not reseeded per worker
    np.random.seed((torch.initial_seed() + worker_id) % 2 ** 32)


def loader(stream, path, batch, shuffle, drop_last, workers=8, concat_hands=False):
    from torch.utils.data import DataLoader
    from signspark.pose_datasets_lmdb import PoseDataset_Lmdb, custom_collate_fn
    ds = PoseDataset_Lmdb(data_args(stream), split=os.path.basename(os.path.dirname(path)), seq_len=304,
                          data=path, concat_hands=concat_hands)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=drop_last, collate_fn=custom_collate_fn,
                      num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
                      prefetch_factor=4 if workers > 0 else None, worker_init_fn=_seed_worker)


def kf_mask(keyframes, B, J, T) -> torch.Tensor:
    m = torch.zeros(B, 1, T, dtype=torch.bool)
    for i, k in enumerate(keyframes):
        k = [f for f in k if f < T]
        if k:
            m[i, 0, k] = True
    return m.expand(B, J, T)


def split_hands(x, y):
    """(B, 180, T) with concat_hands -> (2B, 90, T), L0 R0 L1 R1 ..., as sample.py does."""
    B, _, T = x.shape
    x = torch.stack([x[:, :90], x[:, 90:]], 1).reshape(2 * B, 90, T)
    rep = lambda v: [a for a in v for _ in range(2)]
    y = dict(y, mask=y['mask'].repeat_interleave(2, 0), lengths=y['lengths'].repeat_interleave(2, 0),
             text=rep(y['text']), keyframes=rep(y['keyframes']), video_names=rep(y['video_names']))
    return x, y


def eval_batches(stream, path, n, batch=128, workers=2):
    """The first n clips of a split in loader order (deterministic), both hands for the hand stream."""
    out, seen = [], 0
    for x, y in loader(stream, path, batch, shuffle=False, drop_last=False, workers=workers, concat_hands=stream == 'hand'):
        if seen >= n:
            break
        k = min(len(x), n - seen)
        x, y = x[:k], {kk: v[:k] for kk, v in y.items()}
        if stream == 'hand':
            x, y = split_hands(x, y)
        out.append((x, y))
        seen += k
    return out


# --------------------------------------------------------------------------- metrics
def rot6d_to_mat(d6):
    """SignSparK's row-based 6D (Zhou et al.): rows b1, b2, b3."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], -2)


def split_feats(stream, x):
    """(B, C, T) -> rotations (B, T, J, 9) and extra non-rotation channels (B, T, E) or None."""
    x = x.transpose(1, 2).float()
    rot, extra = (x[..., :6], x[..., 6:]) if stream == 'face' else (x, None)
    R = rot6d_to_mat(rot.reshape(*rot.shape[:2], -1, 6))
    return R.reshape(*R.shape[:3], 9), extra


def angle_deg(tr):
    return torch.rad2deg(torch.acos(((tr - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)))


def frame_error(stream, pred, gt):
    """Per-frame error (B, T): mean joint rotation angle in degrees; for the face, the mean
    absolute difference of the 50 expression coefficients (the jaw moves far less and would only
    dilute it)."""
    Rp, ep = split_feats(stream, pred)
    Rg, eg = split_feats(stream, gt)
    if stream == 'face':
        return (ep - eg).abs().mean(-1)
    return angle_deg((Rp * Rg).sum(-1)).mean(-1)


def cost_matrix(stream, pred, gt):
    """(B, T, T) frame-to-frame cost, same units as frame_error."""
    Rp, ep = split_feats(stream, pred)
    Rg, eg = split_feats(stream, gt)
    if stream == 'face':
        return torch.cdist(ep, eg, p=1) / ep.shape[-1]
    C = 0
    for j in range(Rp.shape[2]):
        C = C + angle_deg(torch.bmm(Rp[:, :, j], Rg[:, :, j].transpose(1, 2)))
    return C / Rp.shape[2]


def dtw(C, L):
    """Batched DTW over anti-diagonals. C (B, T, T), L (B,) lengths; returns cost / L, i.e. the
    mean per-step cost of the best alignment of the first L frames of each pair."""
    B, T, _ = C.shape
    D = torch.full((B, T + 1, T + 1), float('inf'), device=C.device)
    D[:, 0, 0] = 0
    for k in range(2, 2 * T + 1):
        i = torch.arange(max(1, k - T), min(T, k - 1) + 1, device=C.device)
        j = k - i
        D[:, i, j] = C[:, i - 1, j - 1] + torch.minimum(torch.minimum(D[:, i - 1, j], D[:, i - 1, j - 1]), D[:, i, j - 1])
    L = L.to(C.device).long()
    return D[torch.arange(B, device=C.device), L, L] / L


def motion(x, valid):
    """Sum of |frame-to-frame change| over valid transitions, and their count."""
    d = (x[..., 1:] - x[..., :-1]).abs().mean(1)
    v = (valid[..., 1:] & valid[..., :-1]).float()
    return float((d * v).sum()), float(v.sum())


@torch.no_grad()
def evaluate(model, flow, batches, stream, amp, device, kf_steps=10, txt_steps=20, text_scale=2.5, seed=0):
    set_mode(model, False)
    fwd = Amp(model, amp)

    def guided(x, t, y=None, obs_x0=None, obs_mask=None):
        out = fwd(x, t, y=y, obs_x0=obs_x0, obs_mask=obs_mask)
        out_u = fwd(x, t, y=dict(y, uncond=True), obs_x0=obs_x0, obs_mask=obs_mask)
        return out_u + text_scale * (out - out_u)

    acc = {k: [] for k in ('val_loss', 'kf_err', 'txt_err', 'txt_dtw')}
    mot = np.zeros(4)                           # pred sum, pred n, gt sum, gt n
    for bi, (x, y) in enumerate(batches):
        x = x.to(device)
        B, J, T = x.shape
        y = dict(y, mask=y['mask'].to(device))
        valid = y['mask'][:, 0]
        obs = kf_mask(y['keyframes'], B, J, T).to(device).contiguous()
        cond = {'y': y, 'obs_x0': x, 'obs_mask': obs}
        with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
            torch.manual_seed(seed * 1000 + bi)
            t = torch.rand(B, device=device) * (1 - EPS) + EPS
            acc['val_loss'].append(flow.training_losses(fwd, x, t, model_kwargs=cond)['loss'].float().cpu())
            noise = torch.randn_like(x)
            kf, _ = flow.decode(fwd, noise=noise, keyframe_mask=obs, x_embed=x, model_kwargs=cond,
                                ode_package='torchdiffeq', ode_stepnum=kf_steps)
            none = torch.zeros_like(obs)
            tx, _ = flow.decode(guided, noise=noise, keyframe_mask=none, x_embed=x,
                                model_kwargs=dict(cond, obs_mask=none), ode_package='torchdiffeq', ode_stepnum=txt_steps)
        between = valid & ~obs[:, 0]            # keyframes are copied in, so score only the frames between
        e = frame_error(stream, kf, x)
        acc['kf_err'].append(((e * between).sum(1) / between.sum(1).clamp(min=1)).cpu())
        e = frame_error(stream, tx, x)
        acc['txt_err'].append(((e * valid).sum(1) / valid.sum(1)).cpu())
        for s in range(0, B, 32):
            acc['txt_dtw'].append(dtw(cost_matrix(stream, tx[s:s + 32], x[s:s + 32]), y['lengths'][s:s + 32]).cpu())
        mot[:2] += motion(tx, valid)
        mot[2:] += motion(x, valid)
    out = {k: float(torch.cat(v).mean()) for k, v in acc.items()}
    out['txt_motion'] = float((mot[0] / mot[1]) / (mot[2] / mot[3]))
    return out


# --------------------------------------------------------------------------- training
def lr_at(step, run):
    w = run.get('warmup', 200)
    if step < w:
        return run['lr'] * (step + 1) / w
    frac = (step - w) / max(1, run['steps'] - w)
    floor = run.get('lr_floor', 0.1)
    return run['lr'] * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * frac)))


def score(m, lo=0.3, hi=1.6):
    """Model selection: text-only DTW (our use case); a clip set that barely moves or thrashes is
    disqualified whatever its DTW, since a frozen pose can have a deceptively low distance.
    lo is 0.3, not nearer 1: the released model already scores 0.42 on Auslan dev (our fitted clips
    carry frame-to-frame jitter that inflates the reference), while a collapsed model sits near 0.01."""
    return m['txt_dtw'] if lo <= m['txt_motion'] <= hi else float('inf')


def train(run: dict) -> dict:
    """run: name, stream, lr, steps, batch, kf_drop, ema_rates, eval_every, eval_n, amp, seed,
    ssk, ckpt (released weights), data (lmdb root), out (Drive dir for metrics), local (weights)."""
    setup_paths(run['ssk'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = run.get('tf32', True)
    torch.manual_seed(run.get('seed', 0))
    np.random.seed(run.get('seed', 0))
    stream = run['stream']
    out_dir = os.path.join(run['out'], run['name'])
    w_dir = os.path.join(run['local'], run['name'])
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(w_dir, exist_ok=True)
    json.dump(run, open(os.path.join(out_dir, 'run.json'), 'w'), indent=1)

    cfg = load_cfg(run['ssk'], stream, [f"train_keyframe_mask_prob={run['kf_drop']}"] + run.get('cfg_overrides', []))
    texts = lmdb_texts(lmdb_path(run['data'], 'train')) + lmdb_texts(lmdb_path(run['data'], 'dev'))
    model, flow = build(cfg, run['ckpt'], device, texts=texts)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[{run['name']}] trainable {sum(p.numel() for p in params) / 1e6:.0f}M | {device} | "
          f"amp {run['amp']} | batch {run['batch']} | lr {run['lr']:g} | kf_drop {run['kf_drop']}", flush=True)
    opt = torch.optim.AdamW(params, lr=run['lr'], weight_decay=cfg.weight_decay,
                            **({'fused': True} if device.type == 'cuda' else {}))
    emas = {r: [p.detach().clone() for p in params] for r in run['ema_rates']}
    fwd = Amp(model, run['amp'])

    ev = eval_batches(stream, lmdb_path(run['data'], 'dev'), run['eval_n'], workers=min(2, run.get('workers', 8)))
    rows, best = [], {'score': float('inf')}
    metrics_csv = os.path.join(out_dir, 'metrics.csv')

    @torch.no_grad()
    def do_eval(step):
        nonlocal best
        # the live weights wait on the CPU while an EMA is evaluated, so eval needs no extra GPU memory
        backup = [p.detach().to('cpu', copy=True) for p in params] if step > 0 else None
        for r, ema in (emas.items() if step > 0 else [('released', None)]):
            if ema is not None:
                torch._foreach_copy_(params, ema)
            t0 = time.time()
            m = evaluate(model, flow, ev, stream, run['amp'], device, seed=0)
            row = {'step': step, 'ema': r, **{k: round(v, 5) for k, v in m.items()},
                   'score': round(score(m), 5), 'eval_s': round(time.time() - t0, 1)}
            rows.append(row)
            print(f"  eval step {step} ema {r}: " + ' '.join(f'{k} {v}' for k, v in row.items() if k not in ('step', 'ema')), flush=True)
            if step > 0 and row['score'] < best['score']:
                best = dict(row)
                torch.save(trainable_state(model), os.path.join(w_dir, 'best.pt'))
        if backup is not None:
            for p, b in zip(params, backup):
                p.copy_(b)
        with open(metrics_csv, 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        set_mode(model, True)

    do_eval(0)
    it = iter(loader(stream, lmdb_path(run['data'], 'train'), run['batch'], shuffle=True, drop_last=True,
                     workers=run.get('workers', 8)))
    loss_acc = torch.zeros((), device=device)
    t_log, n_log = time.time(), 0
    set_mode(model, True)
    for step in range(1, run['steps'] + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader(stream, lmdb_path(run['data'], 'train'), run['batch'], shuffle=True, drop_last=True,
                             workers=run.get('workers', 8)))
            x, y = next(it)
        B, J, T = x.shape
        obs = kf_mask(y['keyframes'], B, J, T).clone()
        obs[torch.rand(B) < run['kf_drop']] = False          # keyframe-free clips: the text-only case
        x = x.to(device, non_blocking=True)
        y['mask'] = y['mask'].to(device, non_blocking=True)
        cond = {'y': y, 'obs_x0': x, 'obs_mask': obs.to(device, non_blocking=True)}
        t = torch.rand(B, device=device) * (1 - EPS) + EPS
        for g in opt.param_groups:
            g['lr'] = lr_at(step - 1, run)
        loss = flow.training_losses(fwd, x, t, model_kwargs=cond)['loss'].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(params, cfg.gradient_clipping)
        opt.step()
        with torch.no_grad():
            for r, ema in emas.items():
                torch._foreach_lerp_(ema, params, 1 - r)
        loss_acc += loss.detach()
        n_log += 1
        if step % run.get('log_every', 100) == 0 or step == run['steps']:
            el = (time.time() - t_log) / n_log
            mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0
            print(f"[{run['name']}] step {step}/{run['steps']} | loss {float(loss_acc) / n_log:.4f} | "
                  f"lr {lr_at(step - 1, run):.2e} | {el:.3f} s/step | {B / el:.0f} clips/s | "
                  f"peak {mem:.1f} GB | eta {(run['steps'] - step) * el / 60:.0f} min", flush=True)
            loss_acc.zero_()
            t_log, n_log = time.time(), 0
        if step % run['eval_every'] == 0 or step == run['steps']:
            do_eval(step)
            t_log, n_log = time.time(), 0

    if 'step' not in best:
        print(f"[{run['name']}] WARNING: no eval point kept txt_motion in range, nothing saved", flush=True)
    done = {'name': run['name'], 'best': best, 'weights': os.path.join(w_dir, 'best.pt') if 'step' in best else None,
            'baseline': next(r for r in rows if r['ema'] == 'released')}
    json.dump(done, open(os.path.join(out_dir, 'done.json'), 'w'), indent=1)
    print(f"[{run['name']}] done. best: step {best.get('step')} ema {best.get('ema')} score {best['score']}", flush=True)
    return done


# --------------------------------------------------------------------------- speed probe
def probe(ssk, ckpt, data, stream, batches, amp, n_ema=2, steps=5, check_n=0, cfg_overrides=()):
    """Seconds per optimizer step and peak memory for one precision, batch sizes in ascending order,
    with the optimizer state and n_ema EMA copies allocated as in a real run. Stops at the first
    batch that does not fit: after an out-of-memory error the allocator is not reliably back to its
    earlier state, so every later row in the same process would be suspect (the notebook runs one
    process per precision for the same reason).

    With check_n > 0 it first evaluates the released weights on the first check_n dev clips in fp32
    and in bf16: whether bf16 is safe is decided on the eval metrics, not on raw output differences."""
    setup_paths(ssk)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    cfg = load_cfg(ssk, stream, list(cfg_overrides))
    texts = lmdb_texts(lmdb_path(data, 'train')) + (lmdb_texts(lmdb_path(data, 'dev')) if check_n else [])
    model, flow = build(cfg, ckpt, device, texts=texts)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9 if device.type == 'cuda' else 0
    out = {'amp': amp, 'total_GB': round(total, 1), 'rows': []}

    if check_n:
        ev = eval_batches(stream, lmdb_path(data, 'dev'), check_n, workers=0)
        out['check'] = {}
        for a in (False, True):
            m = evaluate(model, flow, ev, stream, a, device)
            out['check']['bf16' if a else 'fp32'] = {k: round(v, 4) for k, v in m.items()}
            print(f"released weights, {check_n} dev clips, {'bf16' if a else 'fp32'}: {out['check']['bf16' if a else 'fp32']}", flush=True)
        del ev

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-12, **({'fused': True} if device.type == 'cuda' else {}))
    emas = [[p.detach().clone() for p in params] for _ in range(n_ema)]
    fwd = Amp(model, amp)
    set_mode(model, True)
    X, Y = next(iter(loader(stream, lmdb_path(data, 'train'), max(batches), shuffle=False, drop_last=True, workers=0)))
    for bs in sorted(batches):
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            x = X[:bs].to(device)
            y = {k: (v[:bs].to(device) if torch.is_tensor(v) else v[:bs]) for k, v in Y.items()}
            cond = {'y': y, 'obs_x0': x, 'obs_mask': kf_mask(y['keyframes'], *x.shape).to(device).contiguous()}
            times = []
            for i in range(steps + 1):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.time()
                tt = torch.rand(bs, device=device) * (1 - EPS) + EPS
                loss = flow.training_losses(fwd, x, tt, model_kwargs=cond)['loss'].mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    for ema in emas:
                        torch._foreach_lerp_(ema, params, 1e-4)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                if i:
                    times.append(time.time() - t0)
            sec = float(np.median(times))
            peak = torch.cuda.max_memory_reserved() / 1e9 if device.type == 'cuda' else 0
            row = {'amp': amp, 'batch': bs, 's_per_step': round(sec, 3), 'clips_per_s': round(bs / sec),
                   'peak_GB': round(peak, 1)}
            out['rows'].append(row)
            print(row, flush=True)
            del x, y, cond, loss, tt
        except torch.OutOfMemoryError:
            out['rows'].append({'amp': amp, 'batch': bs, 's_per_step': None, 'clips_per_s': None, 'peak_GB': 'OOM'})
            print(out['rows'][-1], '- stopping here, larger batches will not fit either', flush=True)
            break
    return out


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('train')
    p.add_argument('--run', required=True, help='json file with the run dict')
    p = sub.add_parser('probe')
    for k in ('ssk', 'ckpt', 'data', 'stream', 'out'):
        p.add_argument(f'--{k}', required=k != 'out')
    p.add_argument('--batches', default='32,48,64,96,128')
    p.add_argument('--amp', type=int, choices=(0, 1), required=True)
    p.add_argument('--check', type=int, default=0, help='also compare fp32/bf16 eval metrics on this many dev clips')
    p = sub.add_parser('eval')
    for k in ('ssk', 'released', 'weights', 'data', 'stream', 'out'):
        p.add_argument(f'--{k}', required=True)
    p.add_argument('--split', default='test')
    p.add_argument('--n', type=int, default=512)
    p.add_argument('--amp', action='store_true')
    a = ap.parse_args()

    if a.cmd == 'train':
        train(json.load(open(a.run)))
    elif a.cmd == 'probe':
        out = probe(a.ssk, a.ckpt, a.data, a.stream, [int(b) for b in a.batches.split(',')], bool(a.amp),
                    check_n=a.check)
        if a.out:
            json.dump(out, open(a.out, 'w'), indent=1)
    else:
        setup_paths(a.ssk)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cfg = load_cfg(a.ssk, a.stream)
        model, flow = build(cfg, a.released, device, texts=lmdb_texts(lmdb_path(a.data, a.split)))
        if a.weights != 'released':
            load_weights(model, a.weights)
        m = evaluate(model, flow, eval_batches(a.stream, lmdb_path(a.data, a.split), a.n), a.stream, a.amp, device)
        m['score'] = score(m)
        print(json.dumps(m), flush=True)
        json.dump(m, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
