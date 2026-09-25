#!/usr/bin/env python3
"""One-off: shrink what the server copies from Drive at every start, without changing any output.

    python scripts/slim_assets.py --signspark /content/SignSparK \
        --weights  <Drive>/auslan_work/signspark_ft_smooth/final \
        --bank     <Drive>/auslan_work/smplx_full/lmdb_smooth/train/AuslanDaily_train.lmdb \
        --out      <Drive>/auslan_work/signchat_assets

1. SignSparK weights (3.1 GB per stream). A key is dropped only when its tensor is bit-identical
   to what building the model from scratch already produces (pretrained parts loaded from Hugging
   Face at build time), so loading the slim file into a freshly built model gives exactly the same
   model. That is checked for every stream: original-loaded vs slim-loaded state dicts, every
   tensor, torch.equal. Kept tensors are cloned before saving, so a small view of a large storage
   no longer drags the whole storage into the file.
   -> <out>/signspark/{hand,body,face}.pt + {stream}.dropped.json (the keys the loader may skip)

2. Retrieval bank (1.2 GB LMDB). Generation reads a retrieved clip only at its keyframes, so only
   those rows are kept, plus each sentence's M-CLIP embedding (computed exactly as the server
   would at startup), which also saves encoding 10k sentences at every start.
   -> <out>/bank_compact.npz; checked by rebuilding every keyframe batch from both banks.

3. <out>/report.json: sizes by key prefix, what was dropped, and the checks.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from signchat.text2sign import STREAMS, compact_bank, load_compact_bank, expand_entry  # noqa: E402


def gb(n):
    return round(n / 2**30, 3)


def prefix_sizes(sd, depth=2):
    out = collections.Counter()
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[".".join(k.split(".")[:depth])] += v.numel() * v.element_size()
    return {k: gb(v) for k, v in out.most_common(15)}


def fresh_model(ft, cfg, seed=0):
    from basic_utils import args_to_dict, create_model_and_flow
    torch.manual_seed(seed)
    model, _ = create_model_and_flow(**args_to_dict(cfg, cfg.keys()))
    return model


def slim_weights(ft, ssk, src_dir, out_dir, report):
    os.makedirs(out_dir, exist_ok=True)
    for s in STREAMS:
        t0 = time.time()
        src = os.path.join(src_dir, f"{s}.pt")
        sd = torch.load(src, map_location="cpu", weights_only=False)
        cfg = ft.load_cfg(ssk, s)
        # only what a build reproduces whatever the random seed (pretrained parts loaded from disk / HF),
        # never a randomly initialised tensor that happens to match one seed
        base, other = fresh_model(ft, cfg, 0).state_dict(), fresh_model(ft, cfg, 1).state_dict()
        dropped = sorted(k for k, v in sd.items() if torch.is_tensor(v) and k in base
                         and v.shape == base[k].shape and v.dtype == base[k].dtype
                         and torch.equal(v, base[k]) and torch.equal(base[k], other[k]))
        del base, other
        slim = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in sd.items() if k not in set(dropped)}
        dst = os.path.join(out_dir, f"{s}.pt")
        torch.save(slim, dst + ".part")
        os.replace(dst + ".part", dst)
        with open(os.path.join(out_dir, f"{s}.dropped.json"), "w") as fh:
            json.dump(dropped, fh)

        # the check: a fresh model + original file == a fresh model + slim file, every tensor
        a = fresh_model(ft, cfg, 2)
        ft.load_weights(a, src)
        b = fresh_model(ft, cfg, 3)             # another seed: a dropped random tensor would show up here
        missing, unexpected = b.load_state_dict(torch.load(dst, map_location="cpu", weights_only=False), strict=False)
        assert not unexpected and set(missing) <= set(dropped) | {k for k in missing if k.startswith(ft.TEXT_KEY)}, s
        sa, sb = a.state_dict(), b.state_dict()
        diff = [k for k in sa if not torch.equal(sa[k], sb[k])]
        assert not diff, f"{s}: slim weights load differently: {diff[:5]}"

        report["weights"][s] = {
            "file_gb": gb(os.path.getsize(src)), "slim_file_gb": gb(os.path.getsize(dst)),
            "tensor_gb": gb(sum(v.numel() * v.element_size() for v in sd.values() if torch.is_tensor(v))),
            "keys": len(sd), "dropped_keys": len(dropped), "dropped_by_prefix": prefix_sizes({k: sd[k] for k in dropped}),
            "kept_by_prefix": prefix_sizes(slim), "identical_after_load": True, "seconds": round(time.time() - t0, 1)}
        print(f"[weights] {s}: {report['weights'][s]['file_gb']} GB -> {report['weights'][s]['slim_file_gb']} GB, "
              f"{len(dropped)} of {len(sd)} keys dropped, loads identically", flush=True)
        print("          kept:", report["weights"][s]["kept_by_prefix"], flush=True)


def slim_bank(ft, R, ssk, weights_dir, bank_path, out_path, device, report):
    t0 = time.time()
    bank = R.load_bank(bank_path)
    texts = [e["text"] for e in bank]
    model, _ = ft.build(ft.load_cfg(ssk, "hand"), os.path.join(weights_dir, "hand.pt"), device,
                        texts=sorted(set(texts)))                  # = the server's startup encoding
    emb = torch.nn.functional.normalize(model.encode_text(texts).float(), dim=-1).cpu().numpy()
    arrays = compact_bank(bank)
    arrays["emb"] = emb.astype(np.float32)
    np.savez(out_path + ".part.npz", **arrays)
    os.replace(out_path + ".part.npz", out_path)

    # the check: every clip's keyframe batch is the same from the compact bank as from the LMDB
    compact = load_compact_bank(out_path)
    rng = np.random.default_rng(0)
    for i in range(len(bank)):
        full, small = bank[i], expand_entry(compact[i])
        assert full["text"] == small["text"] and full["T"] == small["T"] and full["keyframes"] == small["keyframes"], i
        n = int(rng.integers(20, 300))
        for s in STREAMS:
            xa, ya = R.keyframe_batch(s, [full["text"]], [n], [full])
            xb, yb = R.keyframe_batch(s, [small["text"]], [n], [small])
            assert torch.equal(xa, xb) and ya["keyframes"] == yb["keyframes"], (i, s)
    report["bank"] = {"lmdb_gb": gb(sum(os.path.getsize(os.path.join(bank_path, f)) for f in os.listdir(bank_path))),
                      "compact_gb": gb(os.path.getsize(out_path)), "clips": len(bank),
                      "keyframe_rows": int(len(arrays["frames"])), "emb_dim": int(emb.shape[1]),
                      "identical_keyframe_batches": True, "seconds": round(time.time() - t0, 1)}
    print(f"[bank] {report['bank']['lmdb_gb']} GB -> {report['bank']['compact_gb']} GB, "
          f"{len(bank)} clips, every keyframe batch identical", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signspark", required=True)
    ap.add_argument("--code", default=os.path.join(HERE, "..", "..", "auslan_smplx"))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-weights", action="store_true")
    ap.add_argument("--skip-bank", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")
    sys.path.insert(0, os.path.abspath(args.code))
    import signspark_ft as ft
    ft.setup_paths(args.signspark)
    import signspark_render as R
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    report = {"weights": {}, "source": {"weights": args.weights, "bank": args.bank}}
    if not args.skip_weights:
        slim_weights(ft, args.signspark, args.weights, os.path.join(args.out, "signspark"), report)
    if not args.skip_bank:
        slim_bank(ft, R, args.signspark, args.weights, args.bank, os.path.join(args.out, "bank_compact.npz"), device, report)
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print("report:", os.path.join(args.out, "report.json"))


if __name__ == "__main__":
    main()
