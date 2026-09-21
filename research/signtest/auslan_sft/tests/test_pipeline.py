#!/usr/bin/env python3
"""Local CPU tests. Run from auslan_sft/:

    python tests/test_pipeline.py --video path/to/a/signing_clip.mp4 [--heavy]

1. extractor parity   preprocessing/extract_pose.py == ../unisign/extract_pose.py (pose_batch=1)
2. converter parity   normalize_pose + unisign_format.to_unisign_parts == unisign/spec.load_part_kp
3. normalisation      clip-level constants; round-trip denormalisation; gap rules
4. losses             padded frames / invalid joints contribute exactly zero
5. end-to-end smoke   splits -> normalise -> verify load -> train (mode B, 2 epochs) -> evaluate
                      -> infer -> render, with a TINY mT5 written in the Uni-Sign checkpoint key
                      format plus the REAL Uni-Sign pose-encoder tensors. Scores are meaningless;
                      this checks wiring, not learning.
--heavy               also: frozen feature encoder == real Uni_Sign pose branch (loads the full
                      587M model; needs ~6 GB RAM)
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from preprocessing.extract_pose import WholebodyExtractor, read_video  # noqa: E402
from preprocessing.normalize_pose import denormalize, fill_gaps, normalize_clip  # noqa: E402
from slp_models import skeleton as sk  # noqa: E402
from slp_models.losses import PoseLoss  # noqa: E402
from slp_utils.unisign_format import camera_vector, to_unisign_parts  # noqa: E402

PY = sys.executable
UNISIGN_CKPT = REPO / "checkpoints" / "openasl_pose_only_slt.pth"
MT5_DIR = REPO / "Uni-Sign" / "pretrained_weight" / "mt5-base"
UNISIGN_REPO = REPO / "Uni-Sign"


def ok(msg):
    print(f"PASS  {msg}", flush=True)


def test_extractor_parity(video: Path):
    sys.path.append(str(REPO / "unisign"))
    import extract_pose as ref  # ../unisign/extract_pose.py
    frames, w, h, fps = read_video(video, max_frames=40)
    kp_a, sc_a = WholebodyExtractor()(frames, w, h)
    ex = ref.PoseExtractor(pose_batch=1)
    kp_b, sc_b, _ = ex.run(frames, w, h)
    assert np.array_equal(kp_a, kp_b) and np.array_equal(sc_a, sc_b), \
        f"max |dkp|={np.abs(kp_a-kp_b).max()} |dsc|={np.abs(sc_a-sc_b).max()}"
    ok(f"extractor output identical to unisign/extract_pose.py on {len(frames)} frames")
    return frames, w, h, fps, kp_a, sc_a


def test_converter_parity(kp, sc, w, h, fps):
    sys.path.append(str(REPO / "unisign"))
    import spec
    gold = spec.load_part_kp(kp, sc)
    ncfg = {"target_fps": fps, "max_gap_sec": 0.0, "out_of_frame_margin": 1e9, "min_hand_frac": 0.0,
            "min_shoulder_frac": 0.0}
    s, why = normalize_clip(kp, sc, w, h, fps, ncfg)
    assert s is not None, why
    assert s["pose"].shape[0] == kp.shape[0], "no resampling expected at native fps"
    conf = torch.from_numpy(sc[:, sk.WHOLEBODY_INDICES])[None]
    cam = torch.tensor([camera_vector(s["norm"], s["unisign_crop"])])
    pose = torch.nan_to_num(s["pose"])[None]
    parts = to_unisign_parts(pose, s["mask"][None], cam, conf)
    worst = 0.0
    for part in sk.UNISIGN_PARTS:
        g = torch.from_numpy(gold[part])
        o = parts[part][0]
        sl = sk.PART_SLICES[part]
        m = torch.from_numpy(sc[:, sk.WHOLEBODY_INDICES[sl]] > spec.CONF_THRESHOLD)
        if part != "body":
            root = 0 if part in ("left", "right") else -1
            m = m & m[:, [root]]          # gold subtracts even an unconfident root; skip those frames
        d = (g - o).abs()[m]
        assert m.any(), f"no comparable joints in {part}"
        worst = max(worst, float(d.max()))
        assert float(d.max()) < 1e-4, f"{part}: max diff {float(d.max())}"
    ok(f"to_unisign_parts == spec.load_part_kp (max abs diff {worst:.2e})")


def test_normalisation(kp, sc, w, h, fps):
    s, why = normalize_clip(kp, sc, w, h, fps, {"target_fps": 25.0})
    assert s is not None, why
    T25 = s["pose"].shape[0]
    assert abs(T25 - (kp.shape[0] - 1) * 25 / fps - 1) <= 1, T25
    px = denormalize(np.nan_to_num(s["pose"].numpy()), s["norm"])
    s_native, _ = normalize_clip(kp, sc, w, h, fps, {"target_fps": fps, "max_gap_sec": 0})
    px_native = denormalize(np.nan_to_num(s_native["pose"].numpy()), s_native["norm"])
    m = s_native["mask"].numpy()
    orig = kp[:, sk.WHOLEBODY_INDICES] * np.asarray([w, h])
    assert np.abs(px_native[m] - orig[m]).max() < 1e-2, "denormalise(normalise(x)) != x"
    # shoulders: centre of the clip-median ~ 0, width ~ 1
    sh = s_native["pose"][:, [sk.L_SHOULDER, sk.R_SHOULDER]].numpy()
    both = s_native["mask"][:, [sk.L_SHOULDER, sk.R_SHOULDER]].all(1).numpy()
    assert abs(np.median(np.linalg.norm(sh[both, 0] - sh[both, 1], axis=-1)) - 1) < 1e-4
    # gap rules
    T = 20
    pose = np.tile(np.arange(T, dtype=np.float32)[:, None, None], (1, 3, 2))
    valid = np.ones((T, 3), bool)
    valid[5:7, 0] = False      # short internal gap -> interpolated, valid
    valid[8:16, 1] = False     # long internal gap -> interpolated, invalid
    valid[:, 2] = False        # never seen -> NaN
    out, v = fill_gaps(pose, valid, max_gap=3)
    assert v[5:7, 0].all() and np.allclose(out[5:7, 0, 0], [5, 6])
    assert not v[8:16, 1].any() and np.allclose(out[8:16, 1, 0], np.arange(8, 16))
    assert np.isnan(out[:, 2]).all() and not v[:, 2].any()
    ok(f"normalisation round-trip <1e-2 px, shoulder width = 1, gap rules; {kp.shape[0]}@{fps:g} -> {T25}@25 frames")


def test_loss_masking():
    torch.manual_seed(0)
    crit = PoseLoss({"lambda_feat": 0})
    B, T, J = 2, 10, sk.NUM_JOINTS
    gt = torch.randn(B, T, J, 2)
    pred = gt + 0.1 * torch.randn_like(gt)
    fm = torch.ones(B, T, dtype=torch.bool); fm[1, 6:] = False
    jv = torch.ones(B, T, J, dtype=torch.bool); jv[0, :, 40] = False
    base = crit(pred, gt, jv, fm)
    corrupted = pred.clone()
    corrupted[1, 6:] += 1e3          # padded frames
    corrupted[0, :, 40] += 1e3       # invalid joint
    again = crit(corrupted, gt, jv, fm)
    for k in base:
        assert torch.allclose(base[k], again[k]), (k, base[k], again[k])
    ok("padded frames and invalid joints contribute exactly zero to every loss term")


def make_tiny_checkpoint(path: Path):
    from transformers import MT5Config, MT5ForConditionalGeneration
    cfg = MT5Config.from_pretrained(str(MT5_DIR))
    for k, v in dict(d_model=32, d_ff=64, num_heads=2, d_kv=16, num_layers=3, num_decoder_layers=3).items():
        setattr(cfg, k, v)
    torch.manual_seed(0)
    tiny = MT5ForConditionalGeneration(cfg)
    state = {f"mt5_model.{k}": v for k, v in tiny.state_dict().items()}
    real = torch.load(str(UNISIGN_CKPT), map_location="cpu", mmap=True, weights_only=False)["model"]
    state.update({k: v.clone() for k, v in real.items() if not k.startswith("mt5_model.")})
    torch.save({"model": state}, path)
    return {"d_model": 32, "d_ff": 64, "num_heads": 2, "d_kv": 16, "num_layers": 3, "num_decoder_layers": 3}


def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print(r.stdout[-4000:]); print(r.stderr[-4000:])
        raise AssertionError(f"command failed ({r.returncode})")
    return r.stdout


def test_end_to_end(kp, sc, w, h, fps, workdir: Path):
    raw = workdir / "raw_pose"; raw.mkdir(parents=True, exist_ok=True)
    texts = ["how are you", "i am going to school", "thank you very much", "good morning",
             "see you tomorrow", "what is your name", "nice to meet you", "i like coffee",
             "where is the station", "it is raining today", "happy birthday", "see you later"]
    rng = np.random.default_rng(0)
    with (workdir / "annotations.csv").open("w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["video_path", "text"])
        for i, t in enumerate(texts):
            sid = f"clip_{i:03d}"
            T = int(kp.shape[0] * rng.uniform(0.6, 1.0))
            k = kp[:T] + rng.normal(0, 0.003, kp[:T].shape).astype(np.float32)
            np.savez_compressed(raw / f"{sid}.npz", keypoints=k, scores=sc[:T],
                                meta=json.dumps({"width": w, "height": h, "fps": fps}))
            wr.writerow([f"videos/{sid}.mp4", t])
        wr.writerow(["videos/clip_broken.mp4", "this one has no pose file"])
    arch = make_tiny_checkpoint(workdir / "tiny_unisign.pth")
    sets = [f"data.annotations_csv={workdir/'annotations.csv'}", f"data.raw_pose_dir={raw}",
            f"data.pose_dir={workdir/'pose'}", f"data.split_dir={workdir/'splits'}",
            f"pretrained.unisign_checkpoint={workdir/'tiny_unisign.pth'}",
            f"pretrained.unisign_repo={UNISIGN_REPO}", f"pretrained.mt5_dir={MT5_DIR}",
            f"model.debug_architecture={json.dumps(arch)}",
            "split.train_ratio=0.5", "split.val_ratio=0.25", "split.test_ratio=0.25",
            "training.mode=B", "training.mode_b.train_last_decoder_layers=1", "training.mode_b.train_last_encoder_layers=1",
            "training.epochs=2", "training.batch_size=4", "training.num_workers=0", "training.device=cpu",
            "training.log_every=1", f"training.output_dir={workdir/'ckpt'}", "training.lr_new=1e-3",
            "validation.generate_batches=1", "validation.max_frames=12", "validation.patience=5",
            "inference.max_frames=20", "inference.render_size=256", "normalization.min_hand_frac=0.0"]
    cfg = ROOT / "configs" / "auslan_sft.yaml"
    run([PY, "preprocessing/make_splits.py", "--config", cfg, "--set", *sets])
    out = run([PY, "preprocessing/normalize_pose.py", "--config", cfg, "--set", *sets])
    assert "rejected 1" in out, out
    out = run([PY, "verify_checkpoint.py", "--config", cfg, "--skip-sha", "--forward", "--set", *sets])
    assert "Missing parameters: 0" in out and "Unexpected parameters: 0" in out and "VERIFIED" in out, out
    out = run([PY, "train.py", "--config", cfg, "--set", *sets])
    print("\n".join(l for l in out.splitlines() if l.startswith(("Loaded", "Missing", "Unexpected", "Trainable", "[epoch", "[optim", "[data]"))))
    ck = torch.load(str(workdir / "ckpt" / "best.pt"), map_location="cpu", weights_only=False)
    for key in ("model", "optimizer", "scheduler", "epoch", "val_metric", "config", "pose_representation"):
        assert key in ck and ck[key] is not None, key
    # resume continues from latest.pt without error
    run([PY, "train.py", "--config", cfg, "--resume", "--set", *sets, "training.epochs=3"])
    # frozen groups must be bit-identical to the pretrained source
    tiny = torch.load(str(workdir / "tiny_unisign.pth"), map_location="cpu", weights_only=False)["model"]
    trained = torch.load(str(workdir / "ckpt" / "latest.pt"), map_location="cpu", weights_only=False)["model"]
    assert torch.equal(trained["pose_decoder.stack.block.0.layer.0.SelfAttention.q.weight"],
                       tiny["mt5_model.decoder.block.0.layer.0.SelfAttention.q.weight"]), "frozen decoder block changed"
    assert torch.equal(trained["text_encoder.embed.weight"], tiny["mt5_model.shared.weight"]), "frozen embedding changed"
    assert not torch.equal(trained["pose_decoder.stack.block.2.layer.2.DenseReluDense.wo.weight"],
                           tiny["mt5_model.decoder.block.2.layer.2.DenseReluDense.wo.weight"]), "trainable block did not move"
    ok("mode B: frozen tensors bit-identical to checkpoint, trainable last decoder block updated")
    run([PY, "evaluate.py", "--checkpoint", workdir / "ckpt" / "best.pt", "--split", "test", "--device", "cpu", "--render-n", "1"])
    m = json.loads((workdir / "ckpt" / "eval_test" / "metrics.json").read_text())
    assert m["n"] >= 1 and "dtw_distance" in m["model"] and "mpjpe_dtw" in m["mean_pose_baseline"], m
    demo = workdir / "out" / "demo.mp4"
    run([PY, "infer.py", "--checkpoint", workdir / "ckpt" / "best.pt", "--text", "I am going to university today.",
         "--output", demo, "--device", "cpu"])
    g = np.load(demo.with_suffix(".npy"))
    assert g.ndim == 3 and g.shape[1:] == (79, 2) and np.isfinite(g).all(), g.shape
    assert demo.is_file() and demo.stat().st_size > 1000
    run([PY, "render_pose.py", "--pose", demo.with_suffix(".npy"), "--output", workdir / "out" / "rerender.mp4"])
    gt_pt = next((workdir / "pose").glob("*.pt"))
    run([PY, "render_pose.py", "--pose", gt_pt, "--output", workdir / "out" / "gt.mp4"])
    import cv2
    cap = cv2.VideoCapture(str(workdir / "out" / "gt.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); okf, frame = cap.read(); cap.release()
    assert okf and n > 5, n
    cv2.imwrite(str(workdir / "out" / "gt_frame0.png"), frame)
    ok(f"end-to-end: split -> normalise -> verify -> train -> resume -> evaluate -> infer ({g.shape[0]} frames) -> render")


def test_feature_encoder_matches_unisign():
    from slp_models.pretrained_slp import UniSignPoseEncoder, load_pretrained, SignT5TextToPose  # noqa
    sys.path.append(str(REPO / "unisign"))
    import model_adapter
    full = model_adapter.load_unisign(UNISIGN_CKPT, UNISIGN_REPO, MT5_DIR).float().eval()
    fe = UniSignPoseEncoder(UNISIGN_REPO)
    st = {k: v.float() for k, v in full.state_dict().items() if k.startswith(("proj_linear", "gcn_modules", "fusion_gcn_modules", "part_para", "pose_proj"))}
    fe.load_state_dict(st, strict=True)
    fe.eval()
    torch.manual_seed(0)
    B, T = 2, 16
    src = {p: torch.rand(B, T, n, 3) * 2 - 1 for p, n in (("body", 9), ("left", 21), ("right", 21), ("face_all", 18))}
    with torch.no_grad():
        ours = fe(src)
        feats = []
        body = None
        for part in full.modes:   # replicate forward up to pose_proj using the real modules
            x = full.proj_linear[part](src[part]).permute(0, 3, 1, 2)
            g = full.gcn_modules[part](x)
            if part == "body":
                body = g
            else:
                g = g + body[..., {"left": -2, "right": -1, "face_all": 0}[part]][..., None]
            feats.append(full.fusion_gcn_modules[part](g).mean(-1).transpose(1, 2))
        ref = full.pose_proj(torch.cat(feats, -1) + full.part_para)
    assert torch.allclose(ours, ref, atol=1e-5), float((ours - ref).abs().max())
    ok(f"frozen feature encoder == Uni_Sign pose branch (max diff {float((ours-ref).abs().max()):.1e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--heavy", action="store_true")
    ap.add_argument("--keep", default=None, help="keep the end-to-end work dir here")
    args = ap.parse_args()
    frames, w, h, fps, kp, sc = test_extractor_parity(Path(args.video))
    frames_all, w, h, fps = read_video(Path(args.video))
    kp, sc = WholebodyExtractor()(frames_all, w, h)
    test_converter_parity(kp, sc, w, h, fps)
    test_normalisation(kp, sc, w, h, fps)
    test_loss_masking()
    work = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="auslan_sft_e2e_"))
    if work.exists() and args.keep:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    test_end_to_end(kp, sc, w, h, fps, work)
    if args.heavy:
        test_feature_encoder_matches_unisign()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
