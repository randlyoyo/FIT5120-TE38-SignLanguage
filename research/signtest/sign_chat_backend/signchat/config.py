"""Configuration: built-in defaults, overridden by a YAML file, overridden by environment variables.

    SIGNCHAT_CONFIG=path/to/config.yaml   the YAML file (see config.example.yaml)
    SIGNCHAT_MOCK=1                       run without any model (for frontend development)
    SIGNCHAT_DIALOGUE=echo|hf|openai|anthropic
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # the repository root (holds unisign/ and auslan_smplx/)

DEFAULTS: dict = {
    "device": "cuda",
    "mock": False,                      # true: fake models, no GPU, same API (frontend development)
    "media_dir": "media",               # generated videos / pose files, served under /media
    "public_base_url": "",              # prefix for returned URLs, e.g. the tunnel URL; "" = relative
    "cors_origins": ["*"],
    "max_upload_mb": 50,
    "session_turns": 6,                 # dialogue history kept per session (user+avatar pairs)

    "sign2text": {
        "enabled": True,
        "code_dir": str(ROOT / "unisign"),              # our wrappers: extract_pose, spec, dataset, model_adapter
        "unisign_repo": "Uni-Sign",                     # clone of github.com/ZechengLi19/Uni-Sign @ eed438b
        "mt5_path": "Uni-Sign/pretrained_weight/mt5-base",
        # weights-only checkpoint written by train.py at the end of the run
        # arm_a__openasl_pose_only_slt__official_stage3__single_a100__bf16 (best: Comm BLEU-4 18.03)
        "checkpoint": "checkpoints/unisign_auslan_openasl_arm_a.pt",
        "num_beams": 4,                 # the setting every reported score used (plain decoding)
        "max_new_tokens": 100,
        "max_length": 256,              # frames fed to the model; longer clips are strided down
        "pose_device": "cpu",           # rtmlib/onnxruntime: "cuda" needs onnxruntime-gpu
        "pose_batch": 32,
        "pose_cudnn_algo": "HEURISTIC",  # onnxruntime cudnn_conv_algo_search; EXHAUSTIVE (its default) was 2.5x slower
        "target_fps": 25,               # Auslan-Daily is 25 fps; faster webcams are resampled
        "max_seconds": 20,
        "min_person_frames": 0.5,       # fraction of frames with a visible signer, else rejected
    },

    "text2sign": {
        "enabled": True,
        "code_dir": str(ROOT / "auslan_smplx"),         # signspark_ft.py / signspark_render.py
        "signspark_repo": "SignSparK",                  # clone of github.com/JianHe0628/SignSparK @ 22a0b4e
        # hand.pt / body.pt / face.pt (round 2), or the slim copies from scripts/slim_assets.py
        "weights_dir": "checkpoints/signspark_ft_smooth",
        # retrieval bank = round 2's training set: the LMDB, or bank_compact.npz from scripts/slim_assets.py
        "bank_lmdb": "data/lmdb_smooth/train/AuslanDaily_train.lmdb",
        "smplx_npz": "models/SMPLX_NEUTRAL_2020.npz",
        "steps": 20,
        "text_scale": 2.5,
        "sigma": 1.0,
        "seed": 0,
        "encoder_on_gpu": False,        # keep the three M-CLIP text encoders on the GPU (+6.6 GB, faster per sentence)
        "render_video": True,           # skeleton mp4 for quick display; the avatar uses the pose JSON
        "video_size": 480,
    },

    "dialogue": {
        "backend": "hf",                # hf | openai | anthropic | echo
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "device": None,                 # hf only; None = same as `device`
        "max_new_tokens": 48,
        "temperature": 0.7,
        "max_words": 15,                # the reply is signed, and long sentences sign badly (EXPERIMENTS.md)
        "base_url": "http://localhost:8001/v1",         # openai-compatible servers (vLLM, Ollama, ...)
        "api_key_env": "OPENAI_API_KEY",                # openai backend; the anthropic SDK reads its own env
        "system_prompt": None,          # None = dialogue.DEFAULT_SYSTEM_PROMPT
    },
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve(cfg: dict, base: Path) -> dict:
    """Relative paths in the file are relative to the file's own folder."""
    keys = {"sign2text": ["code_dir", "unisign_repo", "mt5_path", "checkpoint"],
            "text2sign": ["code_dir", "signspark_repo", "weights_dir", "bank_lmdb", "smplx_npz"]}
    for sec, names in keys.items():
        for n in names:
            p = cfg[sec].get(n)
            if p and not os.path.isabs(p):
                cfg[sec][n] = str((base / p).resolve())
    if not os.path.isabs(cfg["media_dir"]):
        cfg["media_dir"] = str((base / cfg["media_dir"]).resolve())
    return cfg


def load_config(path: str | None = None, overrides: dict | None = None) -> dict:
    path = path or os.environ.get("SIGNCHAT_CONFIG")
    cfg = copy.deepcopy(DEFAULTS)
    base = Path.cwd()
    if path:
        import yaml
        with open(path) as fh:
            cfg = _merge(cfg, yaml.safe_load(fh) or {})
        base = Path(path).resolve().parent
    if os.environ.get("SIGNCHAT_MOCK", "").lower() in ("1", "true", "yes"):
        cfg["mock"] = True
    if os.environ.get("SIGNCHAT_DIALOGUE"):
        cfg["dialogue"]["backend"] = os.environ["SIGNCHAT_DIALOGUE"]
    cfg = _merge(cfg, overrides or {})
    return _resolve(cfg, base)
