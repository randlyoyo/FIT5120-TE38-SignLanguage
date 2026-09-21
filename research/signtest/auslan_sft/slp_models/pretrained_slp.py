"""English text -> Auslan pose model built on the pretrained Uni-Sign checkpoint.

See PRETRAINED_MODEL_ANALYSIS.md for why this checkpoint and what transfers.

    SignT5TextToPose
      text_encoder   mT5-base embedding + 12 encoder blocks   <- checkpoint
      pose_decoder   12 mT5-base decoder blocks               <- checkpoint
                     pose_in / pose_out / counter_out          <- NEW
      buffers        pose_mean / pose_std (training-set stats)

    UniSignPoseEncoder (frozen; feature loss, not part of generation)
      proj_linear / gcn_modules / fusion_gcn_modules / part_para / pose_proj  <- checkpoint

The loader maps checkpoint keys explicitly, verifies shapes, prints
loaded / missing / unexpected / trainable counts, and raises
PretrainedLoadError when the intended pretrained weights did not load.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from transformers import MT5Config

from slp_models import skeleton as sk
from slp_models.pose_decoder import PoseDecoder, _stack_forward, build_stack
from slp_utils.unisign_format import pad_with_last_frame, to_unisign_parts


class PretrainedLoadError(RuntimeError):
    pass


# --------------------------------------------------------------------------- text encoder

class TextEncoder(nn.Module):
    """mT5 encoder. Replaceable: anything returning (B, L, d_model) works."""

    def __init__(self, config: MT5Config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.stack = build_stack(config, is_decoder=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return _stack_forward(self.stack, self.embed(input_ids), attention_mask.long())


# --------------------------------------------------------------------------- Uni-Sign pose encoder

def import_unisign_stgcn(repo: Path):
    repo = Path(repo).resolve()
    if not (repo / "stgcn_layers").is_dir():
        raise FileNotFoundError(f"Uni-Sign repo with stgcn_layers/ not found at {repo}")
    if str(repo) not in sys.path:
        # Appended, not prepended: Uni-Sign has top-level utils.py / models.py /
        # config.py that must not shadow anything else.
        sys.path.append(str(repo))
    return importlib.import_module("stgcn_layers")


class UniSignPoseEncoder(nn.Module):
    """The pose branch of Uni-Sign models.py::Uni_Sign, constructed identically.

    Built from the Uni-Sign repo's own Graph and get_stgcn_chain, not
    re-implemented, so the checkpoint's keys and shapes line up exactly.
    """

    MODES = ("body", "left", "right", "face_all")

    def __init__(self, unisign_repo: Path, hidden_dim: int = 256):
        super().__init__()
        stg = import_unisign_stgcn(unisign_repo)
        A = []
        self.proj_linear = nn.ModuleDict()
        for mode in self.MODES:
            g = stg.Graph(layout=mode, strategy="distance", max_hop=1)
            A.append(torch.tensor(g.A, dtype=torch.float32, requires_grad=False))
            self.proj_linear[mode] = nn.Linear(3, 64)
        self.gcn_modules = nn.ModuleDict()
        self.fusion_gcn_modules = nn.ModuleDict()
        k = A[0].size(0)
        for i, mode in enumerate(self.MODES):
            self.gcn_modules[mode], final_dim = stg.get_stgcn_chain(64, "spatial", (1, k), A[i].clone(), True)
            self.fusion_gcn_modules[mode], _ = stg.get_stgcn_chain(final_dim, "temporal", (5, k), A[i].clone(), True)
        # Uni-Sign shares the right-hand branch with the left hand.
        self.gcn_modules["left"] = self.gcn_modules["right"]
        self.fusion_gcn_modules["left"] = self.fusion_gcn_modules["right"]
        self.proj_linear["left"] = self.proj_linear["right"]
        self.part_para = nn.Parameter(torch.zeros(hidden_dim * len(self.MODES)))
        self.pose_proj = nn.Linear(256 * 4, 768)

    def forward(self, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        feats, body_feat = [], None
        for part in self.MODES:
            x = self.proj_linear[part](parts[part]).permute(0, 3, 1, 2)   # B,C,T,V
            g = self.gcn_modules[part](x)
            if part == "body":
                body_feat = g
            elif part == "left":
                g = g + body_feat[..., -2][..., None].detach()
            elif part == "right":
                g = g + body_feat[..., -1][..., None].detach()
            else:
                g = g + body_feat[..., 0][..., None].detach()
            g = self.fusion_gcn_modules[part](g)
            feats.append(g.mean(-1).transpose(1, 2))                        # B,T,C
        return self.pose_proj(torch.cat(feats, dim=-1) + self.part_para)    # B,T,768


# --------------------------------------------------------------------------- generator

class SignT5TextToPose(nn.Module):
    def __init__(self, config: MT5Config, num_joints: int = sk.NUM_JOINTS, coord_dim: int = sk.COORD_DIM):
        super().__init__()
        self.config = config
        self.num_joints, self.coord_dim = num_joints, coord_dim
        self.text_encoder = TextEncoder(config)
        self.pose_decoder = PoseDecoder(config, num_joints, coord_dim)
        self.register_buffer("pose_mean", torch.zeros(num_joints, coord_dim))
        self.register_buffer("pose_std", torch.ones(num_joints, coord_dim))

    def set_pose_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != self.pose_mean.shape or std.shape != self.pose_std.shape:
            raise ValueError(f"pose stats shape {tuple(mean.shape)} != {tuple(self.pose_mean.shape)}")
        self.pose_mean.copy_(mean)
        self.pose_std.copy_(std)

    def standardise(self, pose: torch.Tensor) -> torch.Tensor:
        return (pose - self.pose_mean) / self.pose_std

    def destandardise(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.pose_std + self.pose_mean

    def encode(self, input_ids, attention_mask):
        return self.text_encoder(input_ids, attention_mask)

    def forward(self, input_ids, attention_mask, pose, frame_mask, input_noise_std: float = 0.0):
        """Teacher-forced pass.

        pose (B,T,J,C) signer space (NaN-free), frame_mask (B,T) bool.
        Returns pose_hat (B,T,J,C) signer space and counter_hat (B,T).
        """
        B, T, J, C = pose.shape
        lengths = frame_mask.sum(1).clamp(min=1)
        z = self.standardise(pose)
        bos = torch.zeros_like(z[:, :1])
        prev = torch.cat([bos, z[:, :-1]], dim=1)
        if input_noise_std > 0 and self.training:
            # Progressive Transformers' Gaussian-noise augmentation: the model
            # must learn to continue from imperfect frames, as at inference.
            prev = prev + torch.randn_like(prev) * input_noise_std
        t = torch.arange(T, device=pose.device, dtype=z.dtype)[None]
        prev_counter = (t / lengths[:, None].to(z.dtype)).clamp(max=1.0)
        enc = self.encode(input_ids, attention_mask)
        out_z, counter = self.pose_decoder(prev, prev_counter, frame_mask, enc, attention_mask)
        return self.destandardise(out_z), counter

    @staticmethod
    def counter_target(frame_mask: torch.Tensor) -> torch.Tensor:
        lengths = frame_mask.sum(1).clamp(min=1).to(torch.float32)
        t = torch.arange(frame_mask.shape[1], device=frame_mask.device, dtype=torch.float32)[None]
        return ((t + 1) / lengths[:, None]).clamp(max=1.0)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, max_frames: int = 300, min_frames: int = 8,
                 stop_counter: float = 0.97) -> list[torch.Tensor]:
        """Free-running autoregressive generation. Returns B tensors (T_i, J, C), signer space."""
        B = input_ids.shape[0]
        device = input_ids.device
        enc = self.encode(input_ids, attention_mask)
        J, C = self.num_joints, self.coord_dim
        prev = torch.zeros(B, 1, J, C, device=device, dtype=enc.dtype)
        prev_counter = torch.zeros(B, 1, device=device, dtype=enc.dtype)
        outs, counters = [], []
        lengths = torch.full((B,), max_frames, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        for step in range(max_frames):
            fm = torch.ones(B, step + 1, dtype=torch.bool, device=device)
            z, c = self.pose_decoder(prev, prev_counter, fm, enc, attention_mask)
            z_t, c_t = z[:, -1:], c[:, -1:]
            outs.append(z_t)
            counters.append(c_t)
            stop = (~done) & (c_t.squeeze(1) >= stop_counter) & (step + 1 >= min_frames)
            lengths[stop] = step + 1
            done |= stop
            if bool(done.all()):
                break
            prev = torch.cat([prev, z_t.to(prev.dtype)], dim=1)
            prev_counter = torch.cat([prev_counter, c_t.to(prev_counter.dtype)], dim=1)
        seq = self.destandardise(torch.cat(outs, dim=1).float())
        return [seq[i, : int(min(lengths[i].item(), seq.shape[1]))].cpu() for i in range(B)]


# --------------------------------------------------------------------------- loading

NEW_PREFIXES = ("pose_decoder.pose_in.", "pose_decoder.pose_out.", "pose_decoder.counter_out.")
BUFFER_KEYS = ("pose_mean", "pose_std")
# Checkpoint keys that exist but are intentionally not loaded.
IGNORED_CKPT_KEYS = ("lm_head.weight", "encoder.embed_tokens.weight", "decoder.embed_tokens.weight")
UNISIGN_POSE_PREFIXES = ("proj_linear.", "gcn_modules.", "fusion_gcn_modules.", "part_para", "pose_proj.")


def mt5_key_map(model_keys: list[str], enc_prefix: str | None, dec_prefix: str | None) -> dict[str, str]:
    """model key -> checkpoint key, for the parts that come from a checkpoint.

    enc_prefix/dec_prefix are the checkpoint's mT5 prefix ('mt5_model.' for
    Uni-Sign, '' for a HuggingFace mt5-base dir), or None when that half is not
    loaded (explicit random-init ablation).
    """
    m = {}
    for k in model_keys:
        if enc_prefix is not None:
            if k == "text_encoder.embed.weight":
                m[k] = f"{enc_prefix}shared.weight"
            elif k.startswith("text_encoder.stack."):
                m[k] = enc_prefix + "encoder." + k[len("text_encoder.stack."):]
        if dec_prefix is not None and k.startswith("pose_decoder.stack."):
            m[k] = dec_prefix + "decoder." + k[len("pose_decoder.stack."):]
    return m


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"pretrained checkpoint not found: {path}")
    try:
        obj = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
    except Exception:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
    for key in ("model", "state_dict", "module"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    return {k.replace("module.", "", 1): v for k, v in obj.items()}


def _source(cfg_p: dict, which: str) -> tuple[Path | None, str | None]:
    src = cfg_p.get(f"{which}_source", "unisign")
    if src == "unisign":
        return Path(cfg_p["unisign_checkpoint"]), "mt5_model."
    if src == "mt5_base":
        return Path(cfg_p["mt5_dir"]) / "pytorch_model.bin", ""
    if src == "none":
        return None, None
    raise ValueError(f"pretrained.{which}_source must be unisign|mt5_base|none, got {src!r}")


@dataclass
class LoadReport:
    loaded_tensors: int = 0
    loaded_params: int = 0
    missing: list[str] = field(default_factory=list)          # intended to load, absent/mis-shaped
    unexpected: list[str] = field(default_factory=list)       # in checkpoint, not used, not whitelisted
    ignored: list[str] = field(default_factory=list)          # in checkpoint, deliberately unused
    new_params: list[str] = field(default_factory=list)       # randomly initialised by design
    feature_encoder_loaded: int = 0
    total_params: int = 0
    trainable_params: int = 0

    @property
    def loaded_fraction(self) -> float:
        return self.loaded_params / max(1, self.total_params)

    def print(self) -> None:
        print("=" * 72)
        print(f"Loaded pretrained parameters: {self.loaded_params:,} ({self.loaded_tensors} tensors, "
              f"{self.loaded_fraction:.2%} of the generator)")
        print(f"Missing parameters: {len(self.missing)}" + (f"  e.g. {self.missing[:5]}" if self.missing else ""))
        print(f"Unexpected parameters: {len(self.unexpected)}" + (f"  e.g. {self.unexpected[:5]}" if self.unexpected else ""))
        print(f"Deliberately ignored checkpoint tensors: {len(self.ignored)} {self.ignored[:4]}")
        print(f"Newly initialised (by design): {len(self.new_params)} tensors {self.new_params}")
        print(f"Frozen Uni-Sign pose encoder tensors loaded: {self.feature_encoder_loaded}")
        print(f"Trainable parameters: {self.trainable_params:,} / Total parameters: {self.total_params:,}")
        print("=" * 72)


@torch.no_grad()
def load_pretrained(model: SignT5TextToPose, cfg_p: dict,
                    feature_encoder: UniSignPoseEncoder | None = None) -> LoadReport:
    rep = LoadReport()
    params = dict(model.named_parameters())
    rep.total_params = sum(p.numel() for p in params.values())
    rep.new_params = [k for k in params if k.startswith(NEW_PREFIXES)]

    enc_path, enc_prefix = _source(cfg_p, "encoder")
    dec_path, dec_prefix = _source(cfg_p, "decoder")
    random_parts = [w for w, p in (("text encoder", enc_path), ("pose decoder", dec_path)) if p is None]
    if random_parts:
        if not cfg_p.get("allow_random_init", False):
            raise PretrainedLoadError(
                f"{random_parts} configured with source 'none'. Refusing to train from random "
                "initialisation; set pretrained.allow_random_init: true only for an explicit ablation.")
        print(f"!!! WARNING: {random_parts} RANDOMLY INITIALISED (explicit ablation) !!!")

    states: dict[Path, dict] = {}
    for p in {enc_path, dec_path} - {None}:
        states[p] = _load_state(p)

    intended = {}
    for k, ck in mt5_key_map(list(params), enc_prefix, None).items():
        intended[k] = (enc_path, ck)
    for k, ck in mt5_key_map(list(params), None, dec_prefix).items():
        intended[k] = (dec_path, ck)

    used: dict[Path, set] = {p: set() for p in states}
    for k, (path, ck) in intended.items():
        st = states[path]
        if ck not in st or tuple(st[ck].shape) != tuple(params[k].shape):
            got = tuple(st[ck].shape) if ck in st else None
            rep.missing.append(f"{k} <- {ck} (checkpoint shape {got}, model {tuple(params[k].shape)})")
            continue
        params[k].copy_(st[ck].to(params[k].dtype))
        used[path].add(ck)
        rep.loaded_tensors += 1
        rep.loaded_params += params[k].numel()

    # Everything the model has that is neither loaded nor new is a gap.
    for k in params:
        if k not in intended and not k.startswith(NEW_PREFIXES):
            if (k.startswith("text_encoder.") and enc_path is None) or (k.startswith("pose_decoder.") and dec_path is None):
                continue
            rep.missing.append(f"{k} (no checkpoint mapping)")

    prefix_of = {}
    if enc_path is not None:
        prefix_of[enc_path] = enc_prefix
    if dec_path is not None:
        prefix_of[dec_path] = dec_prefix
    for path, st in states.items():
        prefix = prefix_of[path]
        for ck in st:
            if ck in used[path] or ck.startswith(UNISIGN_POSE_PREFIXES):
                continue  # loaded, or consumed by the feature encoder below
            if not ck.startswith(prefix):
                rep.unexpected.append(ck)
                continue
            bare = ck[len(prefix):]
            if bare in IGNORED_CKPT_KEYS:
                rep.ignored.append(ck)
            elif bare.startswith(("encoder.", "shared.")) and path != enc_path:
                continue  # encoder half of a checkpoint used only for its decoder
            elif bare.startswith("decoder.") and path != dec_path:
                continue
            else:
                rep.unexpected.append(ck)

    # Frozen Uni-Sign pose encoder, always from the Uni-Sign checkpoint.
    if feature_encoder is not None:
        up = Path(cfg_p["unisign_checkpoint"])
        st = states.get(up) or _load_state(up)
        fe_state = {k: v for k, v in st.items() if k.startswith(UNISIGN_POSE_PREFIXES)}
        if not fe_state:
            raise PretrainedLoadError(f"{up} has no Uni-Sign pose-encoder tensors")
        own = feature_encoder.state_dict()
        miss = [k for k in own if k not in fe_state or tuple(fe_state[k].shape) != tuple(own[k].shape)]
        if miss:
            raise PretrainedLoadError(f"Uni-Sign pose encoder: {len(miss)} tensors missing/mis-shaped, e.g. {miss[:5]}")
        extra = [k for k in fe_state if k not in own]
        if extra:
            raise PretrainedLoadError(f"Uni-Sign pose encoder: unexpected tensors {extra[:5]}")
        feature_encoder.load_state_dict({k: v.to(own[k].dtype) for k, v in fe_state.items()}, strict=True)
        rep.feature_encoder_loaded = len(fe_state)
        feature_encoder.requires_grad_(False).eval()

    # Match the new input projection's output scale to the pretrained embeddings.
    target_rms = float(model.text_encoder.embed.weight.float().pow(2).mean().sqrt())
    model.pose_decoder.reset_new_parameters(target_input_rms=target_rms)
    return rep


def check_load_gate(rep: LoadReport, cfg_p: dict) -> None:
    min_frac = float(cfg_p.get("min_loaded_fraction", 0.95))
    random_ok = bool(cfg_p.get("allow_random_init", False))
    if rep.missing:
        raise PretrainedLoadError(f"STOP: {len(rep.missing)} intended pretrained tensors did not load: {rep.missing[:5]}")
    if rep.unexpected:
        raise PretrainedLoadError(f"STOP: {len(rep.unexpected)} unexpected checkpoint tensors "
                                  f"(key mapping is wrong for this checkpoint): {rep.unexpected[:5]}")
    if rep.loaded_fraction < min_frac and not random_ok:
        raise PretrainedLoadError(f"STOP: only {rep.loaded_fraction:.1%} of generator parameters came from "
                                  f"pretrained weights (< {min_frac:.0%}). Not training from scratch.")


def mt5_config_from(cfg_model: dict, mt5_dir: Path) -> MT5Config:
    if not (Path(mt5_dir) / "config.json").is_file():
        raise FileNotFoundError(f"mt5-base config.json not found in {mt5_dir}")
    config = MT5Config.from_pretrained(str(mt5_dir))
    config.dropout_rate = float(cfg_model.get("dropout", config.dropout_rate))
    for k in ("num_layers", "num_decoder_layers", "d_model", "d_ff", "num_heads", "d_kv", "vocab_size"):
        if k in cfg_model.get("debug_architecture", {}):
            setattr(config, k, cfg_model["debug_architecture"][k])
    return config
