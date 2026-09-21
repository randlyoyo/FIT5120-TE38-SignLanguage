"""Differentiable conversion: signer-space pose -> Uni-Sign's exact part tensors.

This lets the frozen, pretrained Uni-Sign pose encoder read GENERATED poses
(feature loss) and lets a Uni-Sign SLR model back-translate them. It is a
torch port of unisign/spec.py::load_part_kp, which itself matches
Uni-Sign datasets.py to 3e-08:

    x_norm = (x_hat * s + cx) / W,   y_norm = (y_hat * s + cy) / H     undo our normalisation
    body   : ((p_norm - (xs, ys)) / scale - 0.5) * 2                  crop_scale
    hands  : (p_norm - p_norm[wrist]) / scale                         root = element 0
    face   : (p_norm - p_norm[nose tip]) / scale                      root = LAST element
    all    : clip to [-1, 1]; joints below threshold -> all three channels zero

`tests/test_pipeline.py` checks this against spec.load_part_kp numerically.

Camera parameters per sample, in this order (CAMERA_KEYS):
    cx, cy, s   (pixels)  from normalize_pose
    W, H        (pixels)
    xs, ys, scale         Uni-Sign crop box, frame-normalised units
For generated poses there is no source video; `canonical_camera` in the
normalisation stats supplies dataset medians instead.
"""

from __future__ import annotations

import torch

from slp_models import skeleton as sk

CAMERA_KEYS = ("cx", "cy", "s", "W", "H", "xs", "ys", "scale")


def camera_vector(norm: dict, crop: dict) -> list[float]:
    return [norm["center_px"][0], norm["center_px"][1], norm["scale_px"],
            float(norm["width"]), float(norm["height"]), crop["xs"], crop["ys"], crop["scale"]]


def to_unisign_parts(pose: torch.Tensor, valid: torch.Tensor, camera: torch.Tensor,
                     confidence: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """pose (B,T,79,2), valid (B,T,79) bool, camera (B,8), confidence (B,T,79)|(79,)|None.

    Returns {part: (B,T,J,3)} in Uni-Sign order and normalisation.
    """
    if pose.dim() != 4 or pose.shape[2] != sk.NUM_JOINTS or pose.shape[3] != 2:
        raise ValueError(f"pose must be (B,T,{sk.NUM_JOINTS},2), got {tuple(pose.shape)}")
    if camera.shape != (pose.shape[0], len(CAMERA_KEYS)):
        raise ValueError(f"camera must be (B,{len(CAMERA_KEYS)}), got {tuple(camera.shape)}")
    cam = camera.to(pose.dtype)[:, None, None, :]           # (B,1,1,8)
    cx, cy, s, W, H, xs, ys, scale = (cam[..., i] for i in range(8))
    xn = (pose[..., 0] * s + cx) / W
    yn = (pose[..., 1] * s + cy) / H
    pn = torch.stack([xn, yn], dim=-1)                       # frame-normalised
    sc = scale[..., None]

    if confidence is None:
        conf = torch.ones_like(pose[..., 0])
    else:
        conf = confidence.to(pose.dtype)
        if conf.dim() == 1:
            conf = conf.expand_as(pose[..., 0])
    v = valid.to(pose.dtype)

    out: dict[str, torch.Tensor] = {}
    for part in sk.UNISIGN_PARTS:
        sl = sk.PART_SLICES[part]
        p = pn[:, :, sl]
        if part == "body":
            origin = torch.stack([xs, ys], dim=-1)           # (B,1,1,2)
            q = ((p - origin) / sc - 0.5) * 2
        elif part in ("left", "right"):
            q = (p - p[:, :, :1]) / sc                       # wrist is element 0
        else:
            q = (p - p[:, :, -1:]) / sc                      # nose tip is LAST
        q = q.clamp(-1.0, 1.0)
        vp = v[:, :, sl]
        c = conf[:, :, sl].clamp(-1.0, 1.0) * vp          # upstream clips all 3 channels
        out[part] = torch.cat([q * vp[..., None], c[..., None]], dim=-1)
    return out


def pad_with_last_frame(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Uni-Sign pads with the LAST real frame, not zeros (zero = 'joint missing')."""
    B, T = x.shape[:2]
    idx = torch.arange(T, device=x.device)[None].expand(B, T)
    last = (lengths.to(x.device) - 1).clamp(min=0)[:, None]
    gather = torch.minimum(idx, last)
    shape = gather.shape + (1,) * (x.dim() - 2)
    return torch.gather(x, 1, gather.view(shape).expand_as(x))
