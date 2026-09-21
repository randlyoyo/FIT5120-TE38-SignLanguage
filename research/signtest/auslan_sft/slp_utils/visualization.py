"""Stick-figure rendering with a FIXED camera.

The world-to-pixel transform is computed once per video (from dataset view
bounds, or from the whole sequence) and never per frame, so any motion on
screen is motion in the data, not camera re-framing.

Colours (BGR): signer's RIGHT hand orange, LEFT hand blue -- the convention of
this repo's overlay tools (README: "the orange skeleton must be on the hand you
actually raised"). Fingers get graded shades so handshapes are readable.
There are no hip keypoints in the representation (Uni-Sign's body part has
none, and Auslan-Daily crops rarely show hips); the torso is drawn as a fixed
trapezoid hanging from the shoulders and is marked as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from slp_models import skeleton as sk

BG = (250, 250, 250)
BODY = (60, 60, 60)
TORSO = (170, 170, 170)
FACE = (120, 90, 150)
BROW = (90, 60, 120)
RIGHT_SHADES = [(0, 90, 230), (0, 120, 240), (0, 150, 250), (40, 175, 255), (80, 200, 255)]
LEFT_SHADES = [(200, 90, 0), (220, 120, 20), (235, 150, 40), (245, 175, 80), (255, 200, 120)]


@dataclass
class Camera:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    width: int = 720
    height: int = 720
    margin: float = 0.06

    def __post_init__(self):
        sx = (self.width * (1 - 2 * self.margin)) / max(self.xmax - self.xmin, 1e-6)
        sy = (self.height * (1 - 2 * self.margin)) / max(self.ymax - self.ymin, 1e-6)
        self.scale = min(sx, sy)                                  # isotropic
        self.ox = self.width / 2 - self.scale * (self.xmin + self.xmax) / 2
        self.oy = self.height / 2 - self.scale * (self.ymin + self.ymax) / 2

    def project(self, p: np.ndarray) -> np.ndarray:
        # Image-convention y (down) is preserved: no sign flip.
        return np.stack([p[..., 0] * self.scale + self.ox, p[..., 1] * self.scale + self.oy], -1)


def camera_for(poses: list[np.ndarray], bounds: dict | None = None, size: int = 720) -> Camera:
    if bounds:
        b = bounds
        return Camera(b["xmin"], b["xmax"], b["ymin"], min(b["ymax"], b["ymin"] + 1.4 * (b["xmax"] - b["xmin"])),
                      size, size)
    allp = np.concatenate([p.reshape(-1, 2) for p in poses])
    allp = allp[np.isfinite(allp).all(-1)]
    if len(allp) == 0:
        return Camera(-2, 2, -2, 2, size, size)
    lo, hi = np.percentile(allp, 0.5, axis=0), np.percentile(allp, 99.5, axis=0)
    half = max(hi[0] - lo[0], hi[1] - lo[1]) / 2 + 0.3
    cx, cy = (lo + hi) / 2
    return Camera(cx - half, cx + half, cy - half, cy + half, size, size)


def draw_pose(img: np.ndarray, pose: np.ndarray, valid: np.ndarray | None, cam: Camera,
              label: str | None = None) -> np.ndarray:
    if valid is None:
        valid = np.isfinite(pose).all(-1)
    valid = valid & np.isfinite(pose).all(-1)
    px = cam.project(np.nan_to_num(pose))
    u = max(1, int(round(cam.scale * 0.02)))                     # line width ~ 0.02 shoulder widths

    def pt(j):
        return int(round(px[j, 0])), int(round(px[j, 1]))

    def line(a, b, color, w):
        if valid[a] and valid[b]:
            cv2.line(img, pt(a), pt(b), color, w, cv2.LINE_AA)

    ls, rs = sk.L_SHOULDER, sk.R_SHOULDER
    if valid[ls] and valid[rs]:
        sw = np.linalg.norm(pose[ls] - pose[rs])
        down = np.array([0.0, 1.6 * sw])
        hips = [pose[ls] + down + np.array([-0.1 * sw * np.sign(pose[ls, 0] - pose[rs, 0]), 0]),
                pose[rs] + down + np.array([0.1 * sw * np.sign(pose[ls, 0] - pose[rs, 0]), 0])]
        poly = cam.project(np.stack([pose[ls], pose[rs], hips[1], hips[0]])).astype(np.int32)
        cv2.polylines(img, [poly], True, TORSO, max(1, u), cv2.LINE_AA)   # drawn torso, not data

    # head: circle around the nose, radius from ear distance when available
    if valid[sk.NOSE]:
        if valid[sk.L_EAR] and valid[sk.R_EAR]:
            r = 0.6 * np.linalg.norm(pose[sk.L_EAR] - pose[sk.R_EAR])
        elif valid[ls] and valid[rs]:
            r = 0.28 * np.linalg.norm(pose[ls] - pose[rs])
        else:
            r = 0.25
        cv2.circle(img, pt(sk.NOSE), max(2, int(r * cam.scale)), BODY, u, cv2.LINE_AA)

    for a, b in sk.BODY_EDGES:
        line(a, b, BODY, 2 * u)
    for a, b in sk.FACE_EDGES:
        line(a, b, FACE, u)
    for a, b in sk.BROW_EDGES:
        line(a, b, BROW, u)
    for (a, b), shades in (((sk.L_WRIST_BODY, sk.L_HAND_ROOT), LEFT_SHADES), ((sk.R_WRIST_BODY, sk.R_HAND_ROOT), RIGHT_SHADES)):
        line(a, b, shades[0], 2 * u)
    for edges, shades in ((sk.LHAND_EDGES, LEFT_SHADES), (sk.RHAND_EDGES, RIGHT_SHADES)):
        for e, (a, b) in enumerate(edges):
            line(a, b, shades[sk.FINGER_OF_EDGE[e]], u + 1)
    for j in range(sk.NUM_JOINTS):
        if valid[j]:
            cv2.circle(img, pt(j), max(1, u), (30, 30, 30), -1, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    return img


def render_video(path: Path, panels: list[tuple[np.ndarray, np.ndarray | None, str]], fps: float,
                 cam: Camera, caption: str | None = None) -> Path:
    """panels: [(pose (T,J,2), valid (T,J) or None, label)], rendered side by side.

    Shorter panels hold their last frame. Returns the written path.
    """
    if not panels:
        raise ValueError("nothing to render")
    for p, v, _ in panels:
        if p.ndim != 3 or p.shape[1:] != (sk.NUM_JOINTS, 2):
            raise ValueError(f"pose must be (T, {sk.NUM_JOINTS}, 2), got {p.shape}")
    T = max(p.shape[0] for p, _, _ in panels)
    W, H = cam.width * len(panels), cam.height
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H))
    if not writer.isOpened():
        raise IOError(f"OpenCV could not open a video writer for {path}")
    try:
        for t in range(T):
            frame = np.full((H, W, 3), BG, dtype=np.uint8)
            for k, (p, v, label) in enumerate(panels):
                tt = min(t, p.shape[0] - 1)
                sub = frame[:, k * cam.width:(k + 1) * cam.width]
                draw_pose(sub, p[tt], None if v is None else v[tt], cam, label)
            if caption:
                cv2.putText(frame, caption[:90], (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{t+1}/{T}", (W - 90, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
    return path
