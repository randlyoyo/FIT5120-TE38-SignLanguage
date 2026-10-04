"""The avatar's stage: one fixed camera for every reply, a resting pose, and the 2D skeleton the
browser draws (canvas) or the server draws (mp4).

The page shows the avatar like a person on a video call: between replies it stands with its hands
down and breathes (`idle`); a reply eases out of that pose, signs, and eases back into it. So:

  * every reply starts and ends at the rest pose: `with_transitions` adds `lead_in` frames from rest
    to the first signed frame and `lead_out` frames from the last one back to rest, interpolating
    each joint's rotation (slerp), so limbs keep their length;
  * the camera is fixed, set once from the rest pose (`Stage`), so the avatar does not move or
    change size from one reply to the next. Coordinates are normalised to a square: (0, 0) top left,
    (1, 1) bottom right; draw them in a square of side min(width, height) of the canvas.

Joint numbering, bones and colours follow auslan_smplx/signspark_render.py (SMPL-X, 127 joints).
"""

from __future__ import annotations

import os
import subprocess

import numpy as np

# SMPL-X joints (127-joint output), as in signspark_render
PELVIS, SPINE3, NECK, L_COLLAR, R_COLLAR, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = \
    0, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21
LEGS = {1, 2, 4, 5, 7, 8, 10, 11}
LEFT = {16, 18, 20} | set(range(25, 40)) | set(range(66, 71))
RIGHT = {17, 19, 21} | set(range(40, 55)) | set(range(71, 76))
FINGERS = set(range(25, 55)) | set(range(66, 76))
TIPS = [(39, 66), (27, 67), (30, 68), (36, 69), (33, 70), (54, 71), (42, 72), (45, 73), (51, 74), (48, 75)]
FACE = list(range(76, 127))

# colours of signspark_render's video: the signer's right side orange, left side blue
COLOURS = {"right": "#e65a00", "left": "#0078e6", "centre": "#3c3c3c", "face": "#282828"}
LINE_WIDTH, FINGER_WIDTH, POINT_RADIUS = 0.008, 0.005, 0.0035      # fractions of the square's side

ROT_KEYS = ("global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "jaw_pose", "leye_pose", "reye_pose")


# --------------------------------------------------------------------------- rotations
def _aa_to_quat(aa):
    """(..., 3) axis-angle -> (..., 4) unit quaternion (w, x, y, z)."""
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    half = th / 2
    k = np.where(th > 1e-8, np.sin(half) / np.maximum(th, 1e-12), 0.5)
    return np.concatenate([np.cos(half), aa * k], -1)


def _quat_to_aa(q):
    q = np.where(q[..., :1] < 0, -q, q)
    s = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    th = 2 * np.arctan2(s, q[..., :1])
    k = np.where(s > 1e-8, th / np.maximum(s, 1e-12), 2.0)
    return q[..., 1:] * k


def slerp_aa(a, b, w):
    """Axis-angle rotations a, b (..., 3) -> the rotation a fraction w of the way from a to b."""
    qa, qb = _aa_to_quat(a), _aa_to_quat(b)
    d = (qa * qb).sum(-1, keepdims=True)
    qb = np.where(d < 0, -qb, qb)
    d = np.clip(np.abs(d), -1.0, 1.0)
    th = np.arccos(d)
    sin = np.sin(th)
    near = sin < 1e-6
    wa = np.where(near, 1 - w, np.sin((1 - w) * th) / np.where(near, 1, sin))
    wb = np.where(near, w, np.sin(w * th) / np.where(near, 1, sin))
    q = wa * qa + wb * qb
    return _quat_to_aa(q / np.linalg.norm(q, axis=-1, keepdims=True))


def ease(n):
    """n weights strictly between 0 and 1, slow at both ends (smoothstep)."""
    t = np.arange(1, n + 1) / (n + 1)
    return t * t * (3 - 2 * t)


def blend(a: dict, b: dict, w: float) -> dict:
    """Two poses (dicts of (1, D) SMPL-X parameters) -> the pose a fraction w of the way from a to b:
    rotations by slerp joint by joint, everything else (expression, translation, shape) linearly."""
    out = {}
    for k in a:
        if k in ROT_KEYS:
            out[k] = slerp_aa(a[k].reshape(-1, 3), b[k].reshape(-1, 3), w).reshape(a[k].shape)
        else:
            out[k] = (1 - w) * a[k] + w * b[k]
    return out


def with_transitions(params: dict, rest: dict, lead_in: int, lead_out: int) -> dict:
    """SMPL-X parameters of a signed clip, (T, D) each -> (lead_in + T + lead_out, D): the clip eased
    in from `rest` and back out to it. Parameters that are not per frame, or missing from rest, keep
    their first / last value over the added frames."""
    T = len(params["body_pose"])
    first = {k: v[:1] for k, v in params.items()}
    last = {k: v[-1:] for k, v in params.items()}
    out = {}
    for k, v in params.items():
        if len(v) != T or k not in rest:
            out[k] = np.concatenate([np.repeat(v[:1], lead_in, 0), v, np.repeat(v[-1:], lead_out, 0)]) if len(v) == T else v
            continue
        r = rest[k].reshape(1, -1).astype(v.dtype)
        a = [blend({k: r}, {k: first[k]}, w)[k] for w in ease(lead_in)]
        b = [blend({k: last[k]}, {k: r}, w)[k] for w in ease(lead_out)]
        out[k] = np.concatenate(a + [v] + b).astype(v.dtype)
    return out


def arms_down(body_pose, sign=1.0, shoulder=1.4, elbow=0.2):
    """A copy of (N, 63) SMPL-X body_pose with both arms lowered from the T-pose to the sides and
    the elbows slightly bent. `sign` flips the rotation direction (picked by the caller with forward
    kinematics, so a different rig convention cannot raise the arms instead)."""
    b = np.array(body_pose, dtype=np.float32, copy=True)
    b[:, (L_SHOULDER - 1) * 3 + 2] = -sign * shoulder
    b[:, (R_SHOULDER - 1) * 3 + 2] = sign * shoulder
    b[:, (L_ELBOW - 1) * 3 + 1] = -sign * elbow
    b[:, (R_ELBOW - 1) * 3 + 1] = sign * elbow
    return b


def breathing(rest: dict, frames: int, sign=1.0, chest=0.012, shoulders=0.015) -> dict:
    """`frames` copies of the rest pose with one slow breath: the upper back straightens a little and
    the shoulders rise, then return. Frame 0 is the rest pose itself, and the loop is seamless."""
    b = (1 - np.cos(2 * np.pi * np.arange(frames) / frames)) / 2          # 0 -> 1 -> 0
    out = {k: np.repeat(v.reshape(1, -1), frames, 0).astype(np.float32) for k, v in rest.items()}
    bp = out["body_pose"]
    bp[:, (SPINE3 - 1) * 3 + 0] += -chest * b
    bp[:, (L_COLLAR - 1) * 3 + 2] += sign * shoulders * b
    bp[:, (R_COLLAR - 1) * 3 + 2] += -sign * shoulders * b
    return out


# --------------------------------------------------------------------------- the camera and the drawing
def bones(parents) -> list[tuple[int, int]]:
    """Upper body, head, hands and fingertips (signspark_render.bones)."""
    return [(j, int(parents[j])) for j in range(1, 55)
            if j not in LEGS and int(parents[j]) not in LEGS and parents[j] >= 0] + TIPS


def side(j: int) -> str:
    return "right" if j in RIGHT else "left" if j in LEFT else "centre"


class Stage:
    """A fixed front camera framed on the rest pose: the shoulders' mid-point is centred, the neck
    sits `neck_at` of the way down, and the square spans `width_m` metres (room for the hands above
    the head and out to the sides)."""

    def __init__(self, rest_joints, idle_joints, parents, fps=25, width_m=1.1, neck_at=0.30,
                 lead_in=8, lead_out=10):
        rest_joints = np.asarray(rest_joints, np.float32)
        self.cx = float(rest_joints[[L_SHOULDER, R_SHOULDER], 0].mean())
        self.neck_y = float(rest_joints[NECK, 1])
        self.width_m, self.neck_at = float(width_m), float(neck_at)
        self.fps, self.lead_in, self.lead_out = fps, int(lead_in), int(lead_out)
        self.parents = [int(p) for p in parents]
        self.bones = bones(self.parents)
        self.rest = self.project(rest_joints)
        self.idle = self.project(np.asarray(idle_joints, np.float32))

    def project(self, J) -> np.ndarray:
        """(..., 127, 3) joints in metres -> (..., 127, 2) on the stage, rounded to 1e-4."""
        J = np.asarray(J, np.float32)
        u = 0.5 + (J[..., 0] - self.cx) / self.width_m
        v = self.neck_at + (self.neck_y - J[..., 1]) / self.width_m
        return np.round(np.stack([u, v], -1), 4)

    def rig(self) -> dict:
        """Everything a page needs to draw the avatar (GET /api/avatar)."""
        return {
            "format": "stage-v1",
            "fps": self.fps,
            "coordinates": "x, y in [0, 1] of a square, y down; draw in a square of side min(canvas width, height)",
            "bones": [{"a": a, "b": b, "color": COLOURS[side(a)],
                       "width": FINGER_WIDTH if a in FINGERS else LINE_WIDTH} for a, b in self.bones],
            "points": {"joints": FACE, "color": COLOURS["face"], "radius": POINT_RADIUS},
            "background": "#ffffff",
            "rest": self.rest.tolist(),
            "idle": self.idle.tolist(),
            "lead_in": self.lead_in,
            "lead_out": self.lead_out,
        }

    def draw(self, P, size: int):
        """One frame (127, 2) of stage coordinates -> a size x size BGR image, as a page draws it."""
        import cv2
        img = np.full((size, size, 3), 255, np.uint8)
        Q = np.round(np.asarray(P) * size * 4).astype(np.int32)              # 2 bits of sub-pixel precision
        bgr = lambda h: tuple(int(h[i:i + 2], 16) for i in (5, 3, 1))
        for a, b in self.bones:
            width = max(1, round((FINGER_WIDTH if a in FINGERS else LINE_WIDTH) * size))
            cv2.line(img, tuple(Q[a]), tuple(Q[b]), bgr(COLOURS[side(a)]), width, cv2.LINE_AA, shift=2)
        r = max(1, round(POINT_RADIUS * size))
        for k in FACE:
            cv2.circle(img, tuple(Q[k]), r * 4, bgr(COLOURS["face"]), -1, cv2.LINE_AA, shift=2)
        return img

    def write_video(self, path: str, frames, size: int, captions=None) -> str:
        """(T, 127, 2) stage frames -> H.264 mp4 at `path`, with `captions` (one string per frame) in a
        bar under the avatar. mp4v is kept if ffmpeg is missing."""
        import cv2
        tmp = path + ".tmp.mp4"
        vw = None
        for t, P in enumerate(frames):
            img = self.draw(P, size)
            if captions is not None:
                img = np.concatenate([img, subtitle_bar(captions[t] if t < len(captions) else "", size)], 0)
            if vw is None:
                vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (img.shape[1], img.shape[0]))
            vw.write(img)
        vw.release()
        try:            # browsers need H.264; +faststart puts the index first, or <video> stalls at readyState 0
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", path], check=True)
            os.remove(tmp)
        except (FileNotFoundError, subprocess.CalledProcessError):
            os.replace(tmp, path)
        return path


def subtitle_bar(text: str, width: int, lines: int = 2):
    """signspark_render.subtitle_bar, sized to the video: a dark bar with `text` centred in white,
    wrapped to at most `lines` lines; the font shrinks until it fits. Same height with or without text."""
    import cv2
    font, th = cv2.FONT_HERSHEY_SIMPLEX, 2
    k = width / 480
    line_h = int(30 * k)
    bar = np.full((lines * line_h + int(12 * k), width, 3), 32, np.uint8)
    words = text.split()
    if not words:
        return bar
    for scale in np.arange(0.8, 0.39, -0.05) * k:
        rows, cur = [], ""
        for w in words:
            cand = f"{cur} {w}".strip()
            if cv2.getTextSize(cand, font, scale, th)[0][0] <= width - 16 or not cur:
                cur = cand
            else:
                rows.append(cur)
                cur = w
        rows.append(cur)
        if len(rows) <= lines and all(cv2.getTextSize(r, font, scale, th)[0][0] <= width - 16 for r in rows):
            break
    top = int(6 * k) + (lines - len(rows)) * line_h // 2
    for i, r in enumerate(rows[:lines]):
        tw = cv2.getTextSize(r, font, scale, th)[0][0]
        cv2.putText(bar, r, ((width - tw) // 2, top + (i + 1) * line_h - int(9 * k)), font, scale, (255, 255, 255), th, cv2.LINE_AA)
    return bar
