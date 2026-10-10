"""Stand-ins for the two sign models, with the same interface and output files, so the frontend and
the API can be developed and tested on a laptop with no GPU, weights or datasets (config mock: true).

The mock recogniser returns a fixed sentence; the mock generator writes a pose JSON of the same
format, in which the avatar eases out of its rest pose, waves its right arm and eases back, on the
same stage as the real model (stage.py), and an mp4 when OpenCV is installed. Its rest pose is a
stored SMPL-X skeleton (mock_rest.json), since there is no SMPL-X model here. Nothing here is a
model output.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from . import stage, subtitles

FPS = 25
N_JOINTS = 127
REST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mock_rest.json")


class MockRecognizer:
    def __init__(self, cfg, device=None, log=print):
        log("[sign2text] MOCK recogniser (no model loaded)")

    def recognise(self, video_path: str, mirrored: bool = False) -> dict:
        size = os.path.getsize(video_path)
        if size == 0:
            raise ValueError("the uploaded video has no decodable frames")
        return {"text": "hello , how are you ?", "frames": 0, "source_fps": FPS, "size": [0, 0],
                "signer_visible": 1.0, "timings": {"decode_video": 0.0, "pose": 0.0, "translate": 0.0}, "mock": True}


def _subtree(parents, root):
    """Joints moved by `root`: its descendants in the 55-joint tree, plus the fingertips and face
    landmarks hanging off them (joints 55-126 have no parent in SMPL-X's tree)."""
    out = {root}
    for j in range(len(parents)):
        k = j
        while k >= 0 and k not in out:
            k = parents[k]
        if k >= 0:
            out.add(j)
    out |= {tip for base, tip in stage.TIPS if base in out}
    if 15 in out:                                        # the head carries the face
        out |= set(range(55, 60)) | set(stage.FACE)
    return sorted(out)


def _turn(J, joints, pivot, angle):
    """Rotate `joints` of every frame of J (T, 127, 3) about joint `pivot` in the picture plane."""
    c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
    p = J[:, [pivot], :2]
    d = J[:, joints, :2] - p
    J[:, joints, 0] = p[..., 0] + c * d[..., 0] - s * d[..., 1]
    J[:, joints, 1] = p[..., 1] + s * d[..., 0] + c * d[..., 1]


class MockGenerator:
    def __init__(self, cfg, device=None, log=print):
        self.cfg = c = cfg["text2sign"]
        ref = json.load(open(REST_FILE))
        self.parents = ref["parents"]
        self.rest = np.array(ref["rest"], np.float32)
        n = c["idle_frames"]
        idle = np.repeat(self.rest[None], n, 0)
        upper = _subtree(self.parents, stage.SPINE3)
        idle[:, upper, 1] += 0.004 * (1 - np.cos(2 * np.pi * np.arange(n) / n))[:, None] / 2   # a breath
        self.stage = stage.Stage(self.rest, idle, self.parents, FPS, c["stage_width_m"], c["stage_neck_at"],
                                 c["lead_in"], c["lead_out"])
        log("[text2sign] MOCK generator (no model loaded)")

    def generate(self, sentence: str, out_dir: str, name: str) -> dict:
        t0 = time.time()
        a, b = self.stage.lead_in, self.stage.lead_out
        T = int(np.clip(round(22.6 + 5.39 * len(sentence.split())), 20, 300))   # the real model's length fit
        N = a + T + b
        env = np.concatenate([stage.ease(a), np.ones(T), stage.ease(b)[::-1]])   # rest -> signing -> rest
        J = np.repeat(self.rest[None], N, 0)
        _turn(J, _subtree(self.parents, stage.R_ELBOW), stage.R_ELBOW,
              env * 0.5 * np.sin(2 * np.pi * 1.5 * np.arange(N) / FPS))              # the forearm waves
        _turn(J, _subtree(self.parents, stage.R_SHOULDER), stage.R_SHOULDER, -1.9 * env)   # the arm is raised
        os.makedirs(out_dir, exist_ok=True)
        cues, vtt = subtitles.write(out_dir, name, sentence, T, FPS, offset=a / FPS)
        P = self.stage.project(J)
        z = lambda n: np.zeros((N, n), np.float32).tolist()
        pose = {"format": "smplx-v1", "fps": FPS, "frames": N, "lead_in": a, "lead_out": b,
                "sentence": sentence, "retrieved": "", "seen": False,
                "rotation": "axis-angle (radians), SMPL-X joint order; MOCK data (the rotations are all zero)", "mock": True,
                "smplx": {"global_orient": z(3), "body_pose": z(63), "left_hand_pose": z(45),
                          "right_hand_pose": z(45), "jaw_pose": z(3), "leye_pose": z(3), "reye_pose": z(3),
                          "expression": z(50), "betas": z(10), "transl": z(3)},
                "joints": np.round(J, 4).tolist(), "parents": self.parents,
                "joints2d": P.tolist(), "subtitles": cues}
        with open(os.path.join(out_dir, f"{name}.json"), "w") as fh:
            json.dump(pose, fh, separators=(",", ":"))
        video = self._video(os.path.join(out_dir, f"{name}.mp4"), P, subtitles.per_frame(cues, N, FPS)) \
            if self.cfg["render_video"] else None
        return {"pose_file": f"{name}.json", "video_file": video, "video_status": "ready" if video else "off",
                "subtitle_file": vtt, "subtitles": cues,
                "frames": N, "lead_in": a, "lead_out": b, "fps": FPS, "retrieved": "", "seen": False,
                "timings": {"generate": round(time.time() - t0, 3), "write": 0.0}, "mock": True}

    def _video(self, path, P, captions):
        try:
            import cv2  # noqa: F401
        except ImportError:
            return None
        self.stage.write_video(path, P, self.cfg["video_size"], captions)
        return os.path.basename(path)
