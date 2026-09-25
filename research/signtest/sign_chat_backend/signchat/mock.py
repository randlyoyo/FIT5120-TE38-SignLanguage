"""Stand-ins for the two sign models, with the same interface and output files, so the frontend and
the API can be developed and tested on a laptop with no GPU, weights or datasets (config mock: true).

The mock recogniser returns a fixed sentence; the mock generator writes a pose JSON of the same
format whose right arm waves, and an mp4 when OpenCV is installed. Nothing here is a model output.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

FPS = 25
N_JOINTS = 127


class MockRecognizer:
    def __init__(self, cfg, device=None, log=print):
        log("[sign2text] MOCK recogniser (no model loaded)")

    def recognise(self, video_path: str, mirrored: bool = False) -> dict:
        size = os.path.getsize(video_path)
        if size == 0:
            raise ValueError("the uploaded video has no decodable frames")
        return {"text": "hello , how are you ?", "frames": 0, "source_fps": FPS, "size": [0, 0],
                "signer_visible": 1.0, "timings": {"decode_video": 0.0, "pose": 0.0, "translate": 0.0}, "mock": True}


class MockGenerator:
    def __init__(self, cfg, device=None, log=print):
        self.cfg = cfg["text2sign"]
        log("[text2sign] MOCK generator (no model loaded)")

    def generate(self, sentence: str, out_dir: str, name: str) -> dict:
        t0 = time.time()
        T = int(np.clip(round(22.6 + 5.39 * len(sentence.split())), 20, 300))   # the real model's length fit
        t = np.arange(T) / FPS
        body = np.zeros((T, 63), np.float32)
        body[:, 16 * 3 + 2] = -1.2                      # right shoulder (SMPL-X joint 17) raised
        body[:, 18 * 3 + 1] = 0.6 * np.sin(2 * np.pi * 1.5 * t)   # right elbow (joint 19) waves
        z = lambda n: np.zeros((T, n), np.float32).tolist()
        pose = {"format": "smplx-v1", "fps": FPS, "frames": T, "sentence": sentence, "retrieved": "", "seen": False,
                "rotation": "axis-angle (radians), SMPL-X joint order; MOCK data", "mock": True,
                "smplx": {"global_orient": z(3), "body_pose": body.tolist(), "left_hand_pose": z(45),
                          "right_hand_pose": z(45), "jaw_pose": z(3), "leye_pose": z(3), "reye_pose": z(3),
                          "expression": z(50), "betas": z(10), "transl": z(3)},
                "joints": np.zeros((T, N_JOINTS, 3), np.float32).tolist(), "parents": [-1] + [0] * (N_JOINTS - 1)}
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{name}.json"), "w") as fh:
            json.dump(pose, fh, separators=(",", ":"))
        video = self._video(os.path.join(out_dir, f"{name}.mp4"), sentence, T) if self.cfg["render_video"] else None
        return {"pose_file": f"{name}.json", "video_file": video, "frames": T, "fps": FPS, "retrieved": "", "seen": False,
                "timings": {"generate": round(time.time() - t0, 3), "render": 0.0}, "mock": True}

    @staticmethod
    def _video(path, sentence, T):
        try:
            import cv2
        except ImportError:
            return None
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (320, 320))
        for k in range(T):
            img = np.full((320, 320, 3), 245, np.uint8)
            cv2.circle(img, (160, 90), 28, (60, 60, 60), 2)
            cv2.line(img, (160, 118), (160, 230), (60, 60, 60), 2)
            ang = 0.6 * np.sin(2 * np.pi * 1.5 * k / FPS)
            hand = (int(215 + 50 * np.sin(ang)), int(95 - 50 * np.cos(ang)))
            cv2.line(img, (160, 140), (215, 145), (0, 120, 230), 3)
            cv2.line(img, (215, 145), hand, (0, 120, 230), 3)
            cv2.putText(img, "MOCK: " + sentence[:34], (8, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            vw.write(img)
        vw.release()
        return os.path.basename(path)
