"""One chat turn: what the user said (text, or a signing video) -> the avatar's reply as text + signing.

    text  ───────────────────────────────┐
                                         ├─> Dialogue.reply -> reply text -> SignGenerator -> pose JSON (+ mp4)
    video ─> SignRecognizer -> text ─────┘

Sessions keep the last few turns in memory so the dialogue model has context. They are lost on
restart, which is fine for a demo; a persistent store would go in SessionStore.
"""

from __future__ import annotations

import collections
import os
import re
import threading
import time
import uuid

import torch

from .dialogue import build_dialogue


class SessionStore:
    def __init__(self, turns: int, max_sessions: int = 1000):
        self.turns = turns
        self.max_sessions = max_sessions
        self.data: "collections.OrderedDict[str, collections.deque]" = collections.OrderedDict()
        self.lock = threading.Lock()

    def history(self, sid: str) -> list[dict]:
        with self.lock:
            return list(self.data.get(sid, ()))

    def add(self, sid: str, user: str, avatar: str) -> None:
        with self.lock:
            d = self.data.setdefault(sid, collections.deque(maxlen=2 * self.turns))
            d.extend([{"role": "user", "content": user}, {"role": "assistant", "content": avatar}])
            self.data.move_to_end(sid)
            while len(self.data) > self.max_sessions:
                self.data.popitem(last=False)

    def clear(self, sid: str) -> bool:
        with self.lock:
            return self.data.pop(sid, None) is not None


def _slug(text: str, n: int = 32) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:n] or "sign"


def _pretty(text: str) -> str:
    """'hello , how are you ?' (model output, Auslan-Daily style) -> 'Hello, how are you?'"""
    t = re.sub(r"\s+([.,!?;:'])", r"\1", text.strip())
    t = re.sub(r"\bi\b", "I", t)
    return t[:1].upper() + t[1:]


class ChatPipeline:
    def __init__(self, cfg: dict, log=print):
        self.cfg = cfg
        self.log = log
        self.media_dir = cfg["media_dir"]
        os.makedirs(self.media_dir, exist_ok=True)
        dev = cfg["device"]
        if dev == "cuda" and not torch.cuda.is_available():
            log("[pipeline] CUDA not available, using CPU (slow)")
            dev = "cpu"
        self.device = torch.device(dev)
        self.sessions = SessionStore(cfg["session_turns"])
        self.errors: dict[str, str] = {}

        if cfg["mock"]:
            from .mock import MockGenerator, MockRecognizer
            rec_cls, gen_cls = MockRecognizer, MockGenerator
        else:
            from .sign2text import SignRecognizer
            from .text2sign import SignGenerator
            rec_cls, gen_cls = SignRecognizer, SignGenerator
        # text2sign first: it keeps SignSparK on sys.path; sign2text loads Uni-Sign in a scope.
        self.generator = self._load("text2sign", gen_cls)
        self.recognizer = self._load("sign2text", rec_cls)
        dcfg = cfg if not cfg["mock"] or cfg["dialogue"]["backend"] != "hf" else \
            {**cfg, "dialogue": {**cfg["dialogue"], "backend": "echo"}}     # mock mode never downloads an LLM
        self.dialogue = self._load("dialogue", lambda c, d, log: build_dialogue(c), cfg=dcfg)

    def _load(self, name, factory, cfg=None):
        cfg = cfg or self.cfg
        if name in ("sign2text", "text2sign") and not cfg[name]["enabled"]:
            self.errors[name] = "disabled in config"
            return None
        try:
            return factory(cfg, self.device, log=self.log)
        except Exception as e:                      # keep serving what did load; /api/health says what failed
            self.errors[name] = f"{type(e).__name__}: {e}"
            self.log(f"[pipeline] {name} failed to load: {self.errors[name]}")
            return None

    def status(self) -> dict:
        return {"mock": bool(self.cfg["mock"]), "device": str(self.device),
                "components": {n: {"ready": getattr(self, a) is not None, "error": self.errors.get(n)}
                               for n, a in (("sign2text", "recognizer"), ("text2sign", "generator"), ("dialogue", "dialogue"))},
                "dialogue_backend": getattr(self.dialogue, "name", None),
                "pose_providers": getattr(self.recognizer, "pose_providers", None)}

    def _need(self, attr: str, name: str):
        obj = getattr(self, attr)
        if obj is None:
            raise RuntimeError(f"{name} is not available: {self.errors.get(name, 'not loaded')}")
        return obj

    # ------------------------------------------------------------------ building blocks
    def sign_to_text(self, video_path: str, mirrored: bool = False) -> dict:
        r = self._need("recognizer", "sign2text").recognise(video_path, mirrored)
        return {**r, "raw_text": r["text"], "text": _pretty(r["text"])}

    def text_to_sign(self, text: str) -> dict:
        name = f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}_{_slug(text)}"
        return self._need("generator", "text2sign").generate(text, self.media_dir, name)

    # ------------------------------------------------------------------ chat turns
    def turn(self, session_id: str | None, text: str | None = None, video_path: str | None = None,
             mirrored: bool = False) -> dict:
        sid = session_id or uuid.uuid4().hex
        t0 = time.time()
        if video_path is not None:
            rec = self.sign_to_text(video_path, mirrored)
            user_input = {"mode": "sign", "text": rec["text"], "recognition": rec}
        else:
            if not text or not text.strip():
                raise ValueError("empty message")
            user_input = {"mode": "text", "text": text.strip()}
        t1 = time.time()
        reply = self._need("dialogue", "dialogue").reply(self.sessions.history(sid), user_input["text"])
        t2 = time.time()
        sign = self.text_to_sign(reply)
        self.sessions.add(sid, user_input["text"], reply)
        return {"session_id": sid, "input": user_input, "reply": {"text": reply, "sign": sign},
                "timings": {"input": round(t1 - t0, 3), "dialogue": round(t2 - t1, 3),
                            "sign": round(time.time() - t2, 3), "total": round(time.time() - t0, 3)}}
