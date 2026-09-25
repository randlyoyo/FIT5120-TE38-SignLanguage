"""API tests on the mock models: no GPU, weights or datasets needed.

    cd sign_chat_backend && python -m pytest -q tests
"""

from __future__ import annotations

import json
import os
import sys

os.environ["SIGNCHAT_NO_AUTOAPP"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from signchat._imports import scoped_path  # noqa: E402
from signchat.config import load_config  # noqa: E402
from signchat.dialogue import clean_reply  # noqa: E402
from signchat.pipeline import _pretty  # noqa: E402
from signchat.server import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    cfg = load_config(overrides={"mock": True, "media_dir": str(tmp_path / "media"),
                                 "dialogue": {"backend": "echo"}})
    with TestClient(create_app(cfg)) as c:
        yield c


def test_health(client):
    h = client.get("/api/health").json()
    assert h["mock"] is True
    assert all(v["ready"] for v in h["components"].values())


def test_chat_text_keeps_session_and_serves_pose(client):
    r = client.post("/api/chat/text", json={"text": "Hello, how are you?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["input"] == {"mode": "text", "text": "Hello, how are you?"}
    assert body["reply"]["text"] == "Hello, how are you?"                 # echo backend
    sign = body["reply"]["sign"]
    assert sign["pose_url"].startswith("/media/") and sign["frames"] >= 20
    pose = client.get(sign["pose_url"]).json()
    assert pose["format"] == "smplx-v1" and len(pose["smplx"]["body_pose"]) == sign["frames"]
    assert len(pose["smplx"]["body_pose"][0]) == 63 and len(pose["joints"][0]) == 127
    sid = body["session_id"]
    r2 = client.post("/api/chat/text", json={"text": "Thank you.", "session_id": sid}).json()
    assert r2["session_id"] == sid
    assert client.delete(f"/api/session/{sid}").json()["cleared"] is True


def test_chat_sign_upload(client):
    r = client.post("/api/chat/sign", files={"video": ("clip.webm", b"\x1a\x45\xdf\xa3" * 64, "video/webm")},
                    data={"mirrored": "true"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["input"]["mode"] == "sign"
    assert body["input"]["text"] == "Hello, how are you?"                 # mock recogniser, prettified
    assert body["reply"]["sign"]["pose_url"]


def test_bad_inputs(client):
    assert client.post("/api/chat/text", json={"text": ""}).status_code == 422
    assert client.post("/api/chat/sign", files={"video": ("x.txt", b"abc", "text/plain")}).status_code == 415
    assert client.post("/api/chat/sign", files={"video": ("x.mp4", b"", "video/mp4")}).status_code == 422


def test_translate_endpoints(client):
    s = client.post("/api/translate/text-to-sign", json={"text": "See you tomorrow."}).json()
    assert s["pose_url"] and s["fps"] == 25
    t = client.post("/api/translate/sign-to-text", files={"video": ("a.mp4", b"0" * 100, "video/mp4")}).json()
    assert t["raw_text"] == "hello , how are you ?" and t["text"] == "Hello, how are you?"


def test_clean_reply():
    assert clean_reply("Hi! I'm fine, thanks. And you? I had a long day at work today with lots.", 8) == "Hi! I'm fine, thanks. And you?"
    assert clean_reply("**Sure** - let's eat", 15) == "Sure - let's eat."
    assert clean_reply("one two three four five six", 3) == "one two three."
    assert clean_reply("", 5) == "Sorry, can you sign that again?"


def test_pretty():
    assert _pretty("i am hungry , let us eat .") == "I am hungry, let us eat."


def test_scoped_path_isolates_same_named_modules(tmp_path):
    for repo, val in (("a", 1), ("b", 2)):
        (tmp_path / repo).mkdir()
        (tmp_path / repo / "utils.py").write_text(f"VALUE = {val}\n")
    with scoped_path(str(tmp_path / "a")):
        import utils
        a = utils
    with scoped_path(str(tmp_path / "b")):
        import utils
        b = utils
    assert (a.VALUE, b.VALUE) == (1, 2)
    assert "utils" not in sys.modules or sys.modules["utils"] not in (a, b)


def test_config_resolves_relative_paths(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(json.dumps({"text2sign": {"weights_dir": "w"}, "media_dir": "m"}))
    cfg = load_config(str(p))
    assert cfg["text2sign"]["weights_dir"] == str(tmp_path / "w")
    assert cfg["media_dir"] == str(tmp_path / "m")
