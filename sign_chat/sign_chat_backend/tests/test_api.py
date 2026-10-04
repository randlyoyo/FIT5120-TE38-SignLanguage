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
    assert sign["video_status"] in ("ready", "off")
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
    assert clean_reply("Hi! I'm fine, thanks. And you? I had a long day at work today with lots.", 8) == "Hi! I am fine, thanks. And you?"
    assert clean_reply("**Sure** - let's eat", 15) == "Sure - let us eat."
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


def test_contractions_expanded():
    from signchat.dialogue import expand_contractions
    assert clean_reply("I'm good, thanks. How about you?", 15) == "I am good, thanks. How about you?"
    assert expand_contractions("Don't worry, it's fine and we'll go. Can't wait! Let's eat.") == \
        "Do not worry, it is fine and we will go. Cannot wait! Let us eat."


def test_subtitles_in_reply(client):
    sign = client.post("/api/chat/text", json={"text": "Hello! I am good, thank you. And you?"}).json()["reply"]["sign"]
    cues = sign["subtitles"]
    assert [c["text"] for c in cues] == ["Hello!", "I am good, thank you.", "And you?"]
    # the cues cover the signing, between the avatar easing out of its rest pose and back into it
    assert cues[0]["start"] == round(sign["lead_in"] / sign["fps"], 3)
    assert cues[-1]["end"] == round((sign["frames"] - sign["lead_out"]) / sign["fps"], 3)
    assert all(a["end"] == b["start"] for a, b in zip(cues, cues[1:]))          # no gaps
    vtt = client.get(sign["subtitle_url"]).text
    assert vtt.startswith("WEBVTT") and "I am good, thank you." in vtt
    assert client.get(sign["pose_url"]).json()["subtitles"] == cues


def test_subtitle_cues():
    from signchat.subtitles import cues, per_frame, split_text, to_vtt
    assert split_text("one two three four five six seven eight nine.", 7) == ["one two three four five", "six seven eight nine."]
    assert split_text("I went home, then I had a very long nap today.", 7) == ["I went home,", "then I had a", "very long nap today."]
    c = cues("Hi. See you tomorrow.", frames=50, fps=25)
    assert c == [{"start": 0.0, "end": 0.5, "text": "Hi."}, {"start": 0.5, "end": 2.0, "text": "See you tomorrow."}]
    frames = per_frame(c, 50, 25)
    assert frames[0] == "Hi." and frames[11] == "Hi." and frames[13] == "See you tomorrow." and frames[49] == "See you tomorrow."
    assert "00:00:00.500 --> 00:00:02.000" in to_vtt(c)
    assert cues("", 50, 25) == []


def test_chat_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "/api/chat/sign" in r.text and "/api/chat/text" in r.text
    js = client.get("/static/avatar.js")
    assert js.status_code == 200 and "export class SignAvatar" in js.text


def test_avatar_rig(client):
    rig = client.get("/api/avatar").json()
    assert rig["format"] == "stage-v1" and rig["fps"] == 25
    assert len(rig["rest"]) == 127 and len(rig["idle"]) == 100 and rig["idle"][0] == rig["rest"]
    assert all(0 <= b["a"] < 127 and 0 <= b["b"] < 127 and b["color"].startswith("#") for b in rig["bones"])
    rest = rig["rest"]
    drawn = {j for b in rig["bones"] for j in (b["a"], b["b"])} | set(rig["points"]["joints"])
    assert all(0 <= rest[j][0] <= 1 and 0 <= rest[j][1] <= 1 for j in drawn)       # all of it in the picture
    # hands down: both wrists (20, 21) below both shoulders (16, 17); y grows downwards
    assert min(rest[20][1], rest[21][1]) > max(rest[16][1], rest[17][1]) + 0.2


def test_reply_starts_and_ends_at_rest(client):
    rest = client.get("/api/avatar").json()["rest"]
    sign = client.post("/api/chat/text", json={"text": "See you tomorrow."}).json()["reply"]["sign"]
    pose = client.get(sign["pose_url"]).json()
    P = pose["joints2d"]
    assert len(P) == pose["frames"] == sign["frames"] and len(P[0]) == 127
    assert pose["lead_in"] == sign["lead_in"] > 0 and pose["lead_out"] == sign["lead_out"] > 0
    far = lambda a, b: max(abs(p[0] - q[0]) + abs(p[1] - q[1]) for p, q in zip(a, b))
    lead_in, lead_out = pose["lead_in"], pose["lead_out"]
    assert far(P[0], rest) < 0.1 and far(P[-1], rest) < 0.1                       # one eased step from rest
    assert far(P[0], rest) < far(P[lead_in], rest) and far(P[-1], rest) < far(P[-1 - lead_out], rest)
    assert far(P[len(P) // 2], rest) > 0.1                                          # and it does move in between
