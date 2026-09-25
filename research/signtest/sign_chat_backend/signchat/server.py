"""HTTP API for the chat frontend (FastAPI).

    GET    /api/health                  which models are loaded
    POST   /api/chat/text               {"session_id"?, "text"}               -> chat turn
    POST   /api/chat/sign               multipart: video, session_id?, mirrored? -> chat turn
    POST   /api/translate/sign-to-text  multipart: video, mirrored?           -> recognised text only
    POST   /api/translate/text-to-sign  {"text"}                              -> signing only (no dialogue)
    DELETE /api/session/{session_id}    forget a conversation
    GET    /media/<file>                generated pose JSON / mp4

Run:  uvicorn signchat.server:app --host 0.0.0.0 --port 8000     (config from SIGNCHAT_CONFIG)
"""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import load_config
from .pipeline import ChatPipeline

VIDEO_TYPES = (".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v")


class TextTurn(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    session_id: str | None = None


class TextOnly(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)


def create_app(cfg: dict | None = None, pipeline: ChatPipeline | None = None) -> FastAPI:
    cfg = cfg or load_config()
    state: dict = {"pipeline": pipeline}

    @asynccontextmanager
    async def lifespan(app):
        if state["pipeline"] is None:
            state["pipeline"] = ChatPipeline(cfg)          # loads every model once, before serving
        yield

    app = FastAPI(title="Auslan sign chat backend", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=cfg["cors_origins"], allow_methods=["*"], allow_headers=["*"])
    os.makedirs(cfg["media_dir"], exist_ok=True)
    app.mount("/media", StaticFiles(directory=cfg["media_dir"]), name="media")

    def pipe() -> ChatPipeline:
        return state["pipeline"]

    def url(name: str | None) -> str | None:
        return f"{cfg['public_base_url'].rstrip('/')}/media/{name}" if name else None

    def with_urls(sign: dict) -> dict:
        out = dict(sign)
        out["pose_url"] = url(out.pop("pose_file", None))
        out["video_url"] = url(out.pop("video_file", None))
        return out

    def run(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except ValueError as e:              # bad input: empty text, no signer in the video, ...
            raise HTTPException(422, str(e))
        except RuntimeError as e:            # a model that did not load
            raise HTTPException(503, str(e))

    def save_upload(video: UploadFile) -> str:
        ext = os.path.splitext(video.filename or "")[1].lower() or ".mp4"
        if ext not in VIDEO_TYPES:
            raise HTTPException(415, f"unsupported video type {ext}; use one of {', '.join(VIDEO_TYPES)}")
        fd, path = tempfile.mkstemp(suffix=ext)
        limit = cfg["max_upload_mb"] << 20
        with os.fdopen(fd, "wb") as fh:
            shutil.copyfileobj(video.file, fh)
            size = fh.tell()
        if size > limit:
            os.remove(path)
            raise HTTPException(413, f"video larger than {cfg['max_upload_mb']} MB")
        if size == 0:
            os.remove(path)
            raise HTTPException(422, "empty video")
        return path

    @app.get("/api/health")
    def health():
        return pipe().status()

    @app.post("/api/chat/text")
    def chat_text(body: TextTurn):
        r = run(pipe().turn, body.session_id, text=body.text)
        r["reply"]["sign"] = with_urls(r["reply"]["sign"])
        return r

    @app.post("/api/chat/sign")
    def chat_sign(video: UploadFile = File(...), session_id: str | None = Form(None), mirrored: bool = Form(False)):
        path = save_upload(video)
        try:
            r = run(pipe().turn, session_id, video_path=path, mirrored=mirrored)
        finally:
            os.remove(path)
        r["reply"]["sign"] = with_urls(r["reply"]["sign"])
        return r

    @app.post("/api/translate/sign-to-text")
    def sign_to_text(video: UploadFile = File(...), mirrored: bool = Form(False)):
        path = save_upload(video)
        try:
            return run(pipe().sign_to_text, path, mirrored)
        finally:
            os.remove(path)

    @app.post("/api/translate/text-to-sign")
    def text_to_sign(body: TextOnly):
        return with_urls(run(pipe().text_to_sign, body.text.strip()))

    @app.delete("/api/session/{session_id}")
    def forget(session_id: str):
        return {"session_id": session_id, "cleared": pipe().sessions.clear(session_id)}

    return app


app = create_app() if os.environ.get("SIGNCHAT_NO_AUTOAPP") != "1" else None
