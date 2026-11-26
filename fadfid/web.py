"""
Fadfid web server.

Runs on the PC that hosts Fanar. Any device on the private network (e.g. a
laptop over Tailscale) opens the chat in a browser; all model work stays on
this machine. Sessions live in memory only and are deleted on /api/end or
after the idle timeout, exactly as in the terminal version.

Start it from the project folder:
    uvicorn fadfid.web:app --host 0.0.0.0 --port 8000
"""

import logging
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import Config
from .pipeline import Fadfid

log = logging.getLogger(__name__)
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Fadfid", docs_url=None, redoc_url=None)
cfg = Config()
bot = Fadfid(cfg)
lock = threading.Lock()  # one GPU, one reply at a time


class StartRequest(BaseModel):
    culturalOptIn: bool = False
    adult: bool = True


class MessageRequest(BaseModel):
    sessionId: str
    text: str


class EndRequest(BaseModel):
    sessionId: str


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    return {"model": "Fanar-1-9B-Instruct" if cfg.backend == "fanar" else "mock",
            "fullPrecision": not cfg.load4bit}


@app.post("/api/start")
def start(req: StartRequest):
    sessionId = bot.startSession(culturalOptIn=req.culturalOptIn)
    if not req.adult:
        bot.tracker.get(sessionId).ageCue = True
    return {"sessionId": sessionId}


@app.post("/api/message")
def message(req: MessageRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Empty message")
    if req.sessionId not in bot.tracker.sessions:
        raise HTTPException(404, "Session ended or timed out")
    with lock:
        result = bot.handleMessage(req.sessionId, text[:2000])
    return result


@app.post("/api/optin")
def optIn(req: EndRequest):
    if req.sessionId in bot.tracker.sessions:
        bot.tracker.get(req.sessionId).culturalOptIn = True
    return {"ok": True}


@app.post("/api/end")
def end(req: EndRequest):
    bot.endSession(req.sessionId)
    return {"ok": True}
