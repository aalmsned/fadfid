"""
==============================================================================
 Fadfid web demo server
==============================================================================
 Serves a browser chat page and a small JSON API around the seven-agent
 pipeline. The model runs on this machine; other devices (e.g. your laptop
 over Tailscale) only send and receive chat text.

 Usage (inside WSL, in the Fadfid folder):
   export FADFID_BACKEND=fanar FADFID_4BIT=0
   python webapp/server.py --host 0.0.0.0 --port 8000

 Then open http://localhost:8000 on this PC, or
 http://<this PC's Tailscale name>:8000 on the laptop.

 Public link for anyone (Tailscale Funnel):
   python webapp/server.py --port 8000 --public
   sudo tailscale funnel --bg 8000
 User view: /        Back-end view (agent and tier panel): /backend
 --public says honestly that no real counsellor is connected, and limits how
 many messages one visitor can send per minute.

 Nothing is written to disk except the text-free escalation and
 intervention logs (P3). Sessions live in memory and are deleted on
 "End session" or after 30 minutes idle.
==============================================================================
"""

import argparse
import json
import logging
import sys
import os
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fadfid.config import Config  # noqa: E402
from fadfid.pipeline import Fadfid  # noqa: E402

log = logging.getLogger("fadfid.web")
PAGE = Path(__file__).with_name("index.html")
STATIC = Path(__file__).with_name("static")
MIME = {".woff2": "font/woff2", ".png": "image/png", ".svg": "image/svg+xml", ".css": "text/css"}


RATE_PER_MINUTE = 12
MAX_SESSIONS = 200


class FadfidApp:
    def __init__(self, cfg: Config, public: bool = False):
        self.bot = Fadfid(cfg)
        self.backend = cfg.backend
        self.public = public
        self.lock = threading.Lock()  # one GPU: handle one message at a time
        self.recent = defaultdict(deque)
        self.rateLock = threading.Lock()

    def allow(self, client: str) -> bool:
        """At most RATE_PER_MINUTE messages per visitor per minute (public mode)."""
        if not self.public:
            return True
        now = time.time()
        with self.rateLock:
            q = self.recent[client]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= RATE_PER_MINUTE:
                return False
            q.append(now)
            return True

    def start(self, body: dict) -> dict:
        if len(self.bot.tracker.sessions) >= MAX_SESSIONS:
            self.bot.tracker.expireIdle()
            if len(self.bot.tracker.sessions) >= MAX_SESSIONS:
                raise ValueError("Fadfid is busy right now. Please try again in a few minutes.")
        sessionId = self.bot.startSession(culturalOptIn=bool(body.get("culturalOptIn")))
        if body.get("adult") is False:
            state = self.bot.tracker.get(sessionId)
            state.ageCue = True
            state.minorNoticeShown = True  # the page shows the notice itself
        return {"sessionId": sessionId, "model": "Fanar-1-9B-Instruct" if self.backend == "fanar" else "MOCK",
                "prototype": self.public}

    def message(self, body: dict) -> dict:
        sessionId, text = body.get("sessionId"), (body.get("text") or "").strip()
        if not sessionId or not text:
            raise ValueError("sessionId and text are required")
        if body.get("culturalOptIn") is not None:
            self.bot.tracker.get(sessionId).culturalOptIn = bool(body["culturalOptIn"])
        with self.lock:
            result = self.bot.handleMessage(sessionId, text[:2000])
        return {k: result.get(k) for k in ("reply", "tier", "harm", "variety", "route", "oversight", "signals")}

    def end(self, body: dict) -> dict:
        self.bot.endSession(body.get("sessionId", ""))
        return {"ended": True}


def makeHandler(app: FadfidApp):
    routes = {"/api/start": app.start, "/api/message": app.message, "/api/end": app.end}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # Request lines only; message text is never logged.
            log.debug("%s %s", self.address_string(), fmt % args)

        def sendJson(self, status: int, payload: dict):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.split("?")[0] in ("/", "/index.html", "/backend"):
                data = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path.startswith("/static/"):
                target = (STATIC / self.path[len("/static/"):]).resolve()
                if STATIC.resolve() not in target.parents or not target.is_file():
                    return self.sendJson(404, {"error": "not found"})
                data = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", MIME.get(target.suffix, "application/octet-stream"))
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "max-age=86400")
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/api/health":
                self.sendJson(200, {"ok": True, "model": app.backend})
            else:
                self.sendJson(404, {"error": "not found"})

        def do_POST(self):
            handler = routes.get(self.path)
            if handler is None:
                return self.sendJson(404, {"error": "not found"})
            client = (self.headers.get("X-Forwarded-For") or self.client_address[0]).split(",")[0].strip()
            if self.path == "/api/message" and not app.allow(client):
                return self.sendJson(429, {"error": "You're sending messages very fast. Please wait a moment."})
            try:
                length = min(int(self.headers.get("Content-Length", 0)), 20000)
                body = json.loads(self.rfile.read(length) or b"{}")
                self.sendJson(200, handler(body))
            except ValueError as exc:
                self.sendJson(400, {"error": str(exc)})
            except Exception:
                log.exception("Request failed")
                self.sendJson(500, {"error": "internal error"})

    return Handler


def buildArgs():
    parser = argparse.ArgumentParser(description="Run the Fadfid web demo.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--public", action="store_true", help="Public demo: prototype notices and rate limits.")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = buildArgs()
    if args.public:
        os.environ["FADFID_PROTOTYPE"] = "1"
    cfg = Config()
    log.info("Loading Fadfid (backend=%s, 4-bit=%s)...", cfg.backend, cfg.load4bit)
    app = FadfidApp(cfg, public=args.public)
    server = ThreadingHTTPServer((args.host, args.port), makeHandler(app))
    log.info("Fadfid is running: http://localhost:%d  (Ctrl+C to stop)", args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Stopped.")


if __name__ == "__main__":
    main()
