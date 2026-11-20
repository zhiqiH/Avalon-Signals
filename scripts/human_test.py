"""Human test site: one server (one port, one domain) for both human-test modes.

    python scripts/human_test.py                     # http://127.0.0.1:8770
    python scripts/human_test.py --no-browser        # behind a tunnel that forwards the domain to this port
    python scripts/human_test.py --export-analysis OUT.jsonl   # shared analysis records, then exit

    /            home: consent to data use, then choose a mode
    /match/      human vs. AI: one human and five frozen silent-baseline players (scripts/human_test_match.py)
    /analysis/   assassination analysis of an offline human game (scripts/human_test_analysis.py)

The consent choice made on the home page is remembered in the browser and prefilled in both modes, where it
can still be changed per game. Only shared data is used for research: finished matches of consenting visitors
are appended to the match log; analysis records are kept per visitor and only shared ones enter the pooled
statistics and --export-analysis. Pages, styles and language strings live in web/human_test/.

An experiment interface only: it never touches training code or runs.
"""
from __future__ import annotations

import argparse
import errno
import json
import signal
import sys
import webbrowser
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scripts.human_test_analysis import (SAVE_DIR, SEEDS, Store, analysis_api, code_cookie, load_assassins,
                                         parse_code, public)
from scripts.human_test_match import GameLog, TableSessions

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "human_test"
LOG_DIR = ROOT / "logs" / "human_test"
PAGES = {"/": "home.html", "/match/": "match.html", "/analysis/": "analysis.html"}
ASSETS = {"/common.css": "text/css", "/i18n.js": "text/javascript", "/home.js": "text/javascript",
          "/match.js": "text/javascript", "/analysis.js": "text/javascript"}
REDIRECTS = {"/match": "/match/", "/analysis": "/analysis/", "/play": "/match/", "/play/": "/match/",
             "/advisor": "/analysis/", "/advisor/": "/analysis/"}  # the last four: links from before the rename
CSP = ("default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; "
       "frame-ancestors 'none'")
MATCH_COOKIE = "avalon_session"
MATCH_GET, MATCH_POST = ("/api/state", "/api/replay"), ("/api/new", "/api/action", "/api/advance")
MAX_BODY = {"/match": 8192, "/analysis": 200_000}


class MatchLog(GameLog):
    """The consent-gated match log; records carry "entry": "site" (games served by this site, not the old table)."""

    def write(self, record):
        super().write({**record, "entry": "site"})


def handler_for(sessions: TableSessions, store: Store):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AvalonHumanTest/1"

        def log_message(self, fmt, *args):
            pass

        def secure(self) -> bool:
            return self.headers.get("X-Forwarded-Proto") == "https"

        def respond(self, body, status=200, mime="application/json; charset=utf-8", cookies=(), location=None):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "same-origin")
            self.send_header("Content-Security-Policy", CSP)
            if location:
                self.send_header("Location", location)
            for cookie in cookies:
                self.send_header("Set-Cookie", cookie)
            self.end_headers()
            self.wfile.write(body)

        def match(self, path, data=None):
            cookies = SimpleCookie()
            try:
                cookies.load(self.headers.get("Cookie", ""))
            except CookieError:
                cookies = SimpleCookie()
            morsel = cookies.get(MATCH_COOKIE)
            try:
                result, session = sessions.request(morsel.value if morsel else None, path, data)
            except ValueError as exc:  # includes SessionExpired
                return self.respond({"error": str(exc)}, 400)
            except RuntimeError as exc:
                return self.respond({"error": str(exc)}, 503)
            except OSError:
                return self.respond({"error": "模型文件无法读取。"}, 500)
            cookie = f"{MATCH_COOKIE}={session}; Path=/; HttpOnly; SameSite=Strict" + ("; Secure" if self.secure() else "")
            self.respond(result, cookies=[cookie] if session else ())

        def analysis(self, method, path, query, data=None):
            payload, status, set_code = analysis_api(store, method, path, query, data,
                                                     parse_code(self.headers.get("Cookie", "")))
            self.respond(payload, status, cookies=[code_cookie(set_code, self.secure())] if set_code else ())

        def do_GET(self):
            url = urlsplit(self.path)
            path = url.path
            if path == "/api/health":
                return self.respond({"service": "avalon-human-test", "ready": True})
            if path in REDIRECTS:
                return self.respond(b"", 301, "text/plain", location=REDIRECTS[path])
            if path in PAGES:
                return self.respond((WEB / PAGES[path]).read_bytes(), mime="text/html; charset=utf-8")
            if path in ASSETS:
                return self.respond((WEB / path[1:]).read_bytes(), mime=ASSETS[path] + "; charset=utf-8")
            if path.startswith("/match/api/") and path[6:] in MATCH_GET:
                return self.match(path[6:])
            if path.startswith("/analysis/api/"):
                return self.analysis("GET", path[9:], url.query)
            self.respond({"error": "未找到"}, 404)

        def do_POST(self):
            path = urlsplit(self.path).path
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                return self.respond({"error": "仅支持同源访问。"}, 403)
            if self.headers.get_content_type() != "application/json":
                return self.respond({"error": "需要 JSON 请求。"}, 415)
            prefix = "/match" if path.startswith("/match/api/") else "/analysis" if path.startswith("/analysis/api/") else None
            if prefix is None or (prefix == "/match" and path[6:] not in MATCH_POST):
                return self.respond({"error": "未找到"}, 404)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY[prefix]:
                    raise ValueError("请求大小不合法。")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("请求必须为对象。")
            except (ValueError, UnicodeError) as exc:
                return self.respond({"error": str(exc)}, 400)
            if prefix == "/match":
                return self.match(path[6:], data)
            self.analysis("POST", path[9:], "", data)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--lan", action="store_true", help="listen on 0.0.0.0 so devices on the same network can connect")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--match-seed", type=int, choices=SEEDS, default=0, help="default opponent seed in match mode")
    parser.add_argument("--analysis-seeds", type=int, nargs="+", choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--match-log", type=Path, default=LOG_DIR / "match_games.jsonl",
                        help="shared matches are appended here (JSONL)")
    parser.add_argument("--analysis-dir", type=Path, default=SAVE_DIR, help="analysis records (one JSON file each)")
    parser.add_argument("--export-analysis", type=Path, metavar="OUT.jsonl",
                        help="write the shared analysis records (without record codes) to OUT.jsonl and exit")
    args = parser.parse_args()
    if args.export_analysis:
        shared = [public(g) for g in Store(args.analysis_dir, {}).all_games() if g.get("share") is True]
        args.export_analysis.parent.mkdir(parents=True, exist_ok=True)
        with args.export_analysis.open("w", encoding="utf-8") as f:
            for g in shared:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        print(f"Exported {len(shared)} shared analysis records to {args.export_analysis}")
        return
    torch.set_num_threads(1)
    sessions = TableSessions(args.match_seed, log=MatchLog(args.match_log))
    store = Store(args.analysis_dir, load_assassins(args.analysis_seeds))
    store.adopt_orphans(args.analysis_dir.with_name(args.analysis_dir.name + "_legacy_code.txt"))
    try:
        server = ThreadingHTTPServer(("0.0.0.0" if args.lan else "127.0.0.1", args.port), handler_for(sessions, store))
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            parser.error(f"port {args.port} is in use; try --port {args.port + 1}")
        raise

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Avalon Signals human test: {url}  (match {url}/match/, analysis {url}/analysis/)\n"
          f"  shared matches  -> {args.match_log.resolve()}\n  analysis records -> {args.analysis_dir.resolve()}\n"
          "Press Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
