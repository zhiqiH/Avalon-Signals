"""One public site for the human study, served on a single port (and domain).

    python scripts/study_site.py                      # http://127.0.0.1:8770
    python scripts/study_site.py --no-browser         # behind a tunnel that forwards the domain to this port

    /            home page: say whether data may be shared, then choose a mode
    /play/       human vs. five frozen silent-baseline players (scripts/play_web.py)
    /advisor/    assassin advisor for offline human games (scripts/assassin_advisor.py)

The share choice made on the home page is remembered in the browser and prefilled in both modes,
where it can still be changed per game. Only shared games are written to the research data:
completed /play/ games go to --save-games (JSONL, tagged "entry": "site"); advisor records are kept
per visitor and only shared ones enter the pooled statistics and `assassin_advisor.py --export-shared`.

An experiment interface only: it reuses the two scripts above and never touches training code or runs.
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

from scripts.assassin_advisor import SAVE_DIR, SEEDS, Store, advisor_api, code_cookie, load_assassins, parse_code
from scripts.play_web import GameLog, TableSessions

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
STATIC = {
    "/": (WEB / "site" / "index.html", "text/html"),
    "/home.js": (WEB / "site" / "home.js", "text/javascript"),
    "/ui.css": (WEB / "site" / "ui.css", "text/css"),  # one design system and one language file for all pages
    "/i18n.js": (WEB / "site" / "i18n.js", "text/javascript"),
    "/play/": (WEB / "human" / "index.html", "text/html"),
    "/play/app.js": (WEB / "human" / "app.js", "text/javascript"),
    "/advisor/": (WEB / "site" / "advisor.html", "text/html"),
    "/advisor/advisor.js": (WEB / "site" / "advisor.js", "text/javascript"),
}
STRICT_CSP = ("default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; "
              "frame-ancestors 'none'")
PLAY_COOKIE = "avalon_site_session"
PLAY_GET, PLAY_POST = ("/api/state", "/api/replay"), ("/api/new", "/api/action", "/api/advance")
MAX_BODY = {"/play": 8192, "/advisor": 200_000}


class SiteLog(GameLog):
    """The table's consent-gated game log, with records marked as coming from the combined site."""

    def write(self, record):
        super().write({**record, "entry": "site"})


def handler_for(sessions: TableSessions, store: Store):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AvalonSite/1"

        def log_message(self, fmt, *args):
            pass

        def secure(self) -> bool:
            return self.headers.get("X-Forwarded-Proto") == "https"

        def respond(self, body, status=200, mime="application/json; charset=utf-8", cookies=(), csp=STRICT_CSP,
                    location=None):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "same-origin")
            self.send_header("Content-Security-Policy", csp)
            if location:
                self.send_header("Location", location)
            for cookie in cookies:
                self.send_header("Set-Cookie", cookie)
            self.end_headers()
            self.wfile.write(body)

        def play(self, path, data=None):
            cookies = SimpleCookie()
            try:
                cookies.load(self.headers.get("Cookie", ""))
            except CookieError:
                cookies = SimpleCookie()
            morsel = cookies.get(PLAY_COOKIE)
            try:
                result, session = sessions.request(morsel.value if morsel else None, path, data)
            except ValueError as exc:  # includes SessionExpired
                return self.respond({"error": str(exc)}, 400)
            except RuntimeError as exc:
                return self.respond({"error": str(exc)}, 503)
            except OSError:
                return self.respond({"error": "模型文件无法读取。"}, 500)
            cookie = [f"{PLAY_COOKIE}={session}; Path=/; HttpOnly; SameSite=Strict" + ("; Secure" if self.secure() else "")]
            self.respond(result, cookies=cookie if session else ())

        def advisor(self, method, path, query, data=None):
            payload, status, set_code = advisor_api(store, method, path, query, data,
                                                    parse_code(self.headers.get("Cookie", "")))
            self.respond(payload, status, cookies=[code_cookie(set_code, self.secure())] if set_code else ())

        def do_GET(self):
            url = urlsplit(self.path)
            path = url.path
            if path == "/api/health":
                return self.respond({"service": "avalon-site", "ready": True})
            if path in ("/play", "/advisor"):
                return self.respond(b"", 301, "text/plain", location=path + "/")
            if path.startswith("/play/api/") and path[5:] in PLAY_GET:
                return self.play(path[5:])
            if path.startswith("/advisor/api/"):
                return self.advisor("GET", path[8:], url.query)
            if path in STATIC:
                file, mime = STATIC[path]
                return self.respond(file.read_bytes(), mime=mime + "; charset=utf-8")
            self.respond({"error": "未找到"}, 404)

        def do_POST(self):
            path = urlsplit(self.path).path
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                return self.respond({"error": "仅支持同源访问。"}, 403)
            if self.headers.get_content_type() != "application/json":
                return self.respond({"error": "需要 JSON 请求。"}, 415)
            prefix = "/play" if path.startswith("/play/api/") else "/advisor" if path.startswith("/advisor/api/") else None
            if prefix is None or (prefix == "/play" and path[5:] not in PLAY_POST):
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
            if prefix == "/play":
                return self.play(path[5:], data)
            self.advisor("POST", path[8:], "", data)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--lan", action="store_true", help="监听 0.0.0.0，让同一局域网的设备也能打开")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--baseline-seed", type=int, choices=SEEDS, default=0, help="人机对局默认的对手模型")
    parser.add_argument("--advisor-seeds", type=int, nargs="+", choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--save-games", type=Path, default=ROOT / "logs" / "human_games.jsonl",
                        help="同意共享的人机对局追加到此 JSONL")
    parser.add_argument("--advisor-dir", type=Path, default=SAVE_DIR, help="刺客辅助的对局记录目录")
    args = parser.parse_args()
    torch.set_num_threads(1)
    sessions = TableSessions(args.baseline_seed, log=SiteLog(args.save_games))
    store = Store(args.advisor_dir, load_assassins(args.advisor_seeds))
    store.adopt_orphans(args.advisor_dir.with_name(args.advisor_dir.name + "_legacy_code.txt"))
    try:
        server = ThreadingHTTPServer(("0.0.0.0" if args.lan else "127.0.0.1", args.port), handler_for(sessions, store))
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            parser.error(f"端口 {args.port} 已被占用，可改用 --port {args.port + 1}。")
        raise

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Avalon 真人实验站点：{url}\n  人机对局 {url}/play/ · 刺客辅助 {url}/advisor/\n"
          f"  共享的人机对局 → {args.save_games.resolve()}\n  刺客辅助记录 → {args.advisor_dir.resolve()}\n按 Ctrl+C 关闭。",
          flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已关闭。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
