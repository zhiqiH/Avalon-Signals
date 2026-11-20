"""Browser table for human playtests: one human and five frozen silent-baseline players.

    python scripts/play_web.py                    # local, http://127.0.0.1:8765
    python scripts/play_web.py --share --no-browser   # temporary public HTTPS link (cloudflared)

Visitors choose on the setup screen whether to share their games. Completed games of visitors
who agreed are appended to --save-games (JSONL, one game per line) for later analysis.
Uses only the existing Python dependencies and the standard-library HTTP server.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import secrets
import signal
import socket
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.agents import PolicyController, Request
from src.env import actions as A
from src.env.game import AvalonGame
from src.env.rules import QUEST_SIZES, Phase, Role, WinReason
from src.training.checkpoint import load_policies

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "web" / "human"
SHARED = ROOT / "web" / "site"  # design system and language file shared with the study site
CONSENT_VERSION = 2  # Bump when the consent text in web/human/index.html changes (2: shortened, 2026-11-20).
ROLE_NAMES = {Role.SERVANT: "忠臣", Role.MERLIN: "梅林", Role.PERCIVAL: "派西维尔",
              Role.MORGANA: "莫甘娜", Role.ASSASSIN: "刺客"}
REASONS = {WinReason.ASSASSIN_MISSED: "刺杀未命中梅林", WinReason.ASSASSIN_HIT: "刺客命中梅林",
           WinReason.THREE_FAILS: "三次任务失败", WinReason.FIVE_REJECTIONS: "连续五次组队被否决"}


# Language-neutral option kinds for the browser (the labels above stay for older clients and logs).
OPTION_KINDS = {A.APPROVE: "approve", A.REJECT: "reject", A.SUCCESS: "success", A.FAIL: "fail"}


def seats(values) -> str:
    """The UI uses absolute seats 1–6; the environment uses 0–5."""
    return "、".join(str(int(s) + 1) for s in values) or "无"


def new_game(seed: int, human: int, role: str) -> AvalonGame:
    """A random game; with a requested role, the human swaps roles with whoever drew it."""
    rng = np.random.default_rng([seed, 0])
    game = AvalonGame(rng)
    if role != "random":
        source = int(rng.choice(np.flatnonzero(game.roles == Role[role.upper()])))
        roles = game.roles.copy()  # A new array also invalidates the private-knowledge cache.
        roles[human], roles[source] = roles[source], roles[human]
        game.roles = roles
    return game


def action_options(game: AvalonGame, human: int) -> list[tuple[int, str]]:
    options = []
    labels = {A.APPROVE: "赞成", A.REJECT: "反对", A.SUCCESS: "任务成功", A.FAIL: "任务失败"}
    for raw in np.flatnonzero(game.legal_mask(human)):
        action = int(raw)
        if game.phase == Phase.PROPOSE:
            label = f"组队 [{seats(A.decode_team(human, action, game.team_size))}]"
        elif game.phase == Phase.ASSASSINATE:
            label = f"刺杀 {A.to_absolute(human, action - A.ASSASSIN_OFFSET) + 1} 号"
        else:
            label = labels[action]
        options.append((action, label))
    return options


class GameLog:
    """Append-only JSONL of completed games whose players agreed to share them."""

    def __init__(self, path):
        self.path = Path(path) if path else None

    def write(self, record):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{record['finished_utc']}] 已保存共享对局：访客 {record['player']} 第 {record['player_game']} 局，"
              f"{record['human_role']}，{'胜' if record['human_won'] else '负'}", flush=True)


class HumanTable:
    """One visitor's table; only a filtered human view crosses the HTTP boundary."""

    def __init__(self, default_baseline=0, models=None, log=None, clock=time.time):
        self.default_baseline = default_baseline
        self.game = None
        self.models = models if models is not None else {}
        self.log = log if log is not None else GameLog(None)
        self.clock = clock
        self.share = None  # Last consent choice of this visitor; None until asked.
        self.player = secrets.token_hex(8)  # Pseudonymous visitor id, never the session cookie.
        self.games = 0

    def start(self, data):
        baseline = data.get("baseline", self.default_baseline)
        seat = data.get("seat", 1)
        role = data.get("role", "random")
        share = data.get("share")
        if type(baseline) is not int or baseline not in (0, 1, 2):
            raise ValueError("请选择 baseline 0、1 或 2。")
        if type(seat) is not int or seat not in range(1, 7):
            raise ValueError("座位必须为 1–6。")
        if role not in ["random", *[r.name.lower() for r in Role]]:
            raise ValueError("未知角色。")
        if type(share) is not bool:
            raise ValueError("请选择是否共享本局数据。")
        if baseline not in self.models:
            path = ROOT / f"baseline/seed{baseline}/checkpoint_19532.pt"
            nets = load_policies(path)
            if any(net.comm for net in nets.values()):
                raise ValueError("仅支持 silent baseline。")
            self.models[baseline] = {s: PolicyController(n, learn=False) for s, n in nets.items()}
            self.models[baseline]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.baseline, self.human, self.role, self.share = baseline, seat - 1, role, share
        self.seed = secrets.randbits(32)
        self.game = new_game(self.seed, self.human, role)
        self.rng = np.random.default_rng([self.seed, 1_000_003])
        self.id, self.revision = secrets.token_hex(12), 0
        self.started = self.clock()
        self.games += 1
        self.event = {"kind": "start", "text": "身份已分配。请先阅读左侧的私人信息，再开始行动。"}
        self.decisions = []
        return self.view()

    def step(self, data, advance=False):
        game = self.game
        if game is None or game.done:
            raise ValueError("请先开始一局新游戏。")
        if data.get("id") != self.id or data.get("revision") != self.revision:
            raise ValueError("牌桌已更新，请刷新后再操作。")
        acting = game.acting_players()
        chosen = {}
        if self.human in acting:
            if advance:
                raise ValueError("请先选择你的行动。")
            action = data.get("action")
            if type(action) is not int or action not in [a for a, _ in action_options(game, self.human)]:
                raise ValueError("当前行动不合法。")
            chosen[self.human] = action
        elif not advance:
            raise ValueError("当前请点击继续，等待 AI 行动。")
        phase, q, attempt = game.phase, game.quest, game.attempt
        leader = game.leader
        # All players act on the unchanged state; no pending AI actions are sent to the UI.
        for side in ("good", "evil"):
            requests = []
            for p in acting:
                if p == self.human or game.is_evil_seat(p) != (side == "evil"):
                    continue
                mask = game.legal_mask(p)
                legal = np.flatnonzero(mask)
                if len(legal) == 1:
                    chosen[p] = int(legal[0])
                else:
                    requests.append(Request(0, game, p, mask))
            if requests:
                result = self.models[self.baseline][side].act(requests, self.rng)
                chosen.update({r.player: int(a) for r, a in zip(requests, result.actions)})
        game.step(chosen)
        if self.human in chosen:
            self.decisions.append({"phase": phase.name, "quest": q, "attempt": attempt,
                                   "action": chosen[self.human],
                                   "seconds": round(self.clock() - self.started, 1)})
        if phase == Phase.PROPOSE:
            team = [int(p) + 1 for p in np.flatnonzero(game.team)]
            self.event = {"kind": "proposal", "leader": leader + 1, "team": team,
                          "text": f"{leader + 1} 号提议让 {'、'.join(map(str, team))} 号执行任务。"}
        elif phase == Phase.VOTE:
            votes = game.prop_votes[q, attempt].tolist()
            passed = bool(game.prop_approved[q, attempt])
            self.event = {"kind": "vote", "votes": votes, "passed": passed,
                          "text": f"投票公布：{sum(votes)} 票赞成、{6 - sum(votes)} 票反对，组队{'通过' if passed else '被否决'}。"}
        elif phase == Phase.QUEST:
            fails = int(game.quest_fails[q])
            self.event = {"kind": "quest", "fails": fails, "quest": q,
                          "text": f"第 {q + 1} 次任务{'失败' if fails else '成功'}，共 {fails} 张失败牌。"}
        else:
            self.event = {"kind": "assassination", "target": game.assassin_target + 1,
                          "text": f"刺杀目标为 {game.assassin_target + 1} 号。"}
        self.revision += 1
        if game.done and self.share:
            self.log.write(self.record())
        return self.view()

    def record(self):
        """Full post-game record (seats 0–5, hidden roles included); no IP or browser data."""
        g = self.game
        return {"kind": "human_vs_baseline", "consent_version": CONSENT_VERSION,
                "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "player": self.player, "player_game": self.games, "table_id": self.id,
                "baseline_seed": self.baseline, "checkpoint_sha256": self.models[self.baseline].get("sha256"),
                "seed": self.seed, "human_seat": self.human, "requested_role": self.role,
                "human_role": Role(int(g.roles[self.human])).name,
                "human_won": bool(g.good_won) != g.is_evil_seat(self.human),
                "seconds": round(self.clock() - self.started, 1),
                "human_decisions": self.decisions, "game": g.to_record()}

    def view(self):
        if self.game is None:
            return {"active": False, "default_baseline": self.default_baseline, "share": self.share}
        g, h = self.game, self.human
        evil, candidates = g.private_knowledge(h)
        human_turn = not g.done and h in g.acting_players()
        options = []
        if human_turn:
            for action, label in action_options(g, h):
                option = {"action": action, "label": label, "kind": OPTION_KINDS.get(action, "team" if g.phase == Phase.PROPOSE else "target")}
                if g.phase == Phase.PROPOSE:
                    option["team"] = [s + 1 for s in A.decode_team(h, action, g.team_size)]
                elif g.phase == Phase.ASSASSINATE:
                    option["target"] = A.to_absolute(h, action - A.ASSASSIN_OFFSET) + 1
                options.append(option)
        history = []
        for q in range(5):
            for a in range(5):
                if g.prop_leader[q, a] < 0:
                    continue
                history.append({"quest": q + 1, "attempt": a + 1,
                                "leader": int(g.prop_leader[q, a]) + 1,
                                "team": [int(s) + 1 for s in np.flatnonzero(g.prop_team[q, a])],
                                "votes": g.prop_votes[q, a].tolist() if g.prop_voted[q, a] else None,
                                "approved": bool(g.prop_approved[q, a]) if g.prop_voted[q, a] else None})
        state = {
            "active": True, "id": self.id, "revision": self.revision, "baseline": self.baseline,
            "human": h + 1, "role": ROLE_NAMES[Role(int(g.roles[h]))], "role_key": Role(int(g.roles[h])).name,
            "side": "evil" if g.is_evil_seat(h) else "good",
            "known_evil": [int(s) + 1 for s in np.flatnonzero(evil)],
            "candidates": [int(s) + 1 for s in np.flatnonzero(candidates)],
            "own_cards": g.quest_cards[h].tolist(), "phase": g.phase.name,
            "quest": min(g.quest + 1, 5), "attempt": min(g.attempt + 1, 5),
            "leader": int(g.prop_leader[g.quest, g.attempt]) + 1 if g.phase == Phase.QUEST else g.leader + 1,
            "team": [int(s) + 1 for s in np.flatnonzero(g.team)],
            "team_size": g.team_size if g.phase in (Phase.PROPOSE, Phase.VOTE, Phase.QUEST) else 0,
            "successes": g.successes, "fails": g.fails,
            "quests": [{"size": size, "result": int(g.quest_result[q]),
                        "fails": int(g.quest_fails[q]) if g.quest_result[q] >= 0 else None}
                       for q, size in enumerate(QUEST_SIZES)],
            "human_turn": human_turn, "options": options, "history": history,
            "event": self.event, "done": g.done, "share": self.share,
        }
        if g.done:
            state["result"] = {"good_won": bool(g.good_won), "human_won": bool(g.good_won) != g.is_evil_seat(h),
                               "reason": REASONS[g.win_reason], "target": g.assassin_target + 1,
                               "roles": [ROLE_NAMES[Role(int(r))] for r in g.roles],
                               "reason_key": g.win_reason.name, "role_keys": [Role(int(r)).name for r in g.roles]}
        return state

    def replay(self):
        if self.game is None or not self.game.done:
            raise ValueError("对局结束后才能下载完整复盘。")
        return self.record()


class SessionExpired(ValueError):
    pass


class TableSessions:
    """Independent browser sessions sharing frozen model weights, never game state."""

    def __init__(self, default_baseline=0, ttl=7200, capacity=128, idle_evict=600, log=None,
                 clock=time.monotonic):
        self.default_baseline, self.ttl, self.capacity, self.idle_evict = default_baseline, ttl, capacity, idle_evict
        self.log = log if log is not None else GameLog(None)
        self.clock = clock
        self.tables = {}
        self.models = {}
        # Serializes state changes, inference and log writes, including duplicate requests from one browser.
        self.lock = threading.Lock()

    def request(self, session_id, path, data=None):
        with self.lock:
            now = self.clock()
            for key in list(self.tables):
                if now - self.tables[key][1] >= self.ttl:
                    del self.tables[key]
            created = False
            if session_id not in self.tables:
                # Only starting a game creates a table, so cookie-less page loads and crawlers cost nothing.
                if path == "/api/state":
                    return {"active": False, "default_baseline": self.default_baseline, "share": None}, None
                if path != "/api/new":
                    raise SessionExpired("对局会话已过期，请刷新页面重新开局。")
                if len(self.tables) >= self.capacity:
                    oldest = min(self.tables, key=lambda k: self.tables[k][1])
                    if now - self.tables[oldest][1] < self.idle_evict:
                        raise RuntimeError("当前牌桌人数已满，请稍后再试。")
                    del self.tables[oldest]
                session_id, created = secrets.token_urlsafe(32), True
                self.tables[session_id] = (HumanTable(self.default_baseline, self.models, self.log), now)
            table, _ = self.tables[session_id]
            self.tables[session_id] = (table, now)
            try:
                if path == "/api/state":
                    result = table.view()
                elif path == "/api/new":
                    result = table.start(data)
                elif path == "/api/replay":
                    result = table.replay()
                elif path in ("/api/action", "/api/advance"):
                    result = table.step(data, advance=path.endswith("advance"))
                else:
                    raise ValueError("未知接口。")
            except Exception:
                if created:
                    del self.tables[session_id]
                raise
            return result, session_id


def handler_for(sessions, extra_hosts=(), public_origin=None):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, data, status=200, mime="application/json; charset=utf-8"):
            body = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if getattr(self, "session_id", None):
                secure = "; Secure" if self.public_origin and self.public_origin.startswith("https://") and (
                    self.headers.get("Host") == urlsplit(self.public_origin).netloc
                    or self.headers.get("X-Forwarded-Proto") == "https") else ""
                self.send_header("Set-Cookie", f"avalon_session_{self.server.server_port}={self.session_id}; "
                                 f"Path=/; HttpOnly; SameSite=Strict{secure}")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def api_request(self, path, data=None):
            cookies = SimpleCookie()
            try:
                cookies.load(self.headers.get("Cookie", ""))
            except CookieError:
                cookies = SimpleCookie()
            cookie = cookies.get(f"avalon_session_{self.server.server_port}")
            result, self.session_id = sessions.request(cookie.value if cookie else None, path, data)
            self.respond(result)

        def same_origin(self):
            host = self.headers.get("Host", "")
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            allowed.update(f"{address}:{self.server.server_port}" for address in extra_hosts)
            if self.public_origin:
                allowed.add(urlsplit(self.public_origin).netloc)
            origin = self.headers.get("Origin")
            return host in allowed and (origin is None or origin == f"http://{host}" or origin == self.public_origin)

        def do_GET(self):
            if not self.same_origin():
                return self.respond({"error": "此地址未配置为牌桌入口。"}, 403)
            path = urlsplit(self.path).path
            if path == "/api/health":
                return self.respond({"service": "avalon-human-table", "ready": True})
            if path in ("/api/state", "/api/replay"):
                try:
                    return self.api_request(path)
                except ValueError as exc:
                    return self.respond({"error": str(exc)}, 400)
                except RuntimeError as exc:
                    return self.respond({"error": str(exc)}, 503)
            files = {"/": (ASSETS / "index.html", "text/html"), "/app.js": (ASSETS / "app.js", "text/javascript"),
                     "/ui.css": (SHARED / "ui.css", "text/css"), "/i18n.js": (SHARED / "i18n.js", "text/javascript")}
            if path not in files:
                return self.respond({"error": "Not found"}, 404)
            file, mime = files[path]
            self.respond(file.read_bytes(), mime=mime + "; charset=utf-8")

        def do_POST(self):
            if not self.same_origin():
                return self.respond({"error": "仅支持同源访问。"}, 403)
            if self.headers.get_content_type() != "application/json":
                return self.respond({"error": "需要 JSON 请求。"}, 415)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 8192:
                    raise ValueError("请求大小不合法。")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("请求必须为对象。")
                path = urlsplit(self.path).path
                if path in ("/api/new", "/api/action", "/api/advance"):
                    return self.api_request(path, data)
                return self.respond({"error": "Not found"}, 404)
            except (ValueError, UnicodeError) as exc:
                self.respond({"error": str(exc)}, 400)
            except OSError:
                self.respond({"error": "模型文件无法读取，请确认 baseline 文件已同步。"}, 500)
            except RuntimeError as exc:
                self.respond({"error": str(exc)}, 503)

        def log_message(self, format, *args):
            pass
    Handler.public_origin = public_origin
    return Handler


def lan_addresses():
    """Discover local IPv4 addresses; the UDP route lookup sends no data."""
    found = set()
    try:
        for entry in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(entry[4][0])
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 80))
            found.add(sock.getsockname()[0])
    except OSError:
        pass
    return sorted(address for address in found if not address.startswith("127.") and address != "0.0.0.0")


def public_url(value):
    parsed = urlsplit(value)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise argparse.ArgumentTypeError("请输入完整公网入口，例如 https://avalon.example.com，不要附带路径。")
    try:
        parsed.port
    except ValueError as exc:
        raise argparse.ArgumentTypeError("公网入口端口无效。") from exc
    return f"{parsed.scheme}://{parsed.netloc}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--baseline-seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--lan", action="store_true", help="允许同一局域网访问，并打印可分享的本机 IP 链接")
    parser.add_argument("--share", action="store_true", help="通过 cloudflared 自动生成公网 HTTPS 试玩链接")
    parser.add_argument("--public-url", type=public_url, help="已配置的公网反向代理/隧道入口；此选项本身不会创建隧道")
    parser.add_argument("--save-games", type=Path, default=ROOT / "logs" / "human_games.jsonl",
                        help="同意共享的访客，其完整对局追加到此 JSONL（默认 logs/human_games.jsonl）")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("端口必须在 1–65535 之间。")
    if args.share and args.public_url:
        parser.error("--share 自动分配公网链接，请勿与 --public-url 同时使用。")
    torch.set_num_threads(1)
    addresses = lan_addresses() if args.lan else []
    bind_host = "0.0.0.0" if args.lan else "127.0.0.1"
    if args.lan and not addresses:
        parser.error("未发现可分享的局域网 IPv4 地址，请先连接 Wi-Fi 或有线网络。")
    handler = handler_for(TableSessions(args.baseline_seed, log=GameLog(args.save_games)), addresses, args.public_url)
    try:
        server = ThreadingHTTPServer((bind_host, args.port), handler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            alternative = args.port + 1 if args.port < 65535 else 8765
            parser.error(f"端口 {args.port} 已被占用。若牌桌已经运行，可直接使用现有链接；"
                         f"如需重新启动，请先关闭旧服务，或改用其他端口，例如 --port {alternative}。")
        parser.error(f"无法启动牌桌：{exc}")
    url = f"http://127.0.0.1:{args.port}"
    tunnel = None

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        if args.share:
            from scripts.share_tunnel import QuickTunnel
            tunnel = QuickTunnel(url)
            print("正在创建公网 HTTPS 入口…", flush=True)
            handler.public_origin = tunnel.start()
            print("地址已分配，正在等待 DNS 与公网入口就绪…", flush=True)
            tunnel.wait_ready()
            print(f"公网分享：{handler.public_origin}", flush=True)
            print("任何拿到此链接的人都可以独立试玩。关闭服务后链接失效；重启会生成新地址。", flush=True)
        print(f"Avalon 真人牌桌：{url}\n按 Ctrl+C 关闭。", flush=True)
        for address in addresses:
            print(f"局域网分享：http://{address}:{args.port}", flush=True)
        if args.public_url:
            print(f"公网入口（需已有反向代理或隧道）：{args.public_url}", flush=True)
        print("每位访客独立一局；同一浏览器的多个标签页共用一局。", flush=True)
        print(f"同意共享的对局将追加到：{args.save_games.resolve()}", flush=True)
        if not args.no_browser:
            webbrowser.open(handler.public_origin if args.share else url)
        while server_thread.is_alive():
            server_thread.join(timeout=0.5)
            if tunnel is not None and tunnel.process.poll() is not None:
                raise RuntimeError("公网隧道已断开，请重新启动生成新链接。")
    except KeyboardInterrupt:
        print("\n牌桌已关闭。")
    except RuntimeError as exc:
        parser.exit(1, str(exc) + "\n")
    finally:
        if tunnel is not None:
            tunnel.close()
        server.shutdown()
        server_thread.join(timeout=5)
        server.server_close()


if __name__ == "__main__":
    main()
