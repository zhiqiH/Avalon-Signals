"""Assassin advisor for real (human) Avalon tables: the frozen silent-baseline Assassin names its Merlin guess.

    python scripts/assassin_advisor.py              # http://127.0.0.1:8766
    python scripts/assassin_advisor.py --lan        # also reachable from a phone on the same network

Record a live game in the browser: leader, team and votes of every proposal, then the number of Fail
cards of every quest. If good completes three quests, enter the Assassin's and Morgana's seats; the
script rebuilds the exact information state the trained Assassin would see at the assassination
(`observation.encode`) and reports its probability for every good seat, per baseline seed and averaged.
Afterwards the true Merlin (and what the human Assassin chose) can be added, and the history page
gives the AI hit rate over all recorded games.

Seats are numbered 1–6 in the order the leadership passes (the model assumes leader k is followed by k+1).
When both evil players were on a quest with a single Fail card, it is unknown who played it; the
advice then averages over both possibilities and lists each one.

Each visitor gets a record code (a cookie; it can be typed in on another device) and sees only their
own games. When a game ends the visitor chooses whether to share it; only shared games enter the
"all shared games" statistics and `--export-shared`.

Standalone: it only reads src/ and the frozen checkpoints, and writes one JSON file per game to
logs/assassin_advisor/.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import re
import secrets
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import product
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.env import actions as A
from src.env.game import AvalonGame
from src.env.observation import encode
from src.env.rules import APPROVALS_NEEDED, MAX_PROPOSALS, N_PLAYERS, QUEST_SIZES, Phase, Role
from src.training.checkpoint import load_policies

ROOT = Path(__file__).resolve().parents[1]
SAVE_DIR = ROOT / "logs" / "assassin_advisor"
SEEDS = (0, 1, 2)
ID_RE = re.compile(r"^[0-9a-f]{16}$")
COOKIE = "advisor_code"
SEAT_FIELDS = ("first_leader", "assassin", "morgana", "human_target", "actual_merlin")


def checkpoint(seed: int) -> Path:
    return ROOT / f"baseline/seed{seed}/checkpoint_19532.pt"


def load_assassins(seeds=SEEDS) -> dict[int, torch.nn.Module]:
    """The evil network of each frozen silent baseline (it plays both Morgana and the Assassin)."""
    nets = {}
    for seed in seeds:
        team_nets = load_policies(checkpoint(seed))
        if team_nets["evil"].comm:
            raise ValueError(f"seed {seed} 不是 silent baseline")
        nets[seed] = team_nets["evil"]
    return nets


# ---------------------------------------------------------------------------- record → game

def _seat(value, name: str) -> int:
    if type(value) is not int or not 1 <= value <= N_PLAYERS:
        raise ValueError(f"{name}必须是 1–6 号座位。")
    return value - 1


def _seats(values, name: str) -> list[int]:
    if not isinstance(values, list) or len(set(values)) != len(values):
        raise ValueError(f"{name}格式错误。")
    return sorted(_seat(v, name) for v in values)


def replay(record: dict, evil_fail: dict[int, str] | None = None) -> tuple[AvalonGame, list[int]]:
    """Rebuild the game from a record, using its Assassin and Morgana seats.

    `evil_fail[q]` says who played the Fail card ("assassin" or "morgana") on quest q when both evil
    players were on it and it had exactly one Fail card; unspecified quests default to the Assassin.
    Returns the game (stopped wherever the record ends) and the list of such ambiguous quests.
    Good roles are placed arbitrarily: the Assassin's observation does not depend on them.
    """
    evil_fail = evil_fail or {}
    assassin, morgana = _seat(record.get("assassin"), "刺客"), _seat(record.get("morgana"), "莫甘娜")
    if assassin == morgana:
        raise ValueError("刺客和莫甘娜不能是同一个座位。")
    game = AvalonGame(np.random.default_rng(0))
    roles = np.full(N_PLAYERS, Role.SERVANT, dtype=np.int8)
    roles[assassin], roles[morgana] = Role.ASSASSIN, Role.MORGANA
    good = [s for s in range(N_PLAYERS) if s not in (assassin, morgana)]
    roles[good[0]], roles[good[1]] = Role.MERLIN, Role.PERCIVAL
    game.roles = roles
    game.leader = _seat(record.get("first_leader"), "首任队长")

    fails = list(record.get("quest_fails", []))
    ambiguous = []
    for i, prop in enumerate(record.get("proposals", [])):
        q, a = game.quest + 1, game.attempt + 1
        if game.phase != Phase.PROPOSE:
            raise ValueError(f"第 {i + 1} 条组队记录之前游戏已经结束或仍在等待任务结果。")
        team = _seats(prop.get("team"), f"第 {q} 轮第 {a} 次组队的队伍")
        if len(team) != game.team_size:
            raise ValueError(f"第 {q} 轮任务需要 {game.team_size} 人，记录的队伍是 {len(team)} 人。")
        approvals = set(_seats(prop.get("approvals"), f"第 {q} 轮第 {a} 次组队的赞成票"))
        game.step({game.leader: A.team_action(game.leader, team)})
        game.step({p: A.APPROVE if p in approvals else A.REJECT for p in range(N_PLAYERS)})
        if game.phase != Phase.QUEST:
            continue
        if not fails:
            break  # quest result not entered yet
        n_fail = fails.pop(0)
        quest = game.quest
        evil_on = [s for s in team if s in (assassin, morgana)]
        if type(n_fail) is not int or not 0 <= n_fail <= len(team):
            raise ValueError(f"第 {quest + 1} 轮任务的失败牌数无效。")
        if n_fail > len(evil_on):
            raise ValueError(f"第 {quest + 1} 轮任务有 {n_fail} 张失败牌，但按你给的位置队伍里只有 "
                             f"{len(evil_on)} 个坏人：刺客/莫甘娜的位置与记录矛盾。")
        failing = set(evil_on) if n_fail == len(evil_on) else set()
        if n_fail == 1 and len(evil_on) == 2:
            ambiguous.append(quest)
            failing = {morgana if evil_fail.get(quest) == "morgana" else assassin}
        game.step({p: A.FAIL if p in failing else A.SUCCESS for p in team})
    if fails:
        raise ValueError("任务结果比通过的组队多。")
    return game, ambiguous


def assassin_probs(net: torch.nn.Module, game: AvalonGame) -> np.ndarray:
    """(6,) probability that the network's Assassin shoots each absolute seat (zero for evil seats)."""
    assassin = game.seat_of(Role.ASSASSIN)
    with torch.no_grad():
        logits = net.logits(torch.from_numpy(encode(game, assassin))[None])[0, :A.N_ACTIONS].numpy()
    mask = game.legal_mask(assassin)
    logits = np.where(mask, logits, -np.inf)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    out = np.zeros(N_PLAYERS)
    for rel in range(N_PLAYERS):
        out[A.to_absolute(assassin, rel)] = p[A.ASSASSIN_OFFSET + rel]
    return out


def advise(record: dict, nets: dict[int, torch.nn.Module]) -> dict:
    """The AI Assassin's distribution over good seats (UI seats 1–6), per seed and averaged."""
    game, ambiguous = replay(record)
    if game.phase != Phase.ASSASSINATE:
        raise ValueError("记录还没有走到刺杀阶段（好人需要先完成三次任务）。")
    variants = []
    for choice in product(("assassin", "morgana"), repeat=len(ambiguous)):
        g, _ = replay(record, dict(zip(ambiguous, choice)))
        per_seed = {seed: assassin_probs(net, g) for seed, net in nets.items()}
        label = "；".join(f"第 {q + 1} 轮失败牌由{'刺客' if c == 'assassin' else '莫甘娜'}打出"
                         for q, c in zip(ambiguous, choice))
        variants.append({"label": label, "choice": [[q + 1, c] for q, c in zip(ambiguous, choice)],
                         "per_seed": per_seed, "mean": np.mean(list(per_seed.values()), axis=0)})
    mean = np.mean([v["mean"] for v in variants], axis=0)
    per_seed = {seed: np.mean([v["per_seed"][seed] for v in variants], axis=0) for seed in nets}
    good = [s for s in range(N_PLAYERS) if not game.is_evil_seat(s)]
    ranking = sorted(good, key=lambda s: -mean[s])
    r = lambda x: [round(float(v), 4) for v in x]
    return {
        "pick": ranking[0] + 1,
        "ranking": [s + 1 for s in ranking],
        "mean": r(mean),
        "per_seed": {str(seed): r(p) for seed, p in per_seed.items()},
        "seed_picks": {str(seed): int(np.argmax(p)) + 1 for seed, p in per_seed.items()},
        "variants": [{"label": v["label"], "choice": v["choice"], "mean": r(v["mean"])} for v in variants] if ambiguous else [],
        "computed_utc": now(),
    }


# ---------------------------------------------------------------------------- storage

def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_record(data: dict) -> dict:
    """Keep only known fields with sane types; the rules are checked by `replay` when advice is asked."""
    if not isinstance(data, dict) or not ID_RE.match(str(data.get("id", ""))):
        raise ValueError("对局编号无效。")
    rec = {"id": data["id"], "created_utc": str(data.get("created_utc") or now())[:40],
           "updated_utc": now(), "note": str(data.get("note") or "")[:500]}
    for field in SEAT_FIELDS:
        value = data.get(field)
        if value is not None:
            _seat(value, field)
        rec[field] = value
    props = data.get("proposals", [])
    fails = data.get("quest_fails", [])
    if not isinstance(props, list) or len(props) > len(QUEST_SIZES) * MAX_PROPOSALS:
        raise ValueError("组队记录格式错误。")
    if not isinstance(fails, list) or len(fails) > len(QUEST_SIZES):
        raise ValueError("任务记录格式错误。")
    rec["proposals"] = []
    for prop in props:
        if not isinstance(prop, dict):
            raise ValueError("组队记录格式错误。")
        team, approvals = prop.get("team"), prop.get("approvals")
        _seats(team, "队伍"), _seats(approvals, "赞成票")
        rec["proposals"].append({"team": sorted(team), "approvals": sorted(approvals)})
    for f in fails:
        if type(f) is not int or not 0 <= f <= max(QUEST_SIZES):
            raise ValueError("失败牌数无效。")
    rec["quest_fails"] = list(fails)
    if isinstance(data.get("advice"), dict):
        rec["advice"] = data["advice"]
    share = data.get("share")
    if share is not None and type(share) is not bool:
        raise ValueError("共享选项无效。")
    rec["share"] = share  # None until the visitor decides; only True records enter the research data
    return rec


def owner_of(code: str) -> str:
    """Records store a hash of the visitor's code, so the files never contain the code itself."""
    return hashlib.sha256(code.encode()).hexdigest()[:16]


def row_of(g: dict) -> dict:
    row = {"id": g["id"], "created_utc": g.get("created_utc"), "note": g.get("note", ""),
           "outcome": outcome(g), "outcome_key": outcome_key(g), "actual_merlin": g.get("actual_merlin"),
           "human_target": g.get("human_target"), "share": g.get("share")}
    advice = g.get("advice")
    if advice:
        row["pick"] = advice["pick"]
        merlin = g.get("actual_merlin")
        if merlin is not None:
            row["p_true"] = advice["mean"][merlin - 1]
            row["ai_hit"] = advice["pick"] == merlin
            row["seed_hits"] = {k: v == merlin for k, v in advice["seed_picks"].items()}
            row["seed_p_true"] = {k: p[merlin - 1] for k, p in advice["per_seed"].items()}
    return row


def stats_of(rows: list[dict]) -> dict:
    """Hit rates over the rows whose true Merlin is known and that have AI advice."""
    judged = [r for r in rows if "ai_hit" in r]
    stats = {"games": len(rows), "judged": len(judged)}
    if judged:
        stats["ai_hits"] = sum(r["ai_hit"] for r in judged)
        stats["mean_p_true"] = round(float(np.mean([r["p_true"] for r in judged])), 4)
        seeds = sorted({k for r in judged for k in r["seed_hits"]})
        stats["per_seed"] = {k: {"n": len(rs), "hits": sum(r["seed_hits"][k] for r in rs),
                                 "mean_p_true": round(float(np.mean([r["seed_p_true"][k] for r in rs])), 4)}
                             for k in seeds for rs in [[r for r in judged if k in r["seed_hits"]]]}
        humans = [r for r in judged if r["human_target"] is not None]
        stats["human_judged"] = len(humans)
        stats["human_hits"] = sum(r["human_target"] == r["actual_merlin"] for r in humans)
    return stats


def public(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k != "owner"}


class Store:
    """One JSON file per game; each visitor (identified by the hash of their code) sees only their own games."""

    MAX_FILES = 20_000

    def __init__(self, directory: Path, nets: dict):
        self.dir = directory
        self.nets = nets
        self.lock = threading.Lock()

    def path(self, game_id: str) -> Path:
        if not ID_RE.match(str(game_id)):
            raise ValueError("对局编号无效。")
        return self.dir / f"{game_id}.json"

    def _read(self, game_id: str, owner: str) -> dict:
        rec = json.loads(self.path(game_id).read_text(encoding="utf-8"))
        if rec.get("owner") != owner:
            raise FileNotFoundError(game_id)  # someone else's game looks the same as a missing one
        return rec

    def save(self, data: dict, owner: str) -> dict:
        rec = clean_record(data)
        with self.lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            path = self.path(rec["id"])
            if path.exists():
                try:
                    rec["created_utc"] = self._read(rec["id"], owner).get("created_utc", rec["created_utc"])
                except FileNotFoundError:
                    raise ValueError("找不到这局记录。") from None
            elif sum(1 for _ in self.dir.glob("*.json")) >= self.MAX_FILES:
                raise ValueError("服务器记录已满，请联系管理员。")
            rec["owner"] = owner
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(path)
        return public(rec)

    def load(self, game_id: str, owner: str) -> dict:
        return public(self._read(game_id, owner))

    def delete(self, game_id: str, owner: str) -> None:
        with self.lock:
            self._read(game_id, owner)
            self.path(game_id).unlink()

    def advise(self, data: dict, owner: str) -> dict:
        rec = clean_record(data)
        with self.lock:
            rec["advice"] = advise(rec, self.nets)
        return self.save(rec, owner)

    def all_games(self) -> list[dict]:
        games = []
        for path in sorted(self.dir.glob("*.json")) if self.dir.exists() else []:
            try:
                games.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        games.sort(key=lambda g: g.get("created_utc", ""), reverse=True)
        return games

    def summary(self, owner: str) -> dict:
        games = self.all_games()
        mine = [row_of(g) for g in games if g.get("owner") == owner]
        shared = [row_of(g) for g in games if g.get("share") is True]
        return {"games": mine, "stats": stats_of(mine), "shared": stats_of(shared)}

    def adopt_orphans(self, code_file: Path) -> str | None:
        """Give records saved before visitor codes existed one code (kept in `code_file`); returns it."""
        orphans = [g for g in self.all_games() if "owner" not in g]
        if not orphans:
            return code_file.read_text().strip() if code_file.exists() else None
        code = code_file.read_text().strip() if code_file.exists() else secrets.token_hex(8)
        code_file.write_text(code + "\n")
        for g in orphans:
            g["owner"] = owner_of(code)
            self.path(g["id"]).write_text(json.dumps(g, ensure_ascii=False, indent=1), encoding="utf-8")
        return code


OUTCOMES = {"playing": "进行中", "evil5": "坏人胜（五次否决）", "evil3": "坏人胜（三次失败）", "assassinate": "刺杀阶段"}


def outcome_key(rec: dict) -> str:
    """Where a record stands, from the public record alone (no evil seats needed)."""
    fails, succ, att = 0, 0, 0
    quest_fails = list(rec.get("quest_fails", []))
    for prop in rec.get("proposals", []):
        if len(prop.get("approvals", [])) >= APPROVALS_NEEDED:
            att = 0
            if not quest_fails:
                return "playing"
            if quest_fails.pop(0) > 0:
                fails += 1
            else:
                succ += 1
        else:
            att += 1
            if att == MAX_PROPOSALS:
                return "evil5"
    return "evil3" if fails == 3 else "assassinate" if succ == 3 else "playing"


def outcome(rec: dict) -> str:
    return OUTCOMES[outcome_key(rec)]


# ---------------------------------------------------------------------------- HTTP

def parse_code(cookie_header: str) -> str | None:
    """The visitor's record code from a Cookie header, or None if absent or malformed."""
    cookies = SimpleCookie()
    try:
        cookies.load(cookie_header or "")
    except CookieError:
        return None
    morsel = cookies.get(COOKIE)
    return morsel.value if morsel and ID_RE.match(morsel.value) else None


def code_cookie(code: str, secure: bool) -> str:
    return f"{COOKIE}={code}; Path=/; Max-Age=63072000; HttpOnly; SameSite=Lax" + ("; Secure" if secure else "")


def advisor_api(store: Store, method: str, path: str, query: str, data: dict | None, code: str | None):
    """One advisor API call -> (payload, status, code to set as cookie or None). Paths start with /api/."""
    set_code = None
    if code is None:
        code = set_code = secrets.token_hex(8)
    owner = owner_of(code)
    try:
        if method == "GET" and path == "/api/new":
            return {"id": secrets.token_hex(8), "created_utc": now()}, 200, set_code
        if method == "GET" and path == "/api/me":
            return {"code": code}, 200, set_code
        if method == "GET" and path == "/api/games":
            return store.summary(owner), 200, set_code
        if method == "GET" and path == "/api/game":
            return store.load(parse_qs(query).get("id", [""])[0], owner), 200, set_code
        if method == "POST" and path == "/api/save":
            return store.save(data, owner), 200, set_code
        if method == "POST" and path == "/api/advise":
            return store.advise(data, owner), 200, set_code
        if method == "POST" and path == "/api/delete":
            store.delete(str(data.get("id", "")), owner)
            return {"ok": True}, 200, set_code
        if method == "POST" and path == "/api/code":  # switch to another device's code to see its games
            new = str(data.get("code", "")).strip().lower().replace(" ", "").replace("-", "")
            if not ID_RE.match(new):
                raise ValueError("记录码应为 16 位字母数字（0–9、a–f）。")
            return {"code": new}, 200, new
        return {"error": "未找到"}, 404, set_code
    except FileNotFoundError:
        return {"error": "找不到这局记录。"}, 404, set_code
    except ValueError as exc:  # includes rule violations found by replay
        return {"error": str(exc)}, 400, set_code


def handler_for(store: Store):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AssassinAdvisor/1"

        def log_message(self, fmt, *args):
            pass

        def send_body(self, body: bytes, mime: str, status=200, set_code=None):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", CSP)
            if set_code:
                self.send_header("Set-Cookie", code_cookie(set_code, self.headers.get("X-Forwarded-Proto") == "https"))
            self.end_headers()
            self.wfile.write(body)

        def api(self, method, data=None):
            url = urlsplit(self.path)
            payload, status, set_code = advisor_api(store, method, url.path, url.query, data,
                                                    parse_code(self.headers.get("Cookie", "")))
            self.send_body(json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8", status, set_code)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in STATIC:
                name, mime = STATIC[path]
                self.send_body((WEB / name).read_bytes(), mime + "; charset=utf-8")
            else:
                self.api("GET")

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length > 200_000:
                    raise ValueError("请求过大。")
                data = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(data, dict):
                    raise ValueError("请求必须为对象。")
            except ValueError as exc:
                return self.send_body(json.dumps({"error": str(exc)}, ensure_ascii=False).encode(),
                                      "application/json; charset=utf-8", 400)
            self.api("POST", data)

    return Handler


WEB = ROOT / "web" / "site"
# The page and its shared design/language files (also served by scripts/study_site.py under /advisor/).
STATIC = {"/": ("advisor.html", "text/html"), "/advisor.js": ("advisor.js", "text/javascript"),
          "/ui.css": ("ui.css", "text/css"), "/i18n.js": ("i18n.js", "text/javascript")}
CSP = ("default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; "
       "frame-ancestors 'none'")



def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--lan", action="store_true", help="监听 0.0.0.0，让同一局域网的手机也能打开")
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--save-dir", type=Path, default=SAVE_DIR)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--export-shared", type=Path, metavar="OUT.jsonl",
                        help="只把访客同意共享的对局导出为 JSONL（不含记录码），然后退出")
    args = parser.parse_args()
    if args.export_shared:
        shared = [public(g) for g in Store(args.save_dir, {}).all_games() if g.get("share") is True]
        args.export_shared.parent.mkdir(parents=True, exist_ok=True)
        with args.export_shared.open("w", encoding="utf-8") as f:
            for g in shared:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        print(f"已导出 {len(shared)} 局共享对局到 {args.export_shared}")
        return
    torch.set_num_threads(1)
    store = Store(args.save_dir, load_assassins(args.seeds))
    legacy = store.adopt_orphans(args.save_dir.with_name(args.save_dir.name + "_legacy_code.txt"))
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    try:
        server = ThreadingHTTPServer((host, args.port), handler_for(store))
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            parser.error(f"端口 {args.port} 已被占用，可改用 --port {args.port + 1}。")
        raise
    url = f"http://127.0.0.1:{args.port}"
    print(f"刺客辅助：{url}（baseline seeds {args.seeds}）\n记录保存到 {args.save_dir.resolve()}\n按 Ctrl+C 关闭。", flush=True)
    if legacy:
        print(f"旧记录（加入记录码之前保存的）归在记录码 {legacy} 下，在页面「记录码」里输入即可查看。", flush=True)
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
