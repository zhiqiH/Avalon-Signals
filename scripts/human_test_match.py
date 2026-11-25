"""Human test, match mode: one human and five frozen AI players (served by scripts/human_test.py).

Two kinds of table: "silent" (the silent baseline) and "talk" (the talking agents, which discuss before every
quest in the slot protocol, replies and conditional promises included). At a talk table the human's seat is free:
nothing limits what it says or does, only clear statements reach the agents (scripts/human_test_talk.py), and
deeds against one's words are counted in the public audit. No human seat is held to the good players' vote guard.
A language model may translate on both sides (scripts/human_test_llm.py): the visitor's typed or spoken words into
a stance, and an agent's message into a natural sentence. It runs outside the table lock: the table hands out the
plain facts it needs (`say_context`, `take_job`) and takes back the result (`hear`, `set_said`). The visitor's own
words are kept for the page only while the table lives; they are not part of any record.

Each browser gets its own table; frozen weights are shared, game state is not. Completed games of visitors
who agreed to share are appended to the match log (JSONL, one game per line, no IP or browser data).
"""
from __future__ import annotations

import hashlib
import json
import secrets
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts import human_test_llm as llm
from scripts import human_test_talk as talk
from src.agents import PolicyController, Request
from src.comm import protocol as P
from src.env import actions as A
from src.env.game import AvalonGame
from src.env.rules import N_PLAYERS, N_QUESTS, QUEST_SIZES, Phase, Role, WinReason
from src.training.checkpoint import comm_config, load_checkpoint, load_policies

ROOT = Path(__file__).resolve().parents[1]
TALK_DIR = ROOT / "comm"  # seed<k>/checkpoint.pt: the talking agents (local files, not in git)
MODES = ("silent", "talk")
CONSENT_VERSION = 3  # Bump when the consent wording in web/human_test changes (3: speech goes through a language model, 2026-11-25).
SAY_LIMITS = {"text": 60, "voice": 40}  # translations of the visitor's speech per game
ROLE_NAMES = {Role.SERVANT: "忠臣", Role.MERLIN: "梅林", Role.PERCIVAL: "派西维尔",
              Role.MORGANA: "莫甘娜", Role.ASSASSIN: "刺客"}
REASONS = {WinReason.ASSASSIN_MISSED: "刺杀未命中梅林", WinReason.ASSASSIN_HIT: "刺客命中梅林",
           WinReason.THREE_FAILS: "三次任务失败", WinReason.FIVE_REJECTIONS: "连续五次组队被否决"}


# Language-neutral option kinds for the browser (the labels above stay for older clients and logs).
OPTION_KINDS = {A.APPROVE: "approve", A.REJECT: "reject", A.SUCCESS: "success", A.FAIL: "fail"}


def seats(values) -> str:
    """The UI uses absolute seats 1–6; the environment uses 0–5."""
    return "、".join(str(int(s) + 1) for s in values) or "无"


def new_game(seed: int, human: int, role: str, comm: P.CommConfig | None = None) -> AvalonGame:
    """A random game; with a requested role, the human swaps roles with whoever drew it."""
    rng = np.random.default_rng([seed, 0])
    game = AvalonGame(rng, comm=comm)
    game.unguarded_seats = frozenset({human})  # a human may make a losing vote
    if game.comm.enabled:
        game.free_seats = frozenset({human})  # ... and is not bound by the agents' rules of speech
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

    def __init__(self, default_baseline=0, models=None, log=None, clock=time.time, talk_dir=TALK_DIR, language_model=None):
        self.default_baseline = default_baseline
        self.talk_dir = Path(talk_dir)
        self.language_model = language_model  # name of the model that translates at talk tables (for the record)
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
        mode = data.get("mode", "silent")
        if mode not in MODES:
            raise ValueError("未知模式。")
        if type(baseline) is not int or baseline not in (0, 1, 2):
            raise ValueError("请选择 baseline 0、1 或 2。")
        if type(seat) is not int or seat not in range(1, 7):
            raise ValueError("座位必须为 1–6。")
        if role not in ["random", *[r.name.lower() for r in Role]]:
            raise ValueError("未知角色。")
        if type(share) is not bool:
            raise ValueError("请选择是否共享本局数据。")
        key = (mode, baseline)
        if key not in self.models:
            path = (self.talk_dir / f"seed{baseline}/checkpoint.pt" if mode == "talk"
                    else ROOT / f"baseline/seed{baseline}/checkpoint_19532.pt")
            nets = load_policies(path)
            comm = comm_config(load_checkpoint(path)["config"])
            if comm.enabled != (mode == "talk"):
                raise ValueError("模型与所选模式不符。")
            self.models[key] = {s: PolicyController(n, learn=False) for s, n in nets.items()}
            self.models[key].update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), comm=comm)
        self.mode, self.ai = mode, self.models[key]
        self.baseline, self.human, self.role, self.share = baseline, seat - 1, role, share
        self.seed = secrets.randbits(32)
        self.game = new_game(self.seed, self.human, role, self.ai["comm"])
        self.rng = np.random.default_rng([self.seed, 1_000_003])
        self.id, self.revision = secrets.token_hex(12), 0
        self.started = self.clock()
        self.games += 1
        self.event = {"kind": "start", "text": "身份已分配。请先阅读左侧的私人信息，再开始行动。"}
        self.decisions = []
        self.said, self.job, self.say_used = {}, None, {"text": 0, "voice": 0}
        return self.view()

    # ---- the language model's side of a speech (scripts/human_test_llm.py); the model itself is called by the server
    def my_turn_to_speak(self):
        game = self.game
        if game is None or game.done or not game.in_discussion or game.acting_players() != [self.human]:
            raise ValueError("现在不是你的发言时间。")
        return game

    def say_context(self, data):
        """What a translation of the visitor's speech needs about the table; counts one translation of this game."""
        game, kind = self.my_turn_to_speak(), "voice" if data.get("voice") else "text"
        if self.say_used[kind] >= SAY_LIMITS[kind]:
            raise ValueError("本局的识别次数已用完，请用下方按钮发言。")
        self.say_used[kind] += 1
        return {"ctx": llm.say_context(game, self.human), "id": self.id, "revision": self.revision}

    def hear(self, data):
        """The stance of the statements a language model read from the visitor's speech (changes nothing)."""
        game = self.my_turn_to_speak()
        if data.get("id") != self.id or data.get("revision") != self.revision:
            raise ValueError("牌桌已更新，请刷新后再操作。")
        result = llm.to_stance(data["items"], llm.say_context(game, self.human))
        result["stance"] = talk.clean_stance(result["stance"], game, self.human)
        return result

    def take_job(self):
        """The agent's speech that was just made, for the language model to put into a sentence (given out once)."""
        job, self.job = self.job, None
        return job

    def set_said(self, data):
        """Keep the sentence a language model made of an agent's message."""
        if data.get("id") == self.id and self.game is not None:
            self.said.setdefault(tuple(data["key"]), {})[data["lang"]] = data["text"]
            if self.event.get("kind") == "speech" and self.event.get("key") == list(data["key"]):
                self.event["message"]["said"] = self.said[tuple(data["key"])]
        return self.view()

    def parse(self, data):
        """What the agents would hear in a typed sentence (changes nothing)."""
        game = self.game
        if game is None or game.done or not game.in_discussion or game.acting_players() != [self.human]:
            raise ValueError("现在不是你的发言时间。")
        text = data.get("text")
        if not isinstance(text, str):
            raise ValueError("发言格式不正确。")
        return talk.parse(text, game, self.human)

    def speak(self, data, advance):
        """One speaker of the discussion: the human's stance, or the next agent's message."""
        game, speaker = self.game, self.game.acting_players()[0]
        phase, q = game.phase, game.quest
        size = QUEST_SIZES[0] if phase == Phase.EVIL_DISCUSS else game.team_size
        key = ("evil" if phase == Phase.EVIL_DISCUSS else int(q), int(speaker))  # where this speech sits in the talk log
        self.job = None
        if speaker == self.human:
            if advance or "stance" not in data:
                raise ValueError("请先发言，或选择不发言。")
            stance = talk.clean_stance(data["stance"], game, self.human)
            msg = talk.stance_message(game, self.human, stance)
            words = data.get("text")
            if isinstance(words, str) and words.strip():  # the visitor's own words, for the page only
                self.said[key] = {"raw": words.strip()[:llm.MAX_TEXT]}
            self.decisions.append({"phase": phase.name, "quest": q, "message": P.decode_message(msg, speaker, size),
                                   "typed": bool(data.get("typed")), "seconds": round(self.clock() - self.started, 1)})
        else:
            if not advance:
                raise ValueError("当前请点击继续，等待 AI 行动。")
            ctx = P.message_context(game, speaker)
            msg = P.forced_message(ctx)
            if msg is None:
                side = "evil" if game.is_evil_seat(speaker) else "good"
                msg = self.ai[side].speak([Request(0, game, speaker, None, ctx)], self.rng).messages[0]
            said_ctx = llm.say_context(game, speaker)  # the table as the speaker met it, for the sentence
        message = talk.message_view(P.decode_message(np.asarray(msg), speaker, size))
        if speaker != self.human:
            self.job = {"id": self.id, "key": list(key), "view": message, "ctx": said_ctx,
                        "teams": {o["seat"]: o["team"] for o in said_ctx["earlier"] if o["team"]}}
        game.step({speaker: msg})
        if key in self.said:
            message["said"] = self.said[key]
        self.event = {"kind": "speech", "evil_talk": phase == Phase.EVIL_DISCUSS, "message": message, "key": list(key),
                      "text": f"{speaker + 1} 号发言。"}
        self.revision += 1
        return self.view()

    def step(self, data, advance=False):
        game = self.game
        if game is None or game.done:
            raise ValueError("请先开始一局新游戏。")
        if data.get("id") != self.id or data.get("revision") != self.revision:
            raise ValueError("牌桌已更新，请刷新后再操作。")
        if game.in_discussion:
            return self.speak(data, advance)
        if "stance" in data:
            raise ValueError("现在不是发言时间。")
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
                result = self.ai[side].act(requests, self.rng)
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
        return {"kind": "human_vs_baseline" if self.mode == "silent" else "human_vs_talk", "mode": self.mode,
                "agents": "baseline" if self.mode == "silent" else self.talk_dir.name,
                "human_free": self.mode == "talk", "human_guarded": False, "consent_version": CONSENT_VERSION,
                # The sentences a language model made of the agents' messages, as the visitor read them (quest 0-4
                # or "evil", seat 0-5). The visitor's own words are not recorded, only the stance in `human_decisions`.
                "language_model": self.language_model if self.mode == "talk" else None,
                "spoken": [{"quest": q, "seat": s, **{k: v for k, v in x.items() if k != "raw"}}
                           for (q, s), x in self.said.items() if s != self.human],
                "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "player": self.player, "player_game": self.games, "table_id": self.id,
                "baseline_seed": self.baseline, "checkpoint_sha256": self.ai["sha256"],
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
        if human_turn and not g.in_discussion:
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
            "active": True, "id": self.id, "revision": self.revision, "baseline": self.baseline, "mode": self.mode,
            "human": h + 1, "role": ROLE_NAMES[Role(int(g.roles[h]))], "role_key": Role(int(g.roles[h])).name,
            "side": "evil" if g.is_evil_seat(h) else "good",
            "known_evil": [int(s) + 1 for s in np.flatnonzero(evil)],
            "candidates": [int(s) + 1 for s in np.flatnonzero(candidates)],
            "own_cards": g.quest_cards[h].tolist(), "phase": g.phase.name,
            "quest": min(g.quest + 1, 5), "attempt": min(g.attempt + 1, 5),
            "leader": int(g.prop_leader[g.quest, g.attempt]) + 1 if g.phase == Phase.QUEST else g.leader + 1,
            "team": [int(s) + 1 for s in np.flatnonzero(g.team)],
            "team_size": g.team_size if g.phase in (Phase.DISCUSS, Phase.PROPOSE, Phase.VOTE, Phase.QUEST) else 0,
            "successes": g.successes, "fails": g.fails,
            "quests": [{"size": size, "result": int(g.quest_result[q]),
                        "fails": int(g.quest_fails[q]) if g.quest_result[q] >= 0 else None}
                       for q, size in enumerate(QUEST_SIZES)],
            "human_turn": human_turn, "options": options, "history": history,
            "event": self.event, "done": g.done, "share": self.share,
        }
        if g.comm.enabled:
            state["speaker"] = g.acting_players()[0] + 1 if g.in_discussion else None
            state["talk"] = self.talk_log()
            # Public count per seat of deeds against its own words (judgment, intent, own suggestion, reply, promise).
            state["broken_words"] = [int(n) for n in g.audit[:, [P.A_JUDGE, P.A_INTENT, P.A_TEAM, P.A_REPLY, P.A_PROMISE]].sum(axis=1)]
            state["say"] = talk.openings(g, h) if human_turn and g.phase == Phase.DISCUSS else None
        if g.done:
            state["result"] = {"good_won": bool(g.good_won), "human_won": bool(g.good_won) != g.is_evil_seat(h),
                               "reason": REASONS[g.win_reason], "target": g.assassin_target + 1,
                               "roles": [ROLE_NAMES[Role(int(r))] for r in g.roles],
                               "reason_key": g.win_reason.name, "role_keys": [Role(int(r)).name for r in g.roles]}
        return state

    def talk_log(self):
        """Every discussion so far, in speaking order: [{"quest": 1.., "evil_talk": bool, "messages": [...]}]."""
        g, rounds = self.game, []
        for q in range(N_QUESTS):
            spoke = g.msg_log[q, :, 0] >= 0
            if not spoke.any():
                continue
            leader = int(g.prop_leader[q, 0]) if g.prop_leader[q, 0] >= 0 else g.leader  # still talking
            order = [(leader + k) % N_PLAYERS for k in range(1, N_PLAYERS)] + [leader]
            rounds.append({"quest": q + 1, "evil_talk": False, "messages": [
                self.with_said(talk.message_view(P.decode_message(g.msg_log[q, s], s, QUEST_SIZES[q])), (q, int(s))) for s in order if spoke[s]]})
        spoke = g.evil_log[:, 0] >= 0
        if spoke.any():
            order = [g.seat_of(Role.MORGANA), g.seat_of(Role.ASSASSIN)]
            rounds.append({"quest": None, "evil_talk": True, "messages": [
                self.with_said(talk.message_view(P.decode_message(g.evil_log[s], s, QUEST_SIZES[0])), ("evil", int(s))) for s in order if spoke[s]]})
        return rounds

    def with_said(self, message, key):
        """A message view with its sentences: {"zh" / "en": the language model's, "raw": the visitor's own words}."""
        if key in self.said:
            message["said"] = self.said[key]
        return message

    def replay(self):
        if self.game is None or not self.game.done:
            raise ValueError("对局结束后才能下载完整复盘。")
        return self.record()


class SessionExpired(ValueError):
    pass


class TableSessions:
    """Independent browser sessions sharing frozen model weights, never game state."""

    def __init__(self, default_baseline=0, ttl=7200, capacity=128, idle_evict=600, log=None,
                 clock=time.monotonic, talk_dir=TALK_DIR, language_model=None):
        self.default_baseline, self.ttl, self.capacity, self.idle_evict = default_baseline, ttl, capacity, idle_evict
        self.talk_dir, self.language_model = talk_dir, language_model
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
                self.tables[session_id] = (HumanTable(self.default_baseline, self.models, self.log, talk_dir=self.talk_dir,
                                                      language_model=self.language_model), now)
            table, _ = self.tables[session_id]
            self.tables[session_id] = (table, now)
            try:
                if path == "/api/state":
                    result = table.view()
                elif path == "/api/new":
                    result = table.start(data)
                elif path == "/api/replay":
                    result = table.replay()
                elif path == "/api/parse":
                    result = table.parse(data)
                elif path == "#context":  # the "#" paths serve the language model and are not reachable over HTTP
                    result = table.say_context(data)
                elif path == "#items":
                    result = table.hear(data)
                elif path == "#job":
                    result = table.take_job()
                elif path == "#said":
                    result = table.set_said(data)
                elif path in ("/api/action", "/api/advance", "/api/speak"):
                    result = table.step(data, advance=path.endswith("advance"))
                else:
                    raise ValueError("未知接口。")
            except Exception:
                if created:
                    del self.tables[session_id]
                raise
            return result, session_id
