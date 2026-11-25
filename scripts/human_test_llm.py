"""Human test, language model: the ears and the mouth between a visitor and the talking agents.

The agents decide and speak in the slot protocol only (src/comm/protocol.py); a language model translates on
both sides and decides nothing:

    ears    what the visitor typed or said -> statements -> a stance (scripts/human_test_talk.py). Only clear
            statements are passed on; every statement is checked against the table by fixed rules (`to_stance`).
    mouth   an agent's message -> one natural sentence. The sentence is read back through the ears and is used
            only if it gives the same stance as the message; otherwise the page shows its template sentence.
    voice   recorded speech -> text (then the ears).

The key is read from a local file (OPENAI_API_KEY=...) or the environment and never leaves the server. Text and
audio are sent to the provider for translation and are not written to any log here. Without a key, on an error
or past the daily budget, the caller falls back to the fixed rules and the template sentences.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from scripts import human_test_talk as talk

ROOT = Path(__file__).resolve().parents[1]
KEY_FILE = ROOT / "openai_api_key.txt"
API = "https://api.openai.com/v1"
CHAT_MODEL, VOICE_MODEL = "gpt-4.1-mini", "gpt-4o-mini-transcribe"
MAX_TEXT, MAX_AUDIO = 300, 2_000_000  # characters of one speech; bytes of one recording (about 60 s)
AUDIO_TYPES = {"audio/webm": "webm", "audio/mp4": "mp4", "audio/ogg": "ogg", "audio/wav": "wav", "audio/mpeg": "mp3",
               "audio/x-m4a": "m4a", "audio/aac": "m4a", "video/webm": "webm", "video/mp4": "mp4"}
KINDS = ("claim", "judge", "team", "veto", "require", "reply", "promise", "soft", "other")
VALUES = ("servant", "percival", "merlin", "good", "evil", "endorse", "dispute", "fail_evil", "success_good")
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {
    "type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["kind", "seats", "value", "quote"],
                               "properties": {"kind": {"type": "string", "enum": list(KINDS)},
                                              "seats": {"type": "array", "items": {"type": "integer"}},
                                              "value": {"type": ["string", "null"], "enum": [*VALUES, None]},
                                              "quote": {"type": "string"}}}}}}
SAY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["text"], "properties": {"text": {"type": "string"}}}


class Unavailable(RuntimeError):
    """No key, no budget left, or the provider failed: the caller falls back."""


def read_key(path: Path | None) -> str | None:
    """OPENAI_API_KEY from the environment, else from a file with a line `OPENAI_API_KEY=...` (or the bare key)."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key or path is None or not Path(path).is_file():
        return key or None
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key = line.split("=", 1)[1] if "=" in line else line
            return key.strip().strip("\"'") or None
    return None


class LanguageModel:
    """Calls to the provider with a time limit and a daily budget. Thread-safe; holds no visitor data."""

    def __init__(self, key_file: Path | None = KEY_FILE, chat_model=CHAT_MODEL, voice_model=VOICE_MODEL,
                 timeout=12.0, daily_chat=6000, daily_voice=1500):
        self.key = read_key(key_file)
        self.chat_model, self.voice_model, self.timeout = chat_model, voice_model, timeout
        self.limits = {"chat": daily_chat, "voice": daily_voice}
        self.used, self.day, self.lock = {"chat": 0, "voice": 0}, None, threading.Lock()

    @property
    def available(self) -> bool:
        return bool(self.key)

    def _spend(self, kind: str) -> None:
        if not self.key:
            raise Unavailable("no key")
        with self.lock:
            day = time.strftime("%F")
            if day != self.day:
                self.day, self.used = day, {"chat": 0, "voice": 0}
            if self.used[kind] >= self.limits[kind]:
                raise Unavailable("daily budget used up")
            self.used[kind] += 1

    def _post(self, path: str, body: bytes, content_type: str, timeout: float | None = None) -> dict:
        request = urllib.request.Request(API + path, data=body, method="POST",
                                         headers={"Authorization": f"Bearer {self.key}", "Content-Type": content_type})
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:  # the body may name the request; keep only the status
            raise Unavailable(f"provider status {exc.code}") from None
        except (OSError, ValueError) as exc:
            raise Unavailable(type(exc).__name__) from None

    def chat(self, system: str, user: str, schema: dict, name: str, max_tokens=500, timeout: float | None = None) -> dict:
        """One structured answer (JSON by `schema`)."""
        self._spend("chat")
        body = {"model": self.chat_model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}}
        if self.chat_model.startswith("gpt-5"):
            body.update(max_completion_tokens=max_tokens + 200, reasoning_effort="none" if re.match(r"gpt-5\.\d", self.chat_model) else "minimal")
        else:
            body.update(max_tokens=max_tokens, temperature=0)
        answer = self._post("/chat/completions", json.dumps(body).encode("utf-8"), "application/json", timeout)
        try:
            return json.loads(answer["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError, ValueError):
            raise Unavailable("no structured answer") from None

    def transcribe(self, audio: bytes, mime: str, lang: str) -> str:
        """Speech to text. `lang`: "zh" or "en", the page language (a hint)."""
        ext = AUDIO_TYPES.get(mime.split(";")[0].strip().lower())
        if ext is None or not 0 < len(audio) <= MAX_AUDIO:
            raise ValueError("录音格式或长度不支持。")
        self._spend("voice")
        boundary = uuid.uuid4().hex
        hint = ("阿瓦隆桌游发言。1号、2号、3号、4号、5号、6号，好人，坏人，梅林，派西维尔，莫甘娜，刺客，忠臣，推荐，上车，任务，投票。" if lang == "zh"
                else "Avalon board game talk. Seat 1, seat 2, seat 3, seat 4, seat 5, seat 6, good, evil, Merlin, Percival, Morgana, Assassin, Servant, quest, team.")
        fields = {"model": self.voice_model, "language": "zh" if lang == "zh" else "en", "response_format": "json", "prompt": hint}
        parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode("utf-8") for k, v in fields.items()]
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="speech.{ext}"\r\n'
                     f"Content-Type: {mime.split(';')[0]}\r\n\r\n".encode("utf-8") + audio + f"\r\n--{boundary}--\r\n".encode("utf-8"))
        answer = self._post("/audio/transcriptions", b"".join(parts), f"multipart/form-data; boundary={boundary}", timeout=20)
        text = answer.get("text") if isinstance(answer, dict) else None
        if not isinstance(text, str):
            raise Unavailable("no transcript")
        text = text.strip()[:MAX_TEXT]
        # On silence or noise the provider may answer with the hint itself: that is no speech.
        bare = lambda x: re.sub(r"[\W_]+", "", x).lower()  # noqa: E731
        return "" if len(bare(text)) >= 8 and bare(text) in bare(hint) else text


# ---------------------------------------------------------------------- ears: text -> statements -> stance
EARS = """You translate what ONE player says at a six-player Avalon table into structured statements for game-playing agents. \
You decide nothing and add nothing: output only what the speaker states. The speech is Chinese or English, typed or \
transcribed from voice (it may lack punctuation; "三号", "3号位", "老三", "player 3", "seat 3", "#3" all mean seat 3).

Seats are numbered 1-6. Split the speech into items; every part of the speech belongs to exactly one item. For each item give:
- kind, seats (seat numbers it is about; "我/自己/me/myself" is the speaker's own seat), value, and quote (the exact words of the speech).

Kinds at the table (the discussion before a quest team is picked):
- claim: the speaker states their OWN role ("我是忠臣", "我是派西维尔", "I am Percival", "I'm a loyal servant"). value servant | percival | merlin. seats []. ("我是好人 / I'm good" names no role: other.)
- judge: the speaker states that players ARE good or evil. value good | evil. Slang: 金水/保/可信/干净/clean/trust = good; \
狼/铁狼/坏的/踩/有问题/不可信/spy/minion/sus(stated flatly) = evil. "我觉得/我认为/I think/I believe X是坏人" is a clear statement. \
One item per group of players with the same value.
- team: the speaker suggests who should go on this quest. seats = exactly the players named, plus the speaker's seat only when they name themselves ("我/自己/带上我/me"). Never add a player to reach the team size. value null.
- veto: the speaker will vote against any team that contains these players ("别带3号", "有3号我就反对", "don't take 3", "no team with 3").
- require: the speaker will only approve teams that contain these players ("必须带5号", "没有5号我不投", "5 must be on the team").
- reply: the speaker agrees (value endorse) or disagrees (value dispute) with an EARLIER SPEAKER of this round as a person or with \
the team that player suggested ("同意6号", "6号的队我赞成", "6号说得不对", "6号的车我不上", "I'm with 6", "6's team is bad"). seats = ONLY the earlier speaker(s) answered, never the members of the team they suggested, and the answer is no team suggestion of the speaker's own. \
If the speaker also repeats a judgment ("我同意1号，3号是坏人" -> judge evil [3] plus reply endorse [1]).
- promise: a condition on THIS round's quest result: "如果任务失败，我就认定3号是坏人" -> value fail_evil; \
"如果成功我就认3号是好人" -> value success_good. seats = the players it is about.
- soft: a hedged, uncertain or weak opinion about players (有点怪, 可能, 也许, 好像, 感觉, 不确定, 轻踩, 小踩, 先观察, maybe, might, \
not sure, a bit, kind of, seems, probably). It is NOT passed on to the agents. Never turn a hedged opinion into judge, veto or require.
- other: everything else: questions, reasons, descriptions of what happened, jokes, statements with no player or no stance.

A sentence can hold several items ("3号是坏人，别带他" -> judge evil [3] and veto [3]). A negation flips the value \
("3号不是坏人" -> judge good [3]; "不能信4号" -> judge evil [4]). Do not invent seats that are not named."""

EARS_EVIL = """You translate what ONE evil player says in the private talk before the assassination (six-player Avalon) into \
structured statements. The evil players discuss which good player is Merlin. You decide nothing and add nothing. \
The speech is Chinese or English, typed or transcribed from voice ("三号", "3号位", "player 3" all mean seat 3).

For each part of the speech give kind, seats, value, and quote (the exact words):
- judge: the speaker states which good role players have. value merlin | percival | servant. \
("5号是梅林" -> judge merlin [5]; "2号像忠臣，不是梅林" -> judge servant [2].) "我觉得/我认为/I think/I believe 5 is Merlin" is a clear statement: judge merlin [5].
A proposal to assassinate a player ("刺5号", "杀5号吧", "stab 5", "let's kill 5") says that player is Merlin: judge merlin [5].
- soft: a hedged or uncertain guess (可能, 也许, 好像, 不确定, maybe, might, not sure). Not passed on.
- other: everything else.
Never use the kinds claim, team, veto, require, reply or promise here."""


def table_note(ctx: dict) -> str:
    """The facts about the table that the ears need, for one speaker."""
    if ctx["evil_talk"]:
        return f"The speaker sits in seat {ctx['me']}."
    earlier = "; ".join(f"seat {o['seat']} (suggested team {o['team']})" if o["team"] else f"seat {o['seat']} (suggested no team)"
                        for o in ctx["earlier"]) or "nobody yet"
    return (f"The speaker sits in seat {ctx['me']}. This is the discussion of quest {ctx['quest']}; the team needs {ctx['team_size']} players; "
            f"the leader is seat {ctx['leader']}. Earlier speakers of this round: {earlier}.")


def say_context(game, human: int) -> dict:
    """What the ears need to know about the table when `human` speaks now (plain data, safe outside the table lock)."""
    from src.env.rules import Phase
    evil_talk = game.phase == Phase.EVIL_DISCUSS
    open_ = talk.openings(game, human)
    return {"me": human + 1, "evil_talk": evil_talk, "quest": game.quest + 1, "team_size": None if evil_talk else game.team_size,
            "leader": game.leader + 1, "earlier": open_["reply"], "reply_open": open_["reply_open"], "promise": open_["promise"]}


def understand(model: LanguageModel, text: str, ctx: dict, timeout: float | None = None) -> list[dict]:
    """The statements in `text`, as the model reads them (not yet checked against the table)."""
    text = text.strip()[:MAX_TEXT]
    if not text:
        return []
    answer = model.chat(EARS_EVIL if ctx["evil_talk"] else EARS, f"{table_note(ctx)}\nSpeech: {json.dumps(text, ensure_ascii=False)}",
                        SCHEMA, "statements", timeout=timeout)
    items = answer.get("items")
    if not isinstance(items, list):
        raise Unavailable("no items")
    return [i for i in items if isinstance(i, dict)]


def to_stance(items: list[dict], ctx: dict) -> dict:
    """Statements -> {"stance", "heard", "ignored"} for the speaker of `ctx`, by fixed rules: what the table does
    not allow is left out with its reason (the same reasons as the rule parser `human_test_talk.parse`)."""
    me, evil_talk = ctx["me"], ctx["evil_talk"]
    stance, heard, ignored = talk.empty_stance(), [], []
    spoke = {o["seat"]: o["team"] for o in ctx["earlier"]}
    for item in items:
        kind, value, quote = item.get("kind"), item.get("value"), str(item.get("quote") or "").strip()[:MAX_TEXT]
        named = [s for s in item.get("seats") or [] if type(s) is int and 1 <= s <= 6]
        named = sorted(set(named))
        others = [s for s in named if s != me]
        why = "unknown"
        if kind == "soft":
            why = "soft"
        elif kind == "other" or kind not in KINDS:
            why = "other"
        elif evil_talk:
            if kind == "judge" and value in talk.EVIL_TALK_JUDGMENTS and others:
                stance["judge"].update({str(s): value for s in others})
                why = None
        elif kind == "claim":
            if value == "merlin":
                why = "merlin_claim"
            elif value in talk.CLAIM:
                stance["claim"], why = value, None
        elif kind == "judge" and others:
            if value in talk.TABLE_JUDGMENTS:
                stance["judge"].update({str(s): value for s in others})
                why = None
            elif value in talk.EVIL_TALK_JUDGMENTS:
                why = "role_name"
        elif kind == "team" and named:
            if me == ctx["leader"]:
                why = "leader"
            elif len(named) != ctx["team_size"]:
                why = "size"
            else:
                stance["team"], why = named, None
        elif kind in ("veto", "require") and others:
            stance[kind] = sorted(set(stance[kind]) | set(others))
            why = None
        elif kind == "reply" and others and value in talk.REPLY and ctx["reply_open"]:
            if any(s not in spoke for s in others):
                why = "not_spoken"
            elif value == "endorse" and any(not spoke[s] for s in others):
                why = "no_team"
            else:
                stance["reply"].update({str(s): value for s in others})
                why = None
        elif kind == "promise" and others and value in talk.PROMISE:
            if value not in ctx["promise"]:
                why = "promise_closed"
            elif any(stance["promise"].get(str(s), value) != value for s in others):
                why = "promise_both"
            else:
                stance["promise"].update({str(s): value for s in others})
                why = None
        if quote:
            (heard.append(quote) if why is None else ignored.append({"text": quote, "why": why}))
    stance["require"] = [s for s in stance["require"] if s not in stance["veto"]]
    return {"stance": stance, "heard": heard, "ignored": ignored}


# ---------------------------------------------------------------------- mouth: message -> sentence
_T = {
    "zh": {"claim.servant": "我是忠臣", "claim.percival": "我是派西维尔", "claim.merlin": "我是梅林", "good": "好人", "evil": "坏人",
           "merlin": "梅林", "percival": "派西维尔", "servant": "忠臣", "morgana": "莫甘娜", "assassin": "刺客",
           "judge": "{n} 号是{j}", "basis": "（依据第 {q} 轮{k}记录）", "sight": "（依据身份信息）",
           "quest": "任务", "vote": "投票", "proposal": "组队", "team": "推荐 {team} 号", "veto": "有 {n} 号的队我不赞成",
           "require": "没有 {n} 号的队我不赞成", "none": "（不发言）", "endorse": "同意 {n} 号，他推荐的队我赞成",
           "dispute": "不同意 {n} 号{b}", "disputeTeam": "不同意 {n} 号{b}，他推荐的队我不赞成",
           "rb.defense": "（他指认了我）", "rb.judgment": "（我们对同一人判断相反）", "rb.team": "（他推荐的队里有我指认的人）",
           "fail_evil": "{n} 号若上本轮任务且失败，下轮我说他是坏人", "success_good": "{n} 号若上本轮任务且成功，下轮我说他是好人",
           "fail_evil.n": "本轮任务若失败，{n} 号谁在队里，下轮我说谁是坏人", "success_good.n": "本轮任务若成功，{n} 号谁在队里，下轮我说谁是好人",
           "sep": "；", "seats": "、"},
    "en": {"claim.servant": "I am a Servant", "claim.percival": "I am Percival", "claim.merlin": "I am Merlin", "good": "good", "evil": "evil",
           "merlin": "Merlin", "percival": "Percival", "servant": "Servant", "morgana": "Morgana", "assassin": "Assassin",
           "judge": "Seat {n} is {j}", "basis": " (from the {k} record of quest {q})", "sight": " (from my role)",
           "quest": "quest", "vote": "vote", "proposal": "proposal", "team": "I suggest {team}", "veto": "I reject any team with seat {n}",
           "require": "I reject any team without seat {n}", "none": "(says nothing)", "endorse": "I agree with seat {n} and approve their team",
           "dispute": "I disagree with seat {n}{b}", "disputeTeam": "I disagree with seat {n}{b} and reject their team",
           "rb.defense": " (they accused me)", "rb.judgment": " (we judge a player in opposite ways)", "rb.team": " (their team has a player I accuse)",
           "fail_evil": "If seat {n} goes on this quest and it fails, next round I call them evil",
           "success_good": "If seat {n} goes on this quest and it succeeds, next round I call them good",
           "fail_evil.n": "If this quest fails, next round I call evil whoever of seats {n} was on it",
           "success_good.n": "If this quest succeeds, next round I call good whoever of seats {n} was on it",
           "sep": "; ", "seats": ", "},
}


def template(m: dict, lang: str, teams: dict | None = None) -> str:
    """The fixed sentence of a message view (`human_test_talk.message_view`); the same wording as the page's."""
    t, teams, parts = _T["zh" if lang == "zh" else "en"], teams or {}, []
    seats = lambda xs: t["seats"].join(str(x) for x in xs)  # noqa: E731
    if m.get("claim"):
        parts.append(t["claim." + m["claim"]])
    for j in m.get("judge") or []:
        basis = (t["sight"] if j.get("basis") == "role_knowledge"
                 else t["basis"].format(q=j["quest"], k=t[j["basis"]]) if j.get("basis") and j.get("quest") else "")
        parts.append(t["judge"].format(n=j["seat"], j=t[j["judgment"]]) + basis)
    if m.get("team"):
        parts.append(t["team"].format(team=seats(m["team"])))
    parts += [t["veto"].format(n=n) for n in m.get("veto") or []]
    parts += [t["require"].format(n=n) for n in m.get("require") or []]
    for r in m.get("reply") or []:
        b = t["rb." + r["basis"]] if r.get("basis") else ""
        key = "endorse" if r["reply"] == "endorse" else "disputeTeam" if teams.get(r["seat"]) else "dispute"
        parts.append(t[key].format(n=r["seat"], b=b))
    for kind in ("fail_evil", "success_good"):
        about = [k["seat"] for k in m.get("promise") or [] if k["promise"] == kind]
        if about:
            parts.append(t[kind + (".n" if len(about) > 1 else "")].format(n=seats(about)))
    return t["sep"].join(parts) if parts else t["none"]


def message_stance(m: dict) -> dict:
    """The stance a message view carries (what the ears must find again in its sentence)."""
    return {"claim": m.get("claim"), "judge": {str(j["seat"]): j["judgment"] for j in m.get("judge") or []},
            "team": sorted(m["team"]) if m.get("team") else None, "veto": sorted(m.get("veto") or []), "require": sorted(m.get("require") or []),
            "reply": {str(r["seat"]): r["reply"] for r in m.get("reply") or []},
            "promise": {str(k["seat"]): k["promise"] for k in m.get("promise") or []}}


MOUTH = """You put one Avalon player's table statement into natural spoken {language}, in the first person, the way a \
real player says it out loud at the table. You are a translator: the meaning is fixed.

Keep EVERY statement and its exact meaning: the role the speaker claims, who is called good or evil (or which role), \
the reason given in brackets, the suggested team with all its seat numbers, the vote conditions (voting against teams \
with / without a player), agreeing or disagreeing with a player and what that means for that player's team, and \
the conditional promises (the full condition, and what the speaker will say next round).
Never add anything: no new opinion, reason, player, team member, greeting or filler, nothing about the table that the \
statement does not say, and no words of doubt (maybe, I think, 可能, 感觉, 好像, 应该). Do not weaken or strengthen a statement. \
Name players as {seat_form}. Keep it short: at most {limit}. Join the parts into flowing speech; do not use brackets or semicolons.

Examples of the wording ({language}):
{examples}"""
_EXAMPLES = {
    "zh": """我是忠臣；1 号是好人（依据第 2 轮任务记录） -> 我是忠臣。1号是好人，第2轮的任务结果能说明。
3 号是坏人（依据身份信息） -> 从我的身份能看出来，3号是坏人。
4 号是坏人（依据第 1 轮投票记录）；有 4 号的队我不赞成 -> 看第1轮的投票，4号是坏人，队里有4号我就投反对。
推荐 2、3、6 号；没有 5 号的队我不赞成 -> 这轮我推荐2号、3号、6号上。另外队里没有5号我就投反对。
同意 3 号，他推荐的队我赞成 -> 我同意3号，他推荐的那支队我会投赞成。
不同意 6 号（他推荐的队里有我指认的人），他推荐的队我不赞成 -> 我不同意6号，他推荐的队里有我指认的坏人，那支队我投反对。
不同意 2 号（他指认了我） -> 2号指认我是坏人，我不同意他。
不同意 4 号（我们对同一人判断相反） -> 我不同意4号，我们俩对同一个人的判断正好相反。
3 号若上本轮任务且失败，下轮我说他是坏人 -> 这轮要是3号上了任务而且任务失败，下一轮我就说他是坏人。
本轮任务若成功，5、6 号谁在队里，下轮我说谁是好人 -> 这轮任务要是成功，5号和6号里谁在队里，下一轮我就说谁是好人。
5 号是梅林 -> 我认定5号是梅林。""",
    "en": """I am a Servant; Seat 1 is good (from the quest record of quest 2) -> I'm a Servant. Seat 1 is good, the result of quest 2 shows it.
Seat 3 is evil (from my role) -> I can tell from my role that seat 3 is evil.
Seat 4 is evil (from the vote record of quest 1); I reject any team with seat 4 -> Going by the votes in quest 1, seat 4 is evil, and I'll vote against any team with seat 4 on it.
I suggest 2, 3, 6; I reject any team without seat 5 -> I suggest seats 2, 3 and 6 for this quest. And I'll vote against any team that leaves out seat 5.
I agree with seat 3 and approve their team -> I agree with seat 3, and I'll approve the team they suggested.
I disagree with seat 6 (their team has a player I accuse) and reject their team -> I disagree with seat 6. Their team has a player I call evil, so I'll vote against it.
I disagree with seat 2 (they accused me) -> Seat 2 called me evil, and I disagree with them.
I disagree with seat 4 (we judge a player in opposite ways) -> I disagree with seat 4. We judge the same player in opposite ways.
If seat 3 goes on this quest and it fails, next round I call them evil -> If seat 3 goes on this quest and it fails, I'll call seat 3 evil next round.
If this quest succeeds, next round I call good whoever of seats 5, 6 was on it -> If this quest succeeds, then next round I'll call good whichever of seats 5 and 6 was on it.
Seat 5 is Merlin -> I say seat 5 is Merlin.""",
}


def polish(model: LanguageModel, m: dict, ctx: dict, lang: str, teams: dict | None = None, timeout: float = 5.0) -> str | None:
    """A natural sentence for an agent's message, or None when the template must be shown: nothing to polish, the
    model failed, or the sentence read back through the ears does not give the message's stance.
    `ctx`: `say_context` of the speaker before it spoke (who had spoken, the team size, the leader)."""
    want = message_stance(m)
    if not any(want.values()):
        return None
    zh = lang == "zh"
    system = MOUTH.format(language="Chinese" if zh else "English", seat_form='"3号"' if zh else '"seat 3"',
                          limit="80 个汉字" if zh else "60 words", examples=_EXAMPLES["zh" if zh else "en"])
    try:
        text = model.chat(system, f"The speaker sits in seat {ctx['me']}.\nStatement: {template(m, lang, teams)}", SAY_SCHEMA, "sentence",
                          max_tokens=250, timeout=timeout).get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > (160 if zh else 480):
            return None
        text = " ".join(text.split())
        for kind, outcome in (("fail_evil", r"失败|fail"), ("success_good", r"成功|succe")):  # a promise keeps its condition
            if any(k["promise"] == kind for k in m.get("promise") or []) and not (
                    re.search(r"如果|要是|若|的话|\bif\b", text, re.I) and re.search(outcome, text, re.I)):
                return None
        for j in m.get("judge") or []:  # a cited record keeps its quest number; knowledge by role keeps its mark
            if j.get("basis") == "role_knowledge" and not re.search(r"身份|视角|看到|看见|知道|role|see|saw|know", text, re.I):
                return None
            if j.get("basis") in ("quest", "vote", "proposal") and j.get("quest") and not re.search(
                    f"{j['quest']}|{'一二三四五'[j['quest'] - 1]}|{('first', 'second', 'third', 'fourth', 'fifth')[j['quest'] - 1]}", text, re.I):
                return None
        back = to_stance(understand(model, text, {**ctx, "reply_open": True, "promise": list(talk.PROMISE)}, timeout=timeout),
                         {**ctx, "reply_open": True, "promise": list(talk.PROMISE), "leader": 0})
        hedged = any(x["why"] == "soft" for x in back["ignored"])  # a clause that states nothing (a reason given) is harmless
        return text if not hedged and _same(back["stance"], want, ctx) else None
    except Unavailable:
        return None


def _same(got: dict, want: dict, ctx: dict) -> bool:
    if ctx["evil_talk"]:
        return got["judge"] == want["judge"]
    return all(got[k] == want[k] for k in ("claim", "judge", "team", "veto", "require", "reply", "promise"))
