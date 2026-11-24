"""Human test, speech: what the human says as a stance the agents can hear, and the agents' messages for the page.

The agents talk in the slot protocol (src/comm/protocol.py): five slots, and with the extended configuration
also replies and conditional promises. The human's seat is free (`AvalonGame.free_seats`): no rule limits what it
says or does, and its deeds against its words are only counted in the public audit.
A stance is the part of a human's speech the agents hear, in seats 1-6:

    {"claim": "servant" | "percival" | None,
     "judge": {"3": "evil" | "good"}            (before the assassination: "merlin" | "percival" | "servant")
     "team": [1, 2, 5] | None, "veto": [3], "require": [6],
     "reply": {"2": "endorse" | "dispute"}      (only to players who already spoke in this round)
     "promise": {"4": "fail_evil" | "success_good"}}   (what it will call that player after this round's quest)

`parse` fills a stance from typed text by fixed rules (Chinese and English). Only clear statements are passed on:
hedged ones ("3号有点怪", "maybe 3 is bad") are left out and reported back, so the visitor sees what the agents
heard. A language model can later take the place of `parse` behind the same interface.
"""
from __future__ import annotations

import re

import numpy as np

from src.comm import protocol as P
from src.env import actions as A
from src.env.game import AvalonGame
from src.env.rules import N_PLAYERS, Phase

JUDGE = {"good": P.J_GOOD, "evil": P.J_EVIL, "merlin": P.J_MERLIN, "percival": P.J_PERCIVAL, "servant": P.J_SERVANT}
TABLE_JUDGMENTS, EVIL_TALK_JUDGMENTS = ("good", "evil"), ("merlin", "percival", "servant")
CLAIM = {"servant": P.C_SERVANT, "percival": P.C_PERCIVAL}
REPLY = {"endorse": P.Y_ENDORSE, "dispute": P.Y_DISPUTE}
PROMISE = {"fail_evil": P.K_FAIL_EVIL, "success_good": P.K_SUCCESS_GOOD}


def empty_stance() -> dict:
    return {"claim": None, "judge": {}, "team": None, "veto": [], "require": [], "reply": {}, "promise": {}}


def openings(game: AvalonGame, human: int) -> dict:
    """What the reply and promise slots offer the human, who is about to speak:
    "reply_open": whether the agents at this table hear replies at all;
    "reply": the earlier speakers of this round and the team each suggested (None: no team, nothing to endorse);
    "promise": the promises open in this round ("fail_evil", "success_good");
    "promised": the judgments that the human's promises of the last round call for now (not enforced)."""
    out = {"reply_open": False, "reply": [], "promise": [], "promised": []}
    if game.phase != Phase.DISCUSS:
        return out
    ctx = P.message_context(game, human)
    if ctx["reply"]:
        out["reply_open"] = True
        for seat in game.speakers[:game.speak_idx]:
            team = P.suggested_team(game, int(seat))
            out["reply"].append({"seat": int(seat) + 1, "team": None if team is None else [int(s) + 1 for s in np.flatnonzero(team)]})
    out["promise"] = [name for name, k in PROMISE.items() if ctx["promise_ok"][:, k].any()]
    forced = P.promised_judgments(game, human)
    out["promised"] = [{"seat": int(seat) + 1, "judgment": P.JUDGE_VALUES[v]} for seat, v in zip(P._others(human), forced) if v]
    return out


def message_view(decoded: dict) -> dict:
    """A decoded message (`protocol.decode_message`, seats 0-5) for the page (seats 1-6)."""
    intents = decoded.get("intents", {})
    return {
        "speaker": decoded["speaker"] + 1,
        "claim": decoded.get("claim"),
        "judge": [{"seat": int(s) + 1, "judgment": e["judgment"], "basis": e.get("basis"), "quest": e.get("quest")}
                  for s, e in decoded.get("judgments", {}).items()],
        "team": [s + 1 for s in decoded["team"]] if "team" in decoded else None,
        "veto": [int(s) + 1 for s, v in intents.items() if v == "veto"],
        "require": [int(s) + 1 for s, v in intents.items() if v == "require"],
        "reply": [{"seat": int(s) + 1, "reply": e["reply"], "basis": e.get("basis")} for s, e in decoded.get("replies", {}).items()],
        "promise": [{"seat": int(s) + 1, "promise": v} for s, v in decoded.get("promises", {}).items()],
    }


def clean_stance(data, game: AvalonGame, human: int) -> dict:
    """Validate a stance sent by the browser; raises ValueError with a message for the visitor."""
    if not isinstance(data, dict):
        raise ValueError("发言格式不正确。")
    evil_talk = game.phase == Phase.EVIL_DISCUSS
    others = {s + 1 for s in range(N_PLAYERS) if s != human}

    def seats(value) -> list[int]:
        if not isinstance(value, list) or any(type(s) is not int or s not in others for s in value) or len(set(value)) != len(value):
            raise ValueError("发言里的座位不正确。")
        return sorted(value)

    stance = empty_stance()
    judge = data.get("judge") or {}
    allowed = EVIL_TALK_JUDGMENTS if evil_talk else TABLE_JUDGMENTS
    if not isinstance(judge, dict) or any(not str(s).isdigit() or int(s) not in others or v not in allowed for s, v in judge.items()):
        raise ValueError("发言里的判断不正确。")
    stance["judge"] = {str(int(s)): v for s, v in judge.items()}
    if evil_talk:
        return stance
    if data.get("claim") not in (None, *CLAIM):
        raise ValueError("发言里的身份不正确。")
    stance["claim"] = data.get("claim")
    stance["veto"], stance["require"] = seats(data.get("veto") or []), seats(data.get("require") or [])
    if set(stance["veto"]) & set(stance["require"]):
        raise ValueError("同一名玩家不能既不要又必须带上。")
    team = data.get("team")
    if team:
        if human == game.leader:
            raise ValueError("队长不推荐队伍，直接组队即可。")
        if not isinstance(team, list) or any(type(s) is not int or s not in range(1, N_PLAYERS + 1) for s in team) or len(set(team)) != game.team_size:
            raise ValueError(f"推荐的队伍需要 {game.team_size} 人。")
        stance["team"] = sorted(team)
    open_ = openings(game, human)
    teams = {o["seat"]: o["team"] for o in open_["reply"]}
    reply, promise = data.get("reply") or {}, data.get("promise") or {}
    if not isinstance(reply, dict) or any(not str(s).isdigit() or int(s) not in others or v not in REPLY for s, v in reply.items()):
        raise ValueError("发言里的回复不正确。")
    for s, v in reply.items():
        if int(s) not in teams:
            raise ValueError("只能回复本轮已经发过言的玩家。")
        if v == "endorse" and not teams[int(s)]:
            raise ValueError(f"{int(s)} 号没有推荐队伍，无法表示同意。")
    if not isinstance(promise, dict) or any(not str(s).isdigit() or int(s) not in others or v not in PROMISE for s, v in promise.items()):
        raise ValueError("发言里的承诺不正确。")
    if any(v not in open_["promise"] for v in promise.values()):
        raise ValueError("这一轮不能做这个承诺。")
    stance["reply"] = {str(int(s)): v for s, v in reply.items()}
    stance["promise"] = {str(int(s)): v for s, v in promise.items()}
    return stance


def stance_message(game: AvalonGame, human: int, stance: dict) -> np.ndarray:
    """The protocol message of a (clean) stance."""
    rel = lambda seat: (int(seat) - 1 - human) % N_PLAYERS  # noqa: E731
    judgments, intents, replies, promises = (np.zeros(N_PLAYERS - 1, dtype=np.int64) for _ in range(4))
    for seat, value in stance["judge"].items():
        judgments[rel(seat) - 1] = JUDGE[value]
    for seat in stance["veto"]:
        intents[rel(seat) - 1] = P.I_VETO
    for seat in stance["require"]:
        intents[rel(seat) - 1] = P.I_REQUIRE
    for seat, value in stance["reply"].items():
        replies[rel(seat) - 1] = REPLY[value]
    for seat, value in stance["promise"].items():
        promises[rel(seat) - 1] = PROMISE[value]
    team = 1 + A.TEAM_INDEX[game.team_size][tuple(sorted(rel(s) for s in stance["team"]))] if stance["team"] else 0
    return P.free_message(game, human, CLAIM.get(stance["claim"], P.C_NONE), judgments, team, intents, replies, promises)


# ---------------------------------------------------------------------- typed text -> stance (fixed rules)
_ZH_DIGITS = str.maketrans("一二三四五六１２３４５６", "123456123456")
_NOT_A_SEAT = re.compile(r"第\s*\d\s*(?:轮|次|局|个)?|\d\s*(?:轮|次|局|票|个|人|张|名)|(?:quest|round|mission|game)\s*\d|\d\s*(?:fails?|votes?|rounds?|times?)")
_SEAT = re.compile(r"(?:player|seat|no\.?|#|p)?\s*(?<!\d)([1-6])(?!\d)\s*(?:号位|号玩家|号)?")
_JOIN = re.compile(r"^[\s、,/&和与跟及]*(?:and|or)?[\s、,]*$")
_ONLY_SEATS = re.compile(r"^(?:[\s、/&和与跟及]|and\b|or\b|me\b|myself\b|我|自己|(?:player|seat|no\.?|#|p)?\s*[1-6]\s*(?:号位|号玩家|号)?)+$")
_SENTENCES = re.compile(r"[。；！？.;!?\n]+")
_CLAUSES = re.compile(r"[，,]+|但是|不过|而且|然后|\bbut\b|\bhowever\b")
_SOFT = re.compile(r"有点|有些|可能|也许|或许|好像|似乎|感觉|不确定|不太确定|说不好|轻踩|小踩|怪|maybe|might|perhaps|possibly|"
                   r"not sure|a bit|a little|kind of|kinda|seems?|slightly|i guess|feels? like|probably")
_SELF = re.compile(r"我|自己|\bme\b|\bmyself\b|\bi\b")
_REQUIRE = re.compile(r"(?:没有|没|不带|缺了?|少了?).*(?:不投|不赞成|反对|不同意|否决|不上|不行)|必须(?!是)|一定要|务必带|只投|"
                      r"\b(?:must|need to|have to|has to|require)\b|only (?:approve|accept|vote for)|without .*(?:reject|no\b|veto)|reject .*without")
_VETO = re.compile(r"(?:有|带|含|带着|上了?).*(?:不投|不赞成|反对|不同意|否决)|(?:不同意|不赞成|反对)(?:带|上|让)|不要|别带|不带|不能带|不能上|别上|踢掉|去掉|"
                   r"reject (?:any |every |all )?(?:teams? )?with|\bveto\b|no teams? with|not with|don'?t (?:take|bring|include)|do not (?:take|bring|include)")
_TEAM = re.compile(r"推荐|建议|提议|我想要?上|应该上|可以上|上车|suggest|recommend|propose|\bsend\b|go with|let'?s take|should go")
_CLAIM = re.compile(r"我(?:是|就是|的身份是|跳)\s*(?:一个|个)?\s*(忠臣|派西维尔|派西|梅林)|\bi(?:'m| am)\s+(?:a |the )?(?:loyal )?(servant|percival|merlin)")
_POLAR = [  # checked in order: the negations first
    ("good", re.compile(r"不是(?:坏人|狼|坏的)|不像(?:坏人|狼)|(?:\bnot|n'?t) (?:evil|bad|a spy|sus)")),
    ("evil", re.compile(r"不是好人|不像好人|不可信|不能信|不信|(?:\bnot|n'?t) (?:good|clean)|(?:don'?t|do not|can'?t|cannot) trust")),
    ("evil", re.compile(r"坏人|狼|有问题|是坏的|铁坏|踩|爪牙|反派|evil|bad|\bspy\b|spies|minion|suspicious|\bsus\b")),
    ("good", re.compile(r"好人|金水|可信|信任|我信|相信|保|没问题|干净|good|trust|clean|innocent|\bsafe\b")),
]
_ROLES = [("merlin", re.compile(r"梅林|merlin")), ("percival", re.compile(r"派西|percival")), ("servant", re.compile(r"忠臣|servant"))]
# Replies. About a team ("3号推荐的队我不赞成"): read as a reply whatever else the clause says. Plain agreement
# ("同意3号"): only where the clause judges nobody ("我同意3号是坏人" is a judgment).
_THEIR_TEAM = r"(?:的队|的车|推荐|提的|提议|建议|(?:his|her|their|\w+'s) (?:team|suggestion|pick))"
_NO = r"(?:不赞成|不投|反对|不行|不好|不跟|有问题|不同意|won'?t (?:approve|back)|reject|against|\bbad\b)"
_YES = r"(?:赞成|我投|我跟|同意|可以|没问题|不错|approve|\bback\b|support|\bgood\b|\bfine\b|\bok\b)"
_TEAM_REPLY = [("dispute", re.compile(f"{_THEIR_TEAM}.*{_NO}|{_NO}.*{_THEIR_TEAM}")),
               ("endorse", re.compile(f"{_THEIR_TEAM}.*{_YES}|{_YES}.*{_THEIR_TEAM}"))]
_PLAIN_REPLY = [("dispute", re.compile(r"不同意|不赞同|不认同|(?:说|讲)(?:得|的)不对|\bdisagree\b|(?:don'?t|do not) agree|(?:is|are|'s) wrong")),
                ("endorse", re.compile(r"同意|赞同|认同|(?:说|讲)(?:得|的)对|\bagree\b|(?:is|are|'s) right"))]
# Conditional promises, read from a whole sentence: a condition, a quest outcome and the players it is about.
_COND = re.compile(r"如果|要是|若|假如|只要|万一|的话|\bif\b|\bwhen\b|\bin case\b")
_OUTCOME = [("fail_evil", re.compile(r"失败|翻车|炸了|坏票|\bfail(?:s|ed|ure)?\b")),
            ("success_good", re.compile(r"成功|做成|任务过了|\bsucce(?:ss|ed|eds|eded)\b|\bpass(?:es|ed)?\b"))]


def _mentions(clause: str) -> list[tuple[int, int, int]]:
    """(start, end, seat) of every seat named in the clause; numbers that count rounds or cards are not seats."""
    masked = _NOT_A_SEAT.sub(lambda m: " " * len(m.group()), clause)
    return [(m.start(1), m.end(), int(m.group(1))) for m in _SEAT.finditer(masked)]


def _groups(clause: str, mentions: list[tuple[int, int, int]]) -> list[tuple[int, int, list[int]]]:
    """Seats named together ("3号和5号", "3 and 5") share what is said about them."""
    groups: list[tuple[int, int, list[int]]] = []
    for start, end, seat in mentions:
        if groups and _JOIN.match(clause[groups[-1][1]:start]):
            groups[-1] = (groups[-1][0], end, groups[-1][2] + [seat])
        else:
            groups.append((start, end, [seat]))
    return groups


def _polarity(text: str, table) -> str | None:
    return next((value for value, pattern in table if pattern.search(text)), None)


def parse(text: str, game: AvalonGame, human: int) -> dict:
    """{"stance": ..., "heard": [clauses passed on], "ignored": [{"text": ..., "why": ...}]} for the current speaker."""
    evil_talk = game.phase == Phase.EVIL_DISCUSS
    me, size = human + 1, None if evil_talk else game.team_size
    stance, heard, ignored = empty_stance(), [], []
    open_ = openings(game, human)
    spoke = {o["seat"]: o["team"] for o in open_["reply"]}
    clauses: list[str] = []
    for sentence in _SENTENCES.split(text[:400]):
        low = sentence.translate(_ZH_DIGITS).lower()
        about = sorted({s for _, _, s in _mentions(low) if s != me})
        outcomes = [value for value, pattern in _OUTCOME if pattern.search(low)]
        if not evil_talk and about and outcomes and _COND.search(low):  # a conditional promise
            why = ("promise_both" if len(outcomes) > 1 else "soft" if _SOFT.search(low)
                   else None if outcomes[0] in open_["promise"] else "promise_closed")
            if why is None:
                stance["promise"].update({str(s): outcomes[0] for s in about})
            (heard.append(sentence.strip()) if why is None else ignored.append({"text": sentence.strip(), "why": why}))
            continue
        first = len(clauses)
        for raw in _CLAUSES.split(sentence):
            raw = raw.strip()
            if raw and len(clauses) > first and _ONLY_SEATS.match(raw.translate(_ZH_DIGITS).lower()):
                clauses[-1] += "、" + raw  # "1号，2号，5号": a comma inside a list of seats does not end the clause
            elif raw:
                clauses.append(raw)
    for raw in clauses:
        clause = raw.translate(_ZH_DIGITS).lower()
        mentions = _mentions(clause)
        others = [s for _, _, s in mentions if s != me]
        why = "unknown"
        reply = None
        if others and not evil_talk and not _REQUIRE.search(clause) and not _VETO.search(clause):
            reply = _polarity(clause, _TEAM_REPLY) or (None if _polarity(clause, _POLAR) else _polarity(clause, _PLAIN_REPLY))
        if evil_talk:
            for start, end, seats in _groups(clause, mentions):
                value = _polarity(clause, _ROLES)
                if value and not _SOFT.search(clause):
                    stance["judge"].update({str(s): value for s in seats if s != me})
                    why = None
        elif (claim := _CLAIM.search(clause)):
            role = next(g for g in claim.groups() if g)
            role = {"忠臣": "servant", "派西维尔": "percival", "派西": "percival", "梅林": "merlin"}.get(role, role)
            if role == "merlin":
                why = "merlin_claim"
            else:
                stance["claim"], why = role, None
        elif reply and open_["reply_open"]:
            if _SOFT.search(clause):
                why = "soft"
            elif any(s not in spoke for s in others):
                why = "not_spoken"
            elif reply == "endorse" and any(not spoke[s] for s in others):
                why = "no_team"
            else:
                stance["reply"].update({str(s): reply for s in others})
                why = None
        elif others and _REQUIRE.search(clause):
            stance["require"] = sorted(set(stance["require"]) | set(others))
            why = None
        elif others and _VETO.search(clause):
            stance["veto"] = sorted(set(stance["veto"]) | set(others))
            why = None
        elif _TEAM.search(clause) and mentions:
            team = sorted({s for _, _, s in mentions} | ({me} if _SELF.search(clause) else set()))
            if human == game.leader:
                why = "leader"
            elif len(team) != size:
                why = "size"
            else:
                stance["team"], why = team, None
        elif others:
            if _SOFT.search(clause):
                why = "soft"
            elif _polarity(clause, _ROLES) and not _polarity(clause, _POLAR):
                why = "role_name"
            else:
                groups = _groups(clause, mentions)
                for i, (start, end, seats) in enumerate(groups):
                    after = clause[end:groups[i + 1][0] if i + 1 < len(groups) else len(clause)]
                    before = clause[groups[i - 1][1] if i else 0:start]
                    value = _polarity(after, _POLAR) or _polarity(before, _POLAR)
                    if value:
                        stance["judge"].update({str(s): value for s in seats if s != me})
                        why = None
        (heard.append(raw.strip()) if why is None else ignored.append({"text": raw.strip(), "why": why}))
    stance["require"] = [s for s in stance["require"] if s not in stance["veto"]]
    return {"stance": stance, "heard": heard, "ignored": ignored}
