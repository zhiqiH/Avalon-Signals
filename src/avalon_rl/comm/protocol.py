"""Five-slot structured message protocol.

Timing. Every quest opens with one round of public discussion before its first proposal: starting with
the player to the leader's left and going round the table, each player speaks once and the leader speaks
last; then the leader proposes and everyone votes. After a rejected proposal leadership passes on and the
next leader proposes at once, without discussion. After the third successful quest the two evil players
(Morgana, then the Assassin) each speak once to judge together who is Merlin; then the Assassin strikes.
There is no dialogue: each player speaks once, in order, and cannot address or question anyone.

A message has five slots. Every slot may say nothing, may be false and may be partial, and a message only
concerns the current round (players may change their story in later rounds), but a single message cannot
contradict itself. Heads, in the speaker's relative frame (r = 1..5 are the other players):

    head        values                                                                  slot
    claim       none, Servant, Percival, Merlin                                         (1) role claim
    judge[r]    none, good, evil, Servant, Merlin, Percival, Morgana, Assassin            (2) judgment
    rtype[r]    no basis, quest record, vote record, proposal record, role knowledge       (3) rationale
    rround[r]   quest 1..5 of the cited record                                          (3) rationale
    team        none, or a team of the current quest size (same index as proposals)      (4) team suggestion
    veto[r]     none, "I will reject any team with player r in this round"                (5) action intent

Consistency rules (enforced by masks, sampled in the order above):
- claim and judgments respect the role deck: at most one Merlin, one Percival, one Morgana, one Assassin,
  two Servants; at most two players judged evil (evil, Morgana, Assassin) and four judged good (good, Servant,
  Merlin, Percival), where a claim counts the speaker as good;
- the rationale is the reasoning behind a judgment: it exists only for a judged player and must point the
  same way, citing a resolved quest q:
      quest record     the player was on quest q, which failed (evil) / succeeded (good);
      vote record      the player approved quest q's team that then failed, or rejected one that then
                       succeeded (evil); approved a success or rejected a failure (good);
      proposal record  the player led quest q's team, which failed (evil) / succeeded (good);
      role knowledge   "good"/"evil" with a Merlin claim or no claim (Merlin sees the evil players but not
                       their roles); "Merlin"/"Morgana" with a Percival claim or no claim;
- a team suggestion contains nobody the message judges evil; the leader makes no suggestion (its proposal
  follows its message and may include players it suspects);
- a veto cannot target a player the message judges good or puts in its own suggestion.
Before the assassination only judgments of good players as Servant, Merlin or Percival (with their rationale,
without role knowledge) are open. Talk never binds actions: votes may contradict a veto.

A switched-off slot (`CommConfig`) or a slot silenced at test time can only say nothing. The observation tells
which slots exist in the trained configuration (`slot_open`) but not which are silenced, so a silenced slot looks
exactly like a player choosing to say nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..env import actions as A
from ..env.rules import EVIL_ROLES, N_PLAYERS, N_QUESTS, QUEST_SIZES, Phase

SLOTS = ("claim", "alignment", "rationale", "team", "intent")


@dataclass
class CommConfig:
    enabled: bool = False  # False: the silent game (no discussion phases at all)
    claim: bool = True
    alignment: bool = True  # (2) judgment
    rationale: bool = True
    team: bool = True
    intent: bool = True  # (5) action intent (vetoes)
    silence: list[str] = field(default_factory=list)  # test time: slots forced to say nothing (F -> F-k, F -> S)

    def __post_init__(self) -> None:
        unknown = set(self.silence) - set(SLOTS)
        if unknown:
            raise ValueError(f"unknown slots to silence: {sorted(unknown)}")
        if self.enabled and self.rationale and not self.alignment:
            raise ValueError("the rationale slot needs the judgment slot")

    def is_open(self, slot: str) -> bool:
        """Whether the slot exists in this configuration (what the policy was trained with)."""
        return self.enabled and getattr(self, slot)

    def can_speak(self, slot: str) -> bool:
        return self.is_open(slot) and slot not in self.silence


# ---------------------------------------------------------------------- values and heads
N_OTHERS = N_PLAYERS - 1
CLAIM_VALUES = ("none", "servant", "percival", "merlin")
C_NONE, C_SERVANT, C_PERCIVAL, C_MERLIN = range(len(CLAIM_VALUES))
JUDGE_VALUES = ("none", "good", "evil", "servant", "merlin", "percival", "morgana", "assassin")
J_NONE, J_GOOD, J_EVIL, J_SERVANT, J_MERLIN, J_PERCIVAL, J_MORGANA, J_ASSASSIN = range(len(JUDGE_VALUES))
GOOD_JUDGMENTS = (J_GOOD, J_SERVANT, J_MERLIN, J_PERCIVAL)
EVIL_JUDGMENTS = (J_EVIL, J_MORGANA, J_ASSASSIN)
JUDGE_DIR = np.array([-1, 0, 1, 0, 0, 0, 1, 1])  # direction of a judgment: 0 good, 1 evil
EVIL_TALK_JUDGMENTS = (J_SERVANT, J_MERLIN, J_PERCIVAL)  # before the assassination: which good role
RATIONALE_TYPES = ("none", "quest", "vote", "proposal", "role_knowledge")
R_NONE, R_QUEST, R_VOTE, R_PROPOSAL, R_ROLE = range(len(RATIONALE_TYPES))
RECORD_TYPES = (R_QUEST, R_VOTE, R_PROPOSAL)
VETO_VALUES = ("none", "veto")

H_CLAIM = 0
H_JUDGE = 1  # judge[r] is head H_JUDGE + r - 1
H_RTYPE = H_JUDGE + N_OTHERS
H_RROUND = H_RTYPE + N_OTHERS
H_TEAM = H_RROUND + N_OTHERS
H_VETO = H_TEAM + 1
N_MSG_HEADS = H_VETO + N_OTHERS

MSG_HEAD_SIZES = (
    (len(CLAIM_VALUES),)
    + (len(JUDGE_VALUES),) * N_OTHERS
    + (len(RATIONALE_TYPES),) * N_OTHERS
    + (N_QUESTS,) * N_OTHERS
    + (1 + A.MAX_TEAMS,)
    + (len(VETO_VALUES),) * N_OTHERS
)
MSG_HEAD_OFFSETS = tuple(int(x) for x in np.cumsum((0,) + MSG_HEAD_SIZES[:-1]))
N_MSG_LOGITS = sum(MSG_HEAD_SIZES)
# Sampling order. Stage A (claim, judgments) is what the later heads are conditioned on.
STAGE_A = (H_CLAIM, *range(H_JUDGE, H_JUDGE + N_OTHERS))
STAGE_B = (*range(H_RTYPE, H_RTYPE + N_OTHERS), *range(H_RROUND, H_RROUND + N_OTHERS), H_TEAM,
           *range(H_VETO, H_VETO + N_OTHERS))
COND_DIM = len(CLAIM_VALUES) + N_OTHERS * len(JUDGE_VALUES)

# TEAM_MEMBERS[k][v, s]: team-head value v (team v-1 of size k, 0 = none) contains relative seat s.
TEAM_MEMBERS = {}
for _k, _teams in A.TEAMS_BY_SIZE.items():
    TEAM_MEMBERS[_k] = np.zeros((1 + A.MAX_TEAMS, N_PLAYERS), dtype=bool)
    for _i, _team in enumerate(_teams):
        TEAM_MEMBERS[_k][1 + _i, list(_team)] = True


def head_slice(mask: np.ndarray, h: int) -> np.ndarray:
    return mask[..., MSG_HEAD_OFFSETS[h]:MSG_HEAD_OFFSETS[h] + MSG_HEAD_SIZES[h]]


# ---------------------------------------------------------------------- legality
def _others(speaker: int) -> np.ndarray:
    """Absolute seats of the speaker's relative seats 1..5."""
    return (speaker + np.arange(1, N_PLAYERS)) % N_PLAYERS


def evidence(game, speaker: int) -> np.ndarray:
    """evidence[r-1, k, d, q]: record type RECORD_TYPES[k] about player r in resolved quest q points to d (0 good, 1 evil)."""
    ev = np.zeros((N_OTHERS, len(RECORD_TYPES), 2, N_QUESTS), dtype=bool)
    q = np.flatnonzero(game.quest_result >= 0)
    if len(q) == 0:
        return ev
    others = _others(speaker)
    failed = game.quest_result[q] == 1
    attempt = game.prop_approved[q].argmax(axis=1)  # the proposal that went on the quest
    on_team = game.quest_team[q][:, others].T
    approved = game.prop_votes[q, attempt][:, others].T
    led = game.prop_leader[q, attempt][None, :] == others[:, None]
    wrong_vote = approved == failed[None, :]  # approved a failure or rejected a success
    for k, (hit, evil) in enumerate([(on_team, np.broadcast_to(failed, on_team.shape)),
                                     (np.ones_like(on_team), wrong_vote),
                                     (led, np.broadcast_to(failed, led.shape))]):
        ev[:, k, 1, q] = hit & evil
        ev[:, k, 0, q] = hit & ~evil
    return ev


def message_context(game, speaker: int) -> dict[str, np.ndarray]:
    """Everything the masks need about the game, for one speaker (stack several with `stack_contexts`)."""
    cfg = game.comm
    evil_talk = game.phase == Phase.EVIL_DISCUSS
    claim_ok = np.zeros(len(CLAIM_VALUES), dtype=bool)
    judge_ok = np.zeros((N_OTHERS, len(JUDGE_VALUES)), dtype=bool)
    team_ok = np.zeros(1 + A.MAX_TEAMS, dtype=bool)
    claim_ok[0] = judge_ok[:, 0] = team_ok[0] = True
    size = QUEST_SIZES[min(game.quest, N_QUESTS - 1)]
    if evil_talk:
        if cfg.can_speak("alignment"):
            good = ~np.isin(game.roles[_others(speaker)], list(EVIL_ROLES))
            judge_ok[np.ix_(good, EVIL_TALK_JUDGMENTS)] = True
    else:
        claim_ok[1:] = cfg.can_speak("claim")
        judge_ok[:, 1:] = cfg.can_speak("alignment")
        if cfg.can_speak("team") and speaker != game.leader:
            team_ok[1:1 + len(A.TEAMS_BY_SIZE[size])] = True
    rationale = cfg.can_speak("rationale")
    return {
        "claim_ok": claim_ok, "judge_ok": judge_ok, "team_ok": team_ok, "team_members": TEAM_MEMBERS[size],
        "evidence": evidence(game, speaker) if rationale else np.zeros((N_OTHERS, 3, 2, N_QUESTS), dtype=bool),
        "rationale": np.array(rationale), "role_knowledge": np.array(rationale and not evil_talk),
        "veto": np.array(cfg.can_speak("intent") and not evil_talk),
    }


def stack_contexts(contexts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.stack([c[k] for c in contexts]) for k in contexts[0]}


def head_mask(h: int, msg: np.ndarray, ctx: dict[str, np.ndarray]) -> np.ndarray:
    """Legal values of head h (batch, size) given the heads already chosen in `msg` (batch, N_MSG_HEADS)."""
    n = len(msg)
    rows = np.arange(n)
    claim = msg[:, H_CLAIM]
    if h == H_CLAIM:
        return ctx["claim_ok"].copy()
    if h < H_RTYPE:  # judgment of relative seat r = h - H_JUDGE + 1, limited by the role deck
        prev = msg[:, H_JUDGE:h]
        count = lambda v: (prev == v).sum(axis=1)  # noqa: E731
        n_good = np.isin(prev, GOOD_JUDGMENTS).sum(axis=1) + (claim > C_NONE)
        n_evil = np.isin(prev, EVIL_JUDGMENTS).sum(axis=1)
        ok = ctx["judge_ok"][:, h - H_JUDGE].copy()
        ok[:, list(GOOD_JUDGMENTS)] &= (n_good < 4)[:, None]
        ok[:, list(EVIL_JUDGMENTS)] &= (n_evil < 2)[:, None]
        ok[:, J_SERVANT] &= count(J_SERVANT) + (claim == C_SERVANT) < 2
        ok[:, J_MERLIN] &= count(J_MERLIN) + (claim == C_MERLIN) < 1
        ok[:, J_PERCIVAL] &= count(J_PERCIVAL) + (claim == C_PERCIVAL) < 1
        ok[:, J_MORGANA] &= count(J_MORGANA) < 1
        ok[:, J_ASSASSIN] &= count(J_ASSASSIN) < 1
        return ok
    if h < H_RROUND:  # rationale type: only for a judged player, pointing the judgment's way
        r = h - H_RTYPE
        judge = msg[:, H_JUDGE + r]
        judged = (judge > J_NONE) & ctx["rationale"]
        direction = JUDGE_DIR[judge.clip(0)].clip(0)
        ok = np.zeros((n, len(RATIONALE_TYPES)), dtype=bool)
        ok[:, R_NONE] = judged
        ok[:, R_QUEST:R_PROPOSAL + 1] = judged[:, None] & ctx["evidence"][rows, r, :, direction, :].any(axis=-1)
        merlin_sight = np.isin(judge, (J_GOOD, J_EVIL)) & np.isin(claim, (C_NONE, C_MERLIN))
        percival_sight = np.isin(judge, (J_MERLIN, J_MORGANA)) & np.isin(claim, (C_NONE, C_PERCIVAL))
        ok[:, R_ROLE] = judged & ctx["role_knowledge"] & (merlin_sight | percival_sight)
        return ok
    if h < H_TEAM:  # cited quest of a record rationale
        r = h - H_RROUND
        rtype = msg[:, H_RTYPE + r]
        cites = np.isin(rtype, RECORD_TYPES)
        direction = JUDGE_DIR[msg[:, H_JUDGE + r].clip(0)].clip(0)
        return ctx["evidence"][rows, r, (rtype - R_QUEST).clip(0, 2), direction, :] & cites[:, None]
    accused = np.isin(msg[:, H_JUDGE:H_JUDGE + N_OTHERS], EVIL_JUDGMENTS)  # relative seats 1..5
    if h == H_TEAM:
        return ctx["team_ok"] & ~(ctx["team_members"][:, :, 1:] & accused[:, None, :]).any(axis=-1)
    r = h - H_VETO  # veto of relative seat r + 1
    team = msg[:, H_TEAM]
    in_team = (team > 0) & ctx["team_members"][rows, team.clip(0), r + 1]
    ok = np.zeros((n, len(VETO_VALUES)), dtype=bool)
    ok[:, 0] = True
    ok[:, 1] = ctx["veto"] & ~np.isin(msg[:, H_JUDGE + r], GOOD_JUDGMENTS) & ~in_team
    return ok


def sample_messages(ctx: dict[str, np.ndarray], rng: np.random.Generator, first: bool = False) -> np.ndarray:
    """Messages for a batch of contexts: each head uniformly among its legal values (or its first legal value)."""
    n = len(ctx["claim_ok"])
    msg = np.full((n, N_MSG_HEADS), -1, dtype=np.int64)
    for h in STAGE_A + STAGE_B:
        ok = head_mask(h, msg, ctx)
        score = np.where(ok, 1.0 if first else rng.random(ok.shape), -1.0)
        msg[:, h] = np.where(ok.any(axis=1), score.argmax(axis=1), -1)
    return msg.astype(np.int8)


def forced_message(ctx: dict[str, np.ndarray]) -> np.ndarray | None:
    """The only legal message of one speaker when no slot can say anything (all closed or silenced)."""
    if ctx["claim_ok"][1:].any() or ctx["judge_ok"][:, 1:].any() or ctx["team_ok"][1:].any() or ctx["veto"]:
        return None
    msg = np.full(N_MSG_HEADS, -1, dtype=np.int8)
    msg[[H_CLAIM, *range(H_JUDGE, H_JUDGE + N_OTHERS), H_TEAM, *range(H_VETO, H_VETO + N_OTHERS)]] = 0
    return msg


def check_messages(msgs: np.ndarray, ctx: dict[str, np.ndarray]) -> None:
    """Raise ValueError unless every message of the batch is legal for its context."""
    msgs = np.asarray(msgs, dtype=np.int64)
    if msgs.ndim != 2 or msgs.shape[1] != N_MSG_HEADS:
        raise ValueError(f"messages must have shape (batch, {N_MSG_HEADS}), got {msgs.shape}")
    rows = np.arange(len(msgs))
    for h in STAGE_A + STAGE_B:
        ok = head_mask(h, msgs, ctx)
        v = msgs[:, h]
        legal = np.where(ok.any(axis=1), (v >= 0) & (v < ok.shape[1]) & ok[rows, v.clip(0, ok.shape[1] - 1)], v == -1)
        if not legal.all():
            bad = int(np.flatnonzero(~legal)[0])
            raise ValueError(f"illegal value {v[bad]} for message head {h} (message {bad} of the batch)")


def check_message(game, speaker: int, msg: np.ndarray) -> None:
    """Raise ValueError unless `msg` is a legal message of `speaker` now."""
    check_messages(np.asarray(msg)[None], stack_contexts([message_context(game, speaker)]))


# ---------------------------------------------------------------------- observation features
# Per message, in absolute target seats inside the game; rotated into the observer's frame by `encode_comm`.
_MSG_FIELDS = [
    ("spoke", 1),
    ("claim", len(CLAIM_VALUES) - 1),
    ("judge", N_PLAYERS * (len(JUDGE_VALUES) - 1)),
    ("rtype", N_PLAYERS * (len(RATIONALE_TYPES) - 1)),
    ("rround", N_PLAYERS * N_QUESTS),
    ("team_flag", 1),
    ("team", N_PLAYERS),
    ("veto", N_PLAYERS),
]
MSG_LAYOUT: dict[str, slice] = {}
_o = 0
for _name, _size in _MSG_FIELDS:
    MSG_LAYOUT[_name] = slice(_o, _o + _size)
    _o += _size
MSG_FEAT_DIM = _o
_SEAT_FIELDS = {"judge": len(JUDGE_VALUES) - 1, "rtype": len(RATIONALE_TYPES) - 1, "rround": N_QUESTS, "team": 1, "veto": 1}
N_MSG_SLOTS = N_QUESTS * N_PLAYERS + N_PLAYERS  # one discussion per quest x speaker, then the evil talk by speaker

_COMM_FIELDS = [
    ("slot_open", len(SLOTS)),
    ("discussion", 2),  # in DISCUSS, in EVIL_DISCUSS
    ("spoken", N_PLAYERS),  # who has already spoken in the current discussion
]
COMM_LAYOUT: dict[str, slice] = {}
_o = 0
for _name, _size in _COMM_FIELDS:
    COMM_LAYOUT[_name] = slice(_o, _o + _size)
    _o += _size
COMM_GLOBAL_DIM = _o
COMM_LAYOUT["messages"] = slice(COMM_GLOBAL_DIM, COMM_GLOBAL_DIM + N_MSG_SLOTS * MSG_FEAT_DIM)
COMM_DIM = COMM_LAYOUT["messages"].stop


def _feature_perm(observer: int) -> np.ndarray:
    """idx[j] = absolute-frame feature behind the observer-relative feature j."""
    idx: list[int] = []
    for name, size in _MSG_FIELDS:
        start, per_seat = MSG_LAYOUT[name].start, _SEAT_FIELDS.get(name)
        if per_seat is None:
            idx += range(start, start + size)
            continue
        for rel in range(N_PLAYERS):
            seat = (observer + rel) % N_PLAYERS
            idx += range(start + seat * per_seat, start + (seat + 1) * per_seat)
    return np.array(idx)


_FEATURE_PERM = [_feature_perm(o) for o in range(N_PLAYERS)]


def message_features(speaker: int, msg: np.ndarray, team_size: int) -> np.ndarray:
    """Absolute-frame features of one message."""
    f = np.zeros(MSG_FEAT_DIM, dtype=np.int8)
    f[MSG_LAYOUT["spoke"]] = 1
    if msg[H_CLAIM] > 0:
        f[MSG_LAYOUT["claim"].start + msg[H_CLAIM] - 1] = 1
    for r, seat in enumerate(_others(speaker)):
        if msg[H_JUDGE + r] > 0:
            f[MSG_LAYOUT["judge"].start + _SEAT_FIELDS["judge"] * seat + msg[H_JUDGE + r] - 1] = 1
        if msg[H_RTYPE + r] > 0:
            f[MSG_LAYOUT["rtype"].start + _SEAT_FIELDS["rtype"] * seat + msg[H_RTYPE + r] - 1] = 1
        if msg[H_RROUND + r] >= 0:
            f[MSG_LAYOUT["rround"].start + N_QUESTS * seat + msg[H_RROUND + r]] = 1
        if msg[H_VETO + r] > 0:
            f[MSG_LAYOUT["veto"].start + seat] = 1
    if msg[H_TEAM] > 0:
        f[MSG_LAYOUT["team_flag"]] = 1
        f[MSG_LAYOUT["team"].start + np.array(A.decode_team(speaker, msg[H_TEAM] - 1, team_size))] = 1
    return f


def encode_comm(game, player: int) -> np.ndarray:
    """Communication part of the observation (int8, length COMM_DIM), in the player's relative frame."""
    out = np.zeros(COMM_DIM, dtype=np.int8)
    cfg = game.comm
    out[COMM_LAYOUT["slot_open"]] = [cfg.is_open(s) for s in SLOTS]
    if not cfg.enabled:
        return out
    perm = (player + np.arange(N_PLAYERS)) % N_PLAYERS
    if game.in_discussion:
        out[COMM_LAYOUT["discussion"].start + (game.phase == Phase.EVIL_DISCUSS)] = 1
        for seat in game.speakers[:game.speak_idx]:
            out[COMM_LAYOUT["spoken"].start + (seat - player) % N_PLAYERS] = 1
    slots = np.concatenate([game.msg_feat[:, perm].reshape(-1, MSG_FEAT_DIM), game.evil_feat[perm]])
    out[COMM_LAYOUT["messages"]] = slots[:, _FEATURE_PERM[player]].reshape(-1)
    return out


def decode_message(msg: np.ndarray, speaker: int, team_size: int) -> dict:
    """Readable form of a message in absolute seats (for game records)."""
    out: dict = {"speaker": speaker}
    if msg[H_CLAIM] > 0:
        out["claim"] = CLAIM_VALUES[msg[H_CLAIM]]
    judged = {}
    for r, seat in enumerate(_others(speaker)):
        if msg[H_JUDGE + r] > 0:
            entry = {"judgment": JUDGE_VALUES[msg[H_JUDGE + r]]}
            if msg[H_RTYPE + r] > 0:
                entry["basis"] = RATIONALE_TYPES[msg[H_RTYPE + r]]
                if msg[H_RROUND + r] >= 0:
                    entry["quest"] = int(msg[H_RROUND + r]) + 1
            judged[int(seat)] = entry
    if judged:
        out["judgments"] = judged
    if msg[H_TEAM] > 0:
        out["team"] = [int(s) for s in A.decode_team(speaker, msg[H_TEAM] - 1, team_size)]
    vetoes = [int(seat) for r, seat in enumerate(_others(speaker)) if msg[H_VETO + r] > 0]
    if vetoes:
        out["veto"] = vetoes
    return out
