"""Five-slot structured message protocol.

After every proposal, each player in turn (the leader first, then to the left) sends one public message
before the vote; later speakers see earlier messages. After the third successful quest the two evil
players each send one message before the assassination, using only the alignment slot, read as
"this player is / is not Merlin".

A message is an int array with one entry per message head, written in the speaker's relative frame
(seat 0 = speaker, r = r places to the left):

    head        values                                                              slot
    claim       none, Servant, Percival, Merlin                                     (1) role claim
    align[r]    none, good, evil (before the assassination: none, Merlin, not Merlin)   (2) alignment
    rtype[r]    no basis, quest record, vote record, proposal record, role knowledge   (3) rationale
    rround[r]   quest 1..5 of the cited record                                      (3) rationale
    team        none, or a team of the current quest size (proposal team index + 1)  (4) team suggestion
    intent      none, approve, reject                                               (5) action intent
    name        none, or the relative seat of a player who has not spoken yet       (5) action intent

with r = 1..5 for the other players. Value 0 of every head means "say nothing". rtype[r] is used only
when align[r] judges player r, and rround[r] only when rtype[r] cites a public record; unused heads hold
-1. A public record must be a real event: player r was on quest q (quest record), quest q had a vote
(vote record), or player r led or was on a proposed team in quest q (proposal record). The leader's team
suggestion is fixed to its own proposal whenever the team slot exists, even when it is silenced: it only
restates the proposal, and keeping it avoids an observation never seen in training.

A switched-off slot (`CommConfig`) or a slot silenced at test time can only say nothing. The observation
tells which slots exist in the trained configuration (`slot_open`) but not which are silenced, so a
silenced slot looks exactly like a player choosing to say nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..env import actions as A
from ..env.rules import EVIL_ROLES, MAX_PROPOSALS, N_PLAYERS, N_QUESTS, Phase

SLOTS = ("claim", "alignment", "rationale", "team", "intent")


@dataclass
class CommConfig:
    enabled: bool = False  # False: the silent game (no discussion phases at all)
    claim: bool = True
    alignment: bool = True
    rationale: bool = True
    team: bool = True
    intent: bool = True
    silence: list[str] = field(default_factory=list)  # test time: slots forced to say nothing (F -> F-k, F -> S)

    def __post_init__(self) -> None:
        unknown = set(self.silence) - set(SLOTS)
        if unknown:
            raise ValueError(f"unknown slots to silence: {sorted(unknown)}")
        if self.enabled and self.rationale and not self.alignment:
            raise ValueError("the rationale slot needs the alignment slot")

    def is_open(self, slot: str) -> bool:
        """Whether the slot exists in this configuration (what the policy was trained with)."""
        return self.enabled and getattr(self, slot)

    def can_speak(self, slot: str) -> bool:
        return self.is_open(slot) and slot not in self.silence


# ---------------------------------------------------------------------- message heads
N_OTHERS = N_PLAYERS - 1
CLAIM_VALUES = ("none", "servant", "percival", "merlin")
ALIGN_VALUES = ("none", "good", "evil")
EVIL_ALIGN_VALUES = ("none", "merlin", "not_merlin")
RATIONALE_TYPES = ("none", "quest", "vote", "proposal", "role_knowledge")
RECORD_TYPES = (1, 2, 3)  # rationale types that cite a quest
INTENT_VALUES = ("none", "approve", "reject")

H_CLAIM = 0
H_ALIGN = 1  # align[r] is head H_ALIGN + r - 1
H_RTYPE = H_ALIGN + N_OTHERS
H_RROUND = H_RTYPE + N_OTHERS
H_TEAM = H_RROUND + N_OTHERS
H_INTENT = H_TEAM + 1
H_NAME = H_INTENT + 1
N_MSG_HEADS = H_NAME + 1

MSG_HEAD_SIZES = (
    (len(CLAIM_VALUES),)
    + (len(ALIGN_VALUES),) * N_OTHERS
    + (len(RATIONALE_TYPES),) * N_OTHERS
    + (N_QUESTS,) * N_OTHERS
    + (1 + A.MAX_TEAMS, len(INTENT_VALUES), 1 + N_OTHERS)
)
MSG_HEAD_SLOT = ("claim",) + ("alignment",) * N_OTHERS + ("rationale",) * (2 * N_OTHERS) + ("team", "intent", "intent")
MSG_HEAD_OFFSETS = tuple(int(x) for x in np.cumsum((0,) + MSG_HEAD_SIZES[:-1]))
N_MSG_LOGITS = sum(MSG_HEAD_SIZES)
STATIC_HEADS = (H_CLAIM, *range(H_ALIGN, H_ALIGN + N_OTHERS), H_TEAM, H_INTENT, H_NAME)


def head_mask(mask: np.ndarray, h: int) -> np.ndarray:
    return mask[MSG_HEAD_OFFSETS[h]:MSG_HEAD_OFFSETS[h] + MSG_HEAD_SIZES[h]]


def silent_message() -> np.ndarray:
    msg = np.full(N_MSG_HEADS, -1, dtype=np.int8)
    msg[list(STATIC_HEADS)] = 0
    return msg


# ---------------------------------------------------------------------- legality
def _others(speaker: int) -> np.ndarray:
    """Absolute seats of the speaker's relative seats 1..5."""
    return (speaker + np.arange(1, N_PLAYERS)) % N_PLAYERS


def rationale_validity(game, speaker: int) -> tuple[np.ndarray, np.ndarray]:
    """(rtype_valid [5, 5], rround_valid [5, 5, quests]) for the speaker's relative seats 1..5.

    rround_valid[r-1, t, q]: rationale type t about player r may cite quest q. rtype_valid[r-1, t] says
    whether type t is available at all (a record type needs at least one citable quest).
    """
    rtype_valid = np.zeros((N_OTHERS, len(RATIONALE_TYPES)), dtype=bool)
    rround_valid = np.zeros((N_OTHERS, len(RATIONALE_TYPES), N_QUESTS), dtype=bool)
    if not game.comm.can_speak("rationale") or game.phase != Phase.DISCUSS:
        return rtype_valid, rround_valid
    others = _others(speaker)
    played = game.quest_result >= 0
    rround_valid[:, 1] = played[None, :] & game.quest_team[:, others].T
    rround_valid[:, 2] = game.prop_voted.any(axis=1)[None, :]
    exists = game.prop_leader >= 0
    led = game.prop_leader[None] == others[:, None, None]
    on_team = game.prop_team[:, :, others].transpose(2, 0, 1)
    rround_valid[:, 3] = ((led | on_team) & exists[None]).any(axis=2)
    rtype_valid[:, 0] = rtype_valid[:, 4] = True
    rtype_valid[:, list(RECORD_TYPES)] = rround_valid[:, list(RECORD_TYPES)].any(axis=2)
    return rtype_valid, rround_valid


def message_mask(game, speaker: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legal values of the static heads (rationale heads left empty) plus the rationale validity tables."""
    cfg = game.comm
    mask = np.zeros(N_MSG_LOGITS, dtype=bool)
    for h in STATIC_HEADS:
        head_mask(mask, h)[0] = True
    if game.phase == Phase.EVIL_DISCUSS:
        if cfg.can_speak("alignment"):
            known_evil = np.isin(game.roles, list(EVIL_ROLES))
            for r, seat in enumerate(_others(speaker), start=1):
                head_mask(mask, H_ALIGN + r - 1)[1:] = not known_evil[seat]
        return mask, *rationale_validity(game, speaker)

    if cfg.can_speak("claim"):
        head_mask(mask, H_CLAIM)[1:] = True
    if cfg.can_speak("alignment"):
        for r in range(1, N_PLAYERS):
            head_mask(mask, H_ALIGN + r - 1)[1:] = True
    team = head_mask(mask, H_TEAM)
    if cfg.is_open("team") and speaker == game.leader:  # the leader's suggestion is its own proposal
        team[:] = False
        team[1 + A.team_action(speaker, np.flatnonzero(game.team))] = True
    elif cfg.can_speak("team"):
        team[1:1 + len(A.TEAMS_BY_SIZE[game.team_size])] = True
    if cfg.can_speak("intent"):
        head_mask(mask, H_INTENT)[1:] = True
        name = head_mask(mask, H_NAME)
        for seat in game.speakers[game.speak_idx + 1:]:
            name[(seat - speaker) % N_PLAYERS] = True
    return mask, *rationale_validity(game, speaker)


def dependent_masks(msg: np.ndarray, rtype_valid: np.ndarray, rround_valid: np.ndarray, kind: str, r: int) -> np.ndarray:
    """Legal values of rtype[r] (kind='rtype') or rround[r] (kind='rround') given the earlier heads of `msg`."""
    if kind == "rtype":
        return rtype_valid[r - 1] if msg[H_ALIGN + r - 1] > 0 else np.zeros(len(RATIONALE_TYPES), dtype=bool)
    t = msg[H_RTYPE + r - 1]
    return rround_valid[r - 1, t] if t in RECORD_TYPES else np.zeros(N_QUESTS, dtype=bool)


def forced_message(mask: np.ndarray) -> np.ndarray | None:
    """The only legal message when every static head has a single legal value (e.g. all slots closed)."""
    msg = silent_message()
    for h in STATIC_HEADS:
        legal = np.flatnonzero(head_mask(mask, h))
        if len(legal) != 1:
            return None
        msg[h] = legal[0]
    return msg  # every alignment head says nothing, so no rationale is possible


def sample_message(mask, rtype_valid, rround_valid, rng: np.random.Generator, first: bool = False) -> np.ndarray:
    """Sample uniformly among legal values head by head (or take the first legal value of each head)."""
    pick = (lambda legal: legal[0]) if first else (lambda legal: rng.choice(legal))
    msg = np.full(N_MSG_HEADS, -1, dtype=np.int8)
    for h in STATIC_HEADS:
        msg[h] = pick(np.flatnonzero(head_mask(mask, h)))
    for kind, base in (("rtype", H_RTYPE), ("rround", H_RROUND)):
        for r in range(1, N_PLAYERS):
            legal = np.flatnonzero(dependent_masks(msg, rtype_valid, rround_valid, kind, r))
            if len(legal):
                msg[base + r - 1] = pick(legal)
    return msg


def check_message(game, speaker: int, msg: np.ndarray, masks: tuple | None = None) -> None:
    """Raise ValueError unless `msg` is legal; `masks` is the speaker's `message_mask` if already computed."""
    msg = np.asarray(msg)
    if msg.shape != (N_MSG_HEADS,):
        raise ValueError(f"a message has {N_MSG_HEADS} heads, got shape {msg.shape}")
    mask, rtype_valid, rround_valid = masks if masks is not None else message_mask(game, speaker)
    for h in STATIC_HEADS:
        if not (0 <= msg[h] < MSG_HEAD_SIZES[h]) or not head_mask(mask, h)[msg[h]]:
            raise ValueError(f"illegal value {msg[h]} for message head {h} of player {speaker}")
    for kind, base in (("rtype", H_RTYPE), ("rround", H_RROUND)):
        for r in range(1, N_PLAYERS):
            legal = dependent_masks(msg, rtype_valid, rround_valid, kind, r)
            v = msg[base + r - 1]
            ok = (v == -1) if not legal.any() else (0 <= v < len(legal) and legal[v])
            if not ok:
                raise ValueError(f"illegal {kind}[{r}] = {v} for player {speaker}")


# ---------------------------------------------------------------------- observation features
# Per message slot (quest, attempt, speaker), in absolute target seats inside the game and rotated into the
# observer's relative frame by `encode_comm`.
_MSG_FIELDS = [
    ("spoke", 1),
    ("claim", len(CLAIM_VALUES) - 1),
    ("align", N_PLAYERS * 2),  # per target: good, evil
    ("rtype", N_PLAYERS * (len(RATIONALE_TYPES) - 1)),  # per target: quest, vote, proposal, role knowledge
    ("rround", N_PLAYERS * N_QUESTS),  # per target: cited quest
    ("team_flag", 1),
    ("team", N_PLAYERS),
    ("intent", len(INTENT_VALUES) - 1),
    ("name", N_PLAYERS),
]
MSG_LAYOUT: dict[str, slice] = {}
_o = 0
for _name, _size in _MSG_FIELDS:
    MSG_LAYOUT[_name] = slice(_o, _o + _size)
    _o += _size
MSG_FEAT_DIM = _o
_SEAT_FIELDS = {"align": 2, "rtype": 4, "rround": N_QUESTS, "team": 1, "name": 1}  # features per target seat
N_MSG_SLOTS = N_QUESTS * MAX_PROPOSALS * N_PLAYERS

_COMM_FIELDS = [
    ("slot_open", len(SLOTS)),
    ("discussion", 2),  # in DISCUSS, in EVIL_DISCUSS
    ("spoken", N_PLAYERS),  # who has already spoken in the current discussion
    ("evil_talk", N_PLAYERS * N_PLAYERS * 2),  # speaker x target x (Merlin, not Merlin)
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
        start = MSG_LAYOUT[name].start
        per_seat = _SEAT_FIELDS.get(name)
        if per_seat is None:
            idx += range(start, start + size)
            continue
        for rel in range(N_PLAYERS):
            seat = (observer + rel) % N_PLAYERS
            idx += range(start + seat * per_seat, start + (seat + 1) * per_seat)
    return np.array(idx)


_FEATURE_PERM = [_feature_perm(o) for o in range(N_PLAYERS)]


def message_features(game, speaker: int, msg: np.ndarray) -> np.ndarray:
    """Absolute-frame features of a DISCUSS message."""
    f = np.zeros(MSG_FEAT_DIM, dtype=np.int8)
    f[MSG_LAYOUT["spoke"]] = 1
    if msg[H_CLAIM] > 0:
        f[MSG_LAYOUT["claim"].start + msg[H_CLAIM] - 1] = 1
    for r, seat in enumerate(_others(speaker), start=1):
        if msg[H_ALIGN + r - 1] > 0:
            f[MSG_LAYOUT["align"].start + 2 * seat + msg[H_ALIGN + r - 1] - 1] = 1
        if msg[H_RTYPE + r - 1] > 0:
            f[MSG_LAYOUT["rtype"].start + 4 * seat + msg[H_RTYPE + r - 1] - 1] = 1
        if msg[H_RROUND + r - 1] >= 0:
            f[MSG_LAYOUT["rround"].start + N_QUESTS * seat + msg[H_RROUND + r - 1]] = 1
    if msg[H_TEAM] > 0:
        f[MSG_LAYOUT["team_flag"]] = 1
        f[MSG_LAYOUT["team"].start + np.array(A.decode_team(speaker, msg[H_TEAM] - 1, game.team_size))] = 1
    if msg[H_INTENT] > 0:
        f[MSG_LAYOUT["intent"].start + msg[H_INTENT] - 1] = 1
    if msg[H_NAME] > 0:
        f[MSG_LAYOUT["name"].start + (speaker + msg[H_NAME]) % N_PLAYERS] = 1
    return f


def encode_comm(game, player: int) -> np.ndarray:
    """Communication part of the observation (int8, length COMM_DIM), in the player's relative frame."""
    out = np.zeros(COMM_DIM, dtype=np.int8)
    cfg = game.comm
    out[COMM_LAYOUT["slot_open"]] = [cfg.is_open(s) for s in SLOTS]
    if not cfg.enabled:
        return out
    perm = (player + np.arange(N_PLAYERS)) % N_PLAYERS
    if game.phase in (Phase.DISCUSS, Phase.EVIL_DISCUSS):
        out[COMM_LAYOUT["discussion"].start + (game.phase == Phase.EVIL_DISCUSS)] = 1
        for seat in game.speakers[:game.speak_idx]:
            out[COMM_LAYOUT["spoken"].start + (seat - player) % N_PLAYERS] = 1
    out[COMM_LAYOUT["evil_talk"]] = game.evil_feat[perm][:, perm].reshape(-1)
    out[COMM_LAYOUT["messages"]] = game.msg_feat[:, :, perm][..., _FEATURE_PERM[player]].reshape(-1)
    return out


def decode_message(msg: np.ndarray, speaker: int, team_size: int, evil_talk: bool = False) -> dict:
    """Readable form of a message in absolute seats (for game records)."""
    align_values = EVIL_ALIGN_VALUES if evil_talk else ALIGN_VALUES
    out: dict = {"speaker": speaker}
    if msg[H_CLAIM] > 0:
        out["claim"] = CLAIM_VALUES[msg[H_CLAIM]]
    judged = {}
    for r, seat in enumerate(_others(speaker), start=1):
        v = msg[H_ALIGN + r - 1]
        if v > 0:
            entry = {"judgment": align_values[v]}
            t = msg[H_RTYPE + r - 1]
            if t > 0:
                entry["basis"] = RATIONALE_TYPES[t]
                if msg[H_RROUND + r - 1] >= 0:
                    entry["quest"] = int(msg[H_RROUND + r - 1]) + 1
            judged[int(seat)] = entry
    if judged:
        out["alignment"] = judged
    if msg[H_TEAM] > 0:
        out["team"] = [int(s) for s in A.decode_team(speaker, msg[H_TEAM] - 1, team_size)]
    if msg[H_INTENT] > 0:
        out["intent"] = INTENT_VALUES[msg[H_INTENT]]
    if msg[H_NAME] > 0:
        out["name"] = int((speaker + msg[H_NAME]) % N_PLAYERS)
    return out
