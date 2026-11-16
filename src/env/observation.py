"""Information-state encoding.

Each observation is a fixed-length vector holding everything the player may legally know:
its own role, role-specific private knowledge, the complete public history and its own past
quest cards. Because the public history of a game is bounded (at most 5 quests x 5 proposals),
this vector is a complete information state, so a feedforward policy needs no recurrence.

All seat-indexed fields are rotated into the player's relative frame (index 0 = self).
`LAYOUT` names every field so later analyses (e.g. masking vote history for RQ3) can edit
an observation without knowing the offsets.

Deduction fields (appended last) do the logical bookkeeping a careful player does, as DeepRole does
with its deductive reasoning: of the 15 possible evil pairs, which are still consistent with the quest
results (a quest with f Fail cards had at least f evil members, since good players can only play
Success), first from public information alone and then also from the player's private knowledge.
They are functions of fields already in the vector, so they add no information; an intervention that
edits the quest history must recompute them. Networks trained before these fields existed read only
the first `OBS_DIM_NO_DEDUCTION` entries.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .game import AvalonGame
from .rules import MAX_PROPOSALS, N_DECISION_PHASES, N_PLAYERS, N_QUESTS, N_ROLES

_N_PROPOSALS = N_QUESTS * MAX_PROPOSALS
EVIL_PAIRS = tuple(combinations(range(N_PLAYERS), 2))  # in the player's relative frame inside the vector
N_PAIRS = len(EVIL_PAIRS)
_PAIR_MEMBERS = np.zeros((N_PAIRS, N_PLAYERS), dtype=bool)
for _i, (_a, _b) in enumerate(EVIL_PAIRS):
    _PAIR_MEMBERS[_i, [_a, _b]] = True
_PAIR_COUNTS = _PAIR_MEMBERS.astype(np.int8)
_PAIR_INDEX = {pair: i for i, pair in enumerate(EVIL_PAIRS)}
# _REL_PAIRS[player][r] = absolute pair index of relative pair r.
_REL_PAIRS = np.array([
    [_PAIR_INDEX[tuple(sorted(((player + a) % N_PLAYERS, (player + b) % N_PLAYERS)))] for a, b in EVIL_PAIRS]
    for player in range(N_PLAYERS)
])

_FIELDS = [
    ("role", N_ROLES),
    ("known_evil", N_PLAYERS),
    ("merlin_candidate", N_PLAYERS),
    ("phase", N_DECISION_PHASES),
    ("quest_index", N_QUESTS),
    ("attempt", MAX_PROPOSALS),
    ("leader", N_PLAYERS),
    ("current_team", N_PLAYERS),
    ("score", 2),  # successes / 3, fails / 3
    ("quest_status", N_QUESTS * 3),  # one-hot: not played, success, fail
    ("quest_fail_count", N_QUESTS * 3),  # one-hot: 0, 1, 2 fail cards
    ("quest_team", N_QUESTS * N_PLAYERS),
    ("proposal_exists", _N_PROPOSALS),
    ("proposal_leader", _N_PROPOSALS * N_PLAYERS),
    ("proposal_team", _N_PROPOSALS * N_PLAYERS),
    ("proposal_votes", _N_PROPOSALS * N_PLAYERS),  # +1 approve, -1 reject, 0 not (yet) voted
    ("proposal_approved", _N_PROPOSALS),
    ("my_quest_cards", N_QUESTS * 2),  # one-hot: played success, played fail
    # Deduction (see the module docstring); pairs indexed as EVIL_PAIRS in relative seats.
    ("pair_possible_public", N_PAIRS),
    ("pair_possible_private", N_PAIRS),
    ("certain_evil_public", N_PLAYERS),
    ("certain_good_public", N_PLAYERS),
    ("certain_evil_private", N_PLAYERS),
    ("certain_good_private", N_PLAYERS),
]

LAYOUT: dict[str, slice] = {}
_offset = 0
for _name, _size in _FIELDS:
    LAYOUT[_name] = slice(_offset, _offset + _size)
    _offset += _size
OBS_DIM = _offset
OBS_DIM_NO_DEDUCTION = LAYOUT["pair_possible_public"].start


def possible_evil_pairs(game: AvalonGame, player: int | None = None) -> np.ndarray:
    """(15,) bool over absolute-seat pairs: still possible evil pairs given the quest results and,
    with `player`, that player's private knowledge (own role, known evil, Percival's candidates)."""
    played = game.quest_result >= 0
    counts = _PAIR_COUNTS @ game.quest_team[played].T.astype(np.int8)  # evil members per pair and quest
    possible = (counts >= game.quest_fails[played]).all(axis=1)
    return possible if player is None else possible & _private_filter(game, player)


def _private_filter(game: AvalonGame, player: int) -> np.ndarray:
    """(15,) bool over absolute-seat pairs: the evil pairs `player`'s private knowledge allows."""
    known_evil, candidates = game.private_knowledge(player)
    if known_evil.any():  # Merlin and the evil players know the pair
        return (_PAIR_MEMBERS == known_evil).all(axis=1)
    allowed = ~_PAIR_MEMBERS[:, player]  # a good player knows itself to be good
    if candidates.any():  # exactly one of Percival's two candidates is Morgana
        allowed &= (_PAIR_MEMBERS & candidates).sum(axis=1) == 1
    return allowed


def _certain(possible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    members = _PAIR_MEMBERS[possible]
    return members.all(axis=0) & possible.any(), ~members.any(axis=0)


def _seat_perm(player: int) -> np.ndarray:
    """perm[r] = absolute seat of relative seat r."""
    return (player + np.arange(N_PLAYERS)) % N_PLAYERS


def encode(game: AvalonGame, player: int) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    perm = _seat_perm(player)

    obs[LAYOUT["role"]][int(game.roles[player])] = 1.0
    known_evil, candidates = game.private_knowledge(player)
    obs[LAYOUT["known_evil"]] = known_evil[perm]
    obs[LAYOUT["merlin_candidate"]] = candidates[perm]

    if game.phase < N_DECISION_PHASES:  # discussion phases are flagged in the communication features instead
        obs[LAYOUT["phase"]][int(game.phase)] = 1.0
    if game.quest < N_QUESTS:
        obs[LAYOUT["quest_index"]][game.quest] = 1.0
    if game.attempt < MAX_PROPOSALS:
        obs[LAYOUT["attempt"]][game.attempt] = 1.0
    obs[LAYOUT["leader"]][(game.leader - player) % N_PLAYERS] = 1.0
    obs[LAYOUT["current_team"]] = game.team[perm]
    obs[LAYOUT["score"]] = (game.successes / 3.0, game.fails / 3.0)

    status = obs[LAYOUT["quest_status"]].reshape(N_QUESTS, 3)
    status[np.arange(N_QUESTS), game.quest_result + 1] = 1.0
    fail_count = obs[LAYOUT["quest_fail_count"]].reshape(N_QUESTS, 3)
    played = game.quest_result >= 0
    fail_count[np.flatnonzero(played), game.quest_fails[played]] = 1.0
    obs[LAYOUT["quest_team"]] = game.quest_team[:, perm].reshape(-1)

    exists = game.prop_leader >= 0  # (quests, attempts)
    obs[LAYOUT["proposal_exists"]] = exists.reshape(-1)
    leader_onehot = obs[LAYOUT["proposal_leader"]].reshape(N_QUESTS, MAX_PROPOSALS, N_PLAYERS)
    qi, ai = np.nonzero(exists)
    leader_onehot[qi, ai, (game.prop_leader[qi, ai] - player) % N_PLAYERS] = 1.0
    obs[LAYOUT["proposal_team"]] = game.prop_team[:, :, perm].reshape(-1)
    votes = np.where(game.prop_votes[:, :, perm], 1.0, -1.0) * game.prop_voted[:, :, None]
    obs[LAYOUT["proposal_votes"]] = votes.reshape(-1)
    obs[LAYOUT["proposal_approved"]] = game.prop_approved.reshape(-1)

    cards = obs[LAYOUT["my_quest_cards"]].reshape(N_QUESTS, 2)
    mine = game.quest_cards[player]
    cards[np.flatnonzero(mine >= 0), mine[mine >= 0]] = 1.0

    obs[OBS_DIM_NO_DEDUCTION:] = _deduction_block(game, player, perm)
    return obs


# Fields indexed by seat in their last axis; every other field reads the same for all players (or is private).
_SEAT_FIELDS = ("known_evil", "merlin_candidate", "leader", "current_team", "quest_team",
                "proposal_leader", "proposal_team", "proposal_votes")
# _ROTATE[player] gathers `player`'s relative-seat view from a vector encoded for seat 0.
_ROTATE = np.tile(np.arange(OBS_DIM), (N_PLAYERS, 1))
for _player in range(N_PLAYERS):
    for _name in _SEAT_FIELDS:
        _cols = np.arange(LAYOUT[_name].start, LAYOUT[_name].stop).reshape(-1, N_PLAYERS)
        _ROTATE[_player, LAYOUT[_name]] = _cols[:, _seat_perm(_player)].reshape(-1)


def encode_batch(pairs: list[tuple[AvalonGame, int]]) -> np.ndarray:
    """`np.stack([encode(game, player) for game, player in pairs])`, faster when a game appears several times
    (all six voters): the shared fields are encoded once per game and rotated into each player's seat frame."""
    out = np.empty((len(pairs), OBS_DIM), dtype=np.float32)
    shared: dict[int, np.ndarray] = {}
    for row, (game, player) in enumerate(pairs):
        base = shared.get(id(game))
        if base is None:
            base = shared[id(game)] = encode(game, 0)
        if player == 0:
            out[row] = base
            continue
        obs = out[row]
        obs[:] = base[_ROTATE[player]]
        perm = _seat_perm(player)
        role = obs[LAYOUT["role"]]
        role[:] = 0.0
        role[int(game.roles[player])] = 1.0
        known_evil, candidates = game.private_knowledge(player)
        obs[LAYOUT["known_evil"]] = known_evil[perm]
        obs[LAYOUT["merlin_candidate"]] = candidates[perm]
        cards = obs[LAYOUT["my_quest_cards"]].reshape(N_QUESTS, 2)
        cards[:] = 0.0
        mine = game.quest_cards[player]
        cards[np.flatnonzero(mine >= 0), mine[mine >= 0]] = 1.0
        obs[OBS_DIM_NO_DEDUCTION:] = _deduction_block(game, player, perm)
    return out


def _deduction_block(game: AvalonGame, player: int, perm: np.ndarray) -> np.ndarray:
    """The deduction fields of `player`, cached on the game until the next quest result (or reset)."""
    cache = getattr(game, "_deduction_cache", None)
    if cache is None or cache[0] is not game.quest_result or cache[1] != game.quest:
        public = possible_evil_pairs(game)
        cache = game._deduction_cache = (game.quest_result, game.quest, (public, *_certain(public)), {})
    block = cache[3].get(player)
    if block is None:
        public, evil_pub, good_pub = cache[2]
        private = public & _private_filter(game, player)
        evil_priv, good_priv = _certain(private)
        rel_pairs = _REL_PAIRS[player]
        parts = [public[rel_pairs], private[rel_pairs], evil_pub[perm], good_pub[perm], evil_priv[perm], good_priv[perm]]
        block = cache[3][player] = np.concatenate(parts).astype(np.float32)
    return block
