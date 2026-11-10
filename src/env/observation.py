"""Information-state encoding.

Each observation is a fixed-length vector holding everything the player may legally know:
its own role, role-specific private knowledge, the complete public history and its own past
quest cards. Because the public history of a game is bounded (at most 5 quests x 5 proposals),
this vector is a complete information state, so a feedforward policy needs no recurrence.

All seat-indexed fields are rotated into the player's relative frame (index 0 = self).
`LAYOUT` names every field so later analyses (e.g. masking vote history for RQ3) can edit
an observation without knowing the offsets.
"""
from __future__ import annotations

import numpy as np

from .game import AvalonGame
from .rules import MAX_PROPOSALS, N_DECISION_PHASES, N_PLAYERS, N_QUESTS, N_ROLES

_N_PROPOSALS = N_QUESTS * MAX_PROPOSALS

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
]

LAYOUT: dict[str, slice] = {}
_offset = 0
for _name, _size in _FIELDS:
    LAYOUT[_name] = slice(_offset, _offset + _size)
    _offset += _size
OBS_DIM = _offset


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
    return obs
