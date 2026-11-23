"""A good-team partner that plays the plain logic human testers expect (a training partner; it never learns).

Human testers found that the silent baseline breaks rules any careful good player follows: it brings back
teams that already failed, proposes and approves teams that must hold an evil player, and rejects every
proposal until the fifth, even its own. The baseline can afford this in self-play because all its partners
share the habit, so early proposals never matter. Training some games with this partner in one or two good
seats makes early proposals and votes count again; the reward stays the game result.

Beliefs come from exact deduction: of the 15 possible evil pairs, those consistent with the quest results
and the player's private knowledge (`observation.possible_evil_pairs`), equally likely. The partner
- proposes a team containing itself with the highest chance of being clean (ties at random);
- approves its own proposal and every fifth proposal (rejecting it hands evil the game);
- otherwise approves a team when its chance of being clean is within `slack` of the best team of that size,
  and never approves a team that must hold an evil player.
It only plays Servants and Percival: Merlin and the evil roles stay with the learning policies.
It does not speak or listen: in a communication game the rule teacher speaks for its seat
(`training.workers.teacher_seats`), and those words bind the seat like anyone's: among the choices they leave,
the partner takes the one its logic ranks best.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from ..env import actions as A
from ..env.game import AvalonGame
from ..env.observation import EVIL_PAIRS, possible_evil_pairs
from ..env.rules import MAX_PROPOSALS, N_PLAYERS, Phase, Role
from .base import ActResult, Request

PARTNER_ROLES = (Role.SERVANT, Role.PERCIVAL)
_PAIR_MEMBERS = np.zeros((len(EVIL_PAIRS), N_PLAYERS), dtype=bool)
for _i, _pair in enumerate(EVIL_PAIRS):
    _PAIR_MEMBERS[_i, list(_pair)] = True
_TEAMS = {k: np.array([[s in t for s in range(N_PLAYERS)] for t in combinations(range(N_PLAYERS), k)])
          for k in (2, 3, 4)}


def clean_probability(game: AvalonGame, player: int, teams: np.ndarray) -> np.ndarray:
    """(n_teams,) chance that each team (bool rows over absolute seats) has no evil player, in `player`'s view."""
    members = _PAIR_MEMBERS[possible_evil_pairs(game, player)]
    return (~(members[:, None, :] & teams[None, :, :]).any(axis=2)).mean(axis=0)


class LogicController:
    learn = False

    def __init__(self, slack: float = 0.1):
        self.slack = slack

    def act(self, requests: list[Request], rng: np.random.Generator) -> ActResult:
        return ActResult(np.array([self._act(r.game, r.player, rng) for r in requests], dtype=np.int64))

    def speak(self, requests: list[Request], rng: np.random.Generator) -> ActResult:
        raise NotImplementedError("the logic partner does not speak: give its seat a speaker (runner.play, seat_speakers)")

    def _act(self, game: AvalonGame, player: int, rng: np.random.Generator) -> int:
        if Role(int(game.roles[player])) not in PARTNER_ROLES:
            raise ValueError(f"the logic partner plays only Servants and Percival, not {Role(int(game.roles[player])).name}")
        teams = _TEAMS[game.team_size]
        mask = game.legal_mask(player)
        if game.phase == Phase.PROPOSE:
            legal = mask[[A.team_action(player, np.flatnonzero(t)) for t in teams]]
            mine = teams[legal & teams[:, player]] if (legal & teams[:, player]).any() else teams[legal]
            p = clean_probability(game, player, mine)
            best = np.flatnonzero(p >= p.max() - 1e-9)
            return A.team_action(player, np.flatnonzero(mine[rng.choice(best)]))
        if game.phase == Phase.VOTE:
            if not mask[A.APPROVE] or not mask[A.REJECT]:
                return A.APPROVE if mask[A.APPROVE] else A.REJECT
            if game.attempt == MAX_PROPOSALS - 1 or game.leader == player:
                return A.APPROVE
            p_team = clean_probability(game, player, game.team[None, :])[0]
            p_best = clean_probability(game, player, teams).max()
            return A.APPROVE if p_team > 0 and p_team >= p_best - self.slack else A.REJECT
        raise ValueError(f"no decision for the logic partner in phase {game.phase}")
