"""Builders for scripted games used across the tests."""
from __future__ import annotations

import numpy as np

from avalon_rl.env import AvalonGame, Role
from avalon_rl.env import actions as A

# Seats 0-3 good, 4-5 evil.
ROLES = [Role.MERLIN, Role.PERCIVAL, Role.SERVANT, Role.SERVANT, Role.MORGANA, Role.ASSASSIN]


def make_game(roles=ROLES, leader: int = 0, comm=None) -> AvalonGame:
    game = AvalonGame(np.random.default_rng(0), comm=comm)
    game.roles = np.array([int(r) for r in roles], dtype=np.int8)
    game.leader = leader
    return game


def propose(game: AvalonGame, seats) -> None:
    game.step({game.leader: A.team_action(game.leader, seats)})


def vote(game: AvalonGame, approvers) -> None:
    game.step({p: A.APPROVE if p in approvers else A.REJECT for p in range(6)})


def quest(game: AvalonGame, failers=()) -> None:
    game.step({p: A.FAIL if p in failers else A.SUCCESS for p in game.acting_players()})


def play_quest(game: AvalonGame, seats, failers=()) -> None:
    propose(game, seats)
    vote(game, range(6))
    quest(game, failers)


def good_team(game: AvalonGame) -> list[int]:
    """The first k good seats (0-3) for the current quest size."""
    return list(range(game.team_size))
