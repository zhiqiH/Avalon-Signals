"""Frozen evaluation with paired role assignments.

Game i of every matchup evaluated with the same `seed` gets the same seating and starting leader,
so differences between matchups are not driven by role-assignment luck. Policy sampling uses a
separate random stream.
"""
from __future__ import annotations

import numpy as np

from ..agents.base import Controller
from ..env.game import AvalonGame
from ..runner import play
from .metrics import game_outcome, summarize


def paired_games(seed: int, start: int, stop: int) -> list[AvalonGame]:
    return [AvalonGame(np.random.default_rng([seed, i])) for i in range(start, stop)]


def run_matchup(
    good: Controller,
    evil: Controller,
    n_games: int,
    seed: int,
    batch_size: int = 512,
    keep_records: bool = False,
) -> tuple[dict, list[dict]]:
    outcomes: list[dict] = []
    records: list[dict] = []
    policy_rng = np.random.default_rng([seed, 1_000_003])
    for start in range(0, n_games, batch_size):
        games = paired_games(seed, start, min(start + batch_size, n_games))
        play(games, [good] * len(games), [evil] * len(games), policy_rng)
        outcomes.extend(game_outcome(g) for g in games)
        if keep_records:
            records.extend(g.to_record() for g in games)
    return summarize(outcomes), records
