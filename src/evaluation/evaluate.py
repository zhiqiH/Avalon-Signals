"""Frozen evaluation with paired role assignments.

Game i of every matchup evaluated with the same `seed` gets the same seating and starting leader,
so differences between matchups are not driven by role-assignment luck. Policy sampling uses a
separate random stream.
"""
from __future__ import annotations

import numpy as np

from ..agents.base import Controller
from ..comm.protocol import CommConfig
from ..env.game import AvalonGame
from ..runner import play
from .metrics import game_outcome, summarize


def paired_games(seed: int, start: int, stop: int, comm: CommConfig | None = None) -> list[AvalonGame]:
    return [AvalonGame(np.random.default_rng([seed, i]), comm=comm) for i in range(start, stop)]


def run_matchup(
    good: Controller,
    evil: Controller,
    n_games: int,
    seed: int,
    batch_size: int = 512,
    keep_records: bool = False,
    comm: CommConfig | None = None,
) -> tuple[dict, list[dict]]:
    outcomes: list[dict] = []
    records: list[dict] = []
    policy_rng = np.random.default_rng([seed, 1_000_003])
    for start in range(0, n_games, batch_size):
        games = paired_games(seed, start, min(start + batch_size, n_games), comm)
        play(games, [good] * len(games), [evil] * len(games), policy_rng)
        outcomes.extend(game_outcome(g) for g in games)
        if keep_records:
            records.extend(g.to_record() for g in games)
    return summarize(outcomes), records
