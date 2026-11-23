"""Frozen evaluation with paired role assignments.

Game i of every matchup evaluated with the same `seed` gets the same seating and starting leader,
so differences between matchups are not driven by role-assignment luck. Policy sampling uses a
separate random stream.

`partners=k` seats the logic partner (agents/logic_agent.py) in k of the good team's Servant/Percival seats,
the same seats for every policy evaluated with the same `seed`; in a communication game the rule teacher
speaks for them. It measures how a policy fares next to a player that follows plain logic and none of the
policy's own conventions.
"""
from __future__ import annotations

import numpy as np

from ..agents.base import Controller
from ..agents.logic_agent import PARTNER_ROLES, LogicController
from ..agents.teacher import TeacherSpeaker
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
    partners: int = 0,
) -> tuple[dict, list[dict]]:
    outcomes: list[dict] = []
    records: list[dict] = []
    policy_rng = np.random.default_rng([seed, 1_000_003])
    seat_rng = np.random.default_rng([seed, 2_000_003])
    partner, teacher = LogicController(), TeacherSpeaker(seed=seed)
    for start in range(0, n_games, batch_size):
        games = paired_games(seed, start, min(start + batch_size, n_games), comm)
        seats = speakers = None
        if partners:
            candidates = [np.flatnonzero(np.isin(g.roles, [int(r) for r in PARTNER_ROLES])) for g in games]
            seats = [{int(s): partner for s in seat_rng.permutation(c)[:partners]} for c in candidates]
            speakers = [{s: teacher for s in d} for d in seats] if comm is not None and comm.enabled else None
        play(games, [good] * len(games), [evil] * len(games), policy_rng, seat_ctrls=seats, seat_speakers=speakers)
        outcomes.extend(game_outcome(g) for g in games)
        if keep_records:
            records.extend(g.to_record() for g in games)
    return summarize(outcomes), records
