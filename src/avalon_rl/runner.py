"""Plays many games in lockstep so each controller makes one batched decision per tick."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .agents.base import ActResult, Controller, Request
from .env.game import AvalonGame

DecisionHook = Callable[[Request, ActResult, int], None]


def play(
    games: Sequence[AvalonGame],
    good_ctrls: Sequence[Controller],
    evil_ctrls: Sequence[Controller],
    rng: np.random.Generator,
    on_decision: DecisionHook | None = None,
) -> None:
    """Advance every game to the end.

    `good_ctrls[i]` and `evil_ctrls[i]` control the two teams of game i. All decisions of one tick
    are made from the same state before any of them is applied, so votes and quest cards are
    simultaneous. Decisions with a single legal action (a good player's quest card) are taken
    automatically and never reach a controller. `on_decision(request, result, row)` is called for
    every decision made by a controller whose `learn` flag is set.
    """
    active = [i for i, g in enumerate(games) if not g.done]
    while active:
        chosen: dict[int, dict[int, int]] = {i: {} for i in active}
        groups: dict[int, tuple[Controller, list[Request]]] = {}
        for i in active:
            game = games[i]
            for player in game.acting_players():
                mask = game.legal_mask(player)
                legal = np.flatnonzero(mask)
                if len(legal) == 1:
                    chosen[i][player] = int(legal[0])
                    continue
                ctrl = evil_ctrls[i] if game.is_evil_seat(player) else good_ctrls[i]
                groups.setdefault(id(ctrl), (ctrl, []))[1].append(Request(i, game, player, mask))

        for ctrl, requests in groups.values():
            result = ctrl.act(requests, rng)
            for row, req in enumerate(requests):
                chosen[req.game_idx][req.player] = int(result.actions[row])
                if on_decision is not None and ctrl.learn:
                    on_decision(req, result, row)

        for i in active:
            games[i].step(chosen[i])
        active = [i for i in active if not games[i].done]
