"""Plays many games in lockstep so each controller makes one batched decision per tick."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .agents.base import ActResult, Controller, Request
from .comm.protocol import forced_message, message_mask
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
    simultaneous. Decisions with a single legal choice (a good player's quest card, a message whose
    slots are all closed or silenced) are taken automatically and never reach a controller.
    In a discussion the speaker's controller is asked to `speak` instead of `act`.
    `on_decision(request, result, row)` is called for every decision made by a controller whose
    `learn` flag is set.
    """
    active = [i for i, g in enumerate(games) if not g.done]
    while active:
        chosen: dict[int, dict[int, int | np.ndarray]] = {i: {} for i in active}
        msg_masks: dict[int, tuple] = {}  # game -> speaker's message masks, reused by the legality check
        groups: dict[tuple[int, str], tuple[Controller, list[Request]]] = {}
        for i in active:
            game = games[i]
            for player in game.acting_players():
                ctrl = evil_ctrls[i] if game.is_evil_seat(player) else good_ctrls[i]
                if game.in_discussion:
                    mask, rtype_valid, rround_valid = msg_masks[i] = message_mask(game, player)
                    forced = forced_message(mask)
                    if forced is not None:
                        chosen[i][player] = forced
                        continue
                    req = Request(i, game, player, mask, rtype_valid, rround_valid)
                    groups.setdefault((id(ctrl), "speak"), (ctrl, []))[1].append(req)
                    continue
                mask = game.legal_mask(player)
                legal = np.flatnonzero(mask)
                if len(legal) == 1:
                    chosen[i][player] = int(legal[0])
                    continue
                groups.setdefault((id(ctrl), "act"), (ctrl, []))[1].append(Request(i, game, player, mask))

        for (_, kind), (ctrl, requests) in groups.items():
            if kind == "act":
                result = ctrl.act(requests, rng)
                decisions = [int(a) for a in result.actions]
            else:
                result = ctrl.speak(requests, rng)
                decisions = list(result.messages)
            for row, req in enumerate(requests):
                chosen[req.game_idx][req.player] = decisions[row]
                if on_decision is not None and ctrl.learn:
                    on_decision(req, result, row)

        for i in active:
            games[i].step(chosen[i], msg_masks.get(i))
        active = [i for i in active if not games[i].done]
