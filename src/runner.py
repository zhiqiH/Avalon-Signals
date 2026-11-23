"""Plays many games in lockstep so each controller makes one batched decision per tick."""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

from .agents.base import ActResult, Controller, Request
from .comm.protocol import check_messages, forced_message, message_context, stack_contexts
from .env.game import AvalonGame
from .env.rules import Phase

DecisionHook = Callable[[Request, ActResult, int], None]


def play(
    games: Sequence[AvalonGame],
    good_ctrls: Sequence[Controller],
    evil_ctrls: Sequence[Controller],
    rng: np.random.Generator,
    on_decision: DecisionHook | None = None,
    seat_ctrls: Sequence[Mapping[int, Controller]] | None = None,
    seat_speakers: Sequence[Mapping[int, Controller]] | None = None,
) -> None:
    """Advance every game to the end.

    `good_ctrls[i]` and `evil_ctrls[i]` control the two teams of game i. All decisions of one tick
    are made from the same state before any of them is applied, so votes and quest cards are
    simultaneous. Decisions with a single legal choice (a good player's quest card, a message whose
    slots are all closed or silenced) are taken automatically and never reach a controller.
    In a discussion the speaker's controller is asked to `speak` instead of `act`.
    `on_decision(request, result, row)` is called for every decision made by a controller whose
    `learn` flag is set (for messages, its `learn_speech` flag where it has one: a policy that speaks
    through a frozen speaker learns only from its actions).
    `seat_ctrls[i]`, if given, maps seats of game i to a controller that replaces the team controller there
    (training partners such as `LogicController` in some good seats).
    `seat_speakers[i]`, if given, maps seats of game i to a controller that speaks for them in the public
    discussions while their own controller keeps acting (the rule teacher in some seats; a seat of a partner
    that cannot speak). The evil players' talk before the assassination always stays their own.
    """
    active = [i for i, g in enumerate(games) if not g.done]
    while active:
        chosen: dict[int, dict[int, int | np.ndarray]] = {i: {} for i in active}
        groups: dict[tuple[int, str], tuple[Controller, list[Request]]] = {}
        for i in active:
            game = games[i]
            for player in game.acting_players():
                ctrl = evil_ctrls[i] if game.is_evil_seat(player) else good_ctrls[i]
                if seat_ctrls is not None and player in seat_ctrls[i]:
                    ctrl = seat_ctrls[i][player]
                if game.in_discussion:
                    if seat_speakers is not None and player in seat_speakers[i] and game.phase != Phase.EVIL_DISCUSS:
                        ctrl = seat_speakers[i][player]
                    ctx = message_context(game, player)
                    forced = forced_message(ctx)
                    if forced is not None:
                        chosen[i][player] = forced
                        continue
                    req = Request(i, game, player, None, ctx)
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
                check_messages(result.messages, stack_contexts([r.ctx for r in requests]))
                decisions = list(result.messages)
            for row, req in enumerate(requests):
                chosen[req.game_idx][req.player] = decisions[row]
                if on_decision is not None and (ctrl.learn if kind == "act" else getattr(ctrl, "learn_speech", ctrl.learn)):
                    on_decision(req, result, row)

        for i in active:
            games[i].step(chosen[i], message_checked=True)  # messages were checked per batch above
        active = [i for i in active if not games[i].done]
