"""How do trained talking agents fare next to one seat the word rules do not bind (a stand-in for a human)?

One seat of every game is the test seat; the other five are the checkpoint's policies under the usual rules. The
same deals are played under several conditions:

    control       the test seat is an agent like the others (bound, speaks as trained)
    free_full     free seat: speaks as the agent would, acts on the policy's unbound choice
    free_sparse   free seat: keeps its claim and half of its judgments, names no team, announces no intent
    free_silent   free seat: says nothing
    logic_sparse  like free_sparse, and as Servant or Percival it acts by plain deduction (agents.logic_agent)
    logic_loose   deduction for the deeds; judgments by its own belief, with or without a public record

A free seat's deeds against its words are counted in the public audit (`AvalonGame.free_seats`). Reported per
role group of the test seat: results, how often the agents put it on teams, and its audit counts.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.agents import PolicyController, Request
from src.agents.logic_agent import PARTNER_ROLES, LogicController
from src.comm import protocol as P
from src.env.game import AvalonGame
from src.env.observation import _PAIR_MEMBERS, possible_evil_pairs
from src.env.rules import N_PLAYERS, Phase, Role, WinReason
from src.evaluation.metrics import wilson_interval
from src.training.checkpoint import comm_config, load_checkpoint, load_policies

CONDITIONS = {  # name: (free, speech, deeds)
    "control": (False, "own", "policy"),
    "free_full": (True, "own", "policy"),
    "free_sparse": (True, "sparse", "policy"),
    "free_silent": (True, "silent", "policy"),
    "logic_sparse": (True, "sparse", "logic"),
    "logic_loose": (True, "loose", "logic"),
}


def group(game: AvalonGame, seat: int) -> str:
    role = Role(int(game.roles[seat]))
    return "evil" if game.is_evil_seat(seat) else "merlin" if role == Role.MERLIN else "servant_percival"


def loose_judgments(game: AvalonGame, seat: int, rng: np.random.Generator) -> np.ndarray:
    """Judgments of relative seats 1..5 by the seat's own belief, whatever the public record holds."""
    others = (seat + np.arange(1, N_PLAYERS)) % N_PLAYERS
    judged = np.zeros(N_PLAYERS - 1, dtype=np.int64)
    evil = np.array([game.is_evil_seat(int(s)) for s in others])
    if game.is_evil_seat(seat):  # blames one good player half of the time, sometimes vouches for its partner
        if rng.random() < 0.5:
            judged[rng.choice(np.flatnonzero(~evil))] = P.J_EVIL
        if rng.random() < 0.3:
            judged[np.flatnonzero(evil)[0]] = P.J_GOOD
    elif game.roles[seat] == Role.MERLIN:  # a careful Merlin: points at an evil player now and then
        judged[evil & (rng.random(len(others)) < 0.3)] = P.J_EVIL
    else:  # deduction over the evil pairs still possible: accuses the likely, vouches for the cleared
        p_evil = _PAIR_MEMBERS[possible_evil_pairs(game, seat)].mean(axis=0)[others]
        likely = np.argsort(-p_evil)[:2]
        judged[likely[p_evil[likely] >= 0.6]] = P.J_EVIL
        judged[p_evil == 0] = P.J_GOOD
        if np.isin(judged, P.GOOD_JUDGMENTS).sum() > 3:  # the role deck leaves three other good players
            judged[np.flatnonzero(judged == P.J_GOOD)[3:]] = P.J_NONE
    return judged


def free_speech(game: AvalonGame, seat: int, own: np.ndarray, speech: str, rng: np.random.Generator) -> np.ndarray:
    if speech == "own" or game.phase == Phase.EVIL_DISCUSS:
        return own
    if speech == "silent":
        return P.free_message(game, seat)
    judged = own[P.H_JUDGE:P.H_JUDGE + N_PLAYERS - 1].astype(np.int64)
    if speech == "sparse":
        judged = np.where(rng.random(len(judged)) < 0.5, judged, P.J_NONE)
    else:
        judged = loose_judgments(game, seat, rng)
    return P.free_message(game, seat, claim=int(own[P.H_CLAIM]), judgments=judged)


def play(game: AvalonGame, seat: int, condition: str, ctrls: dict, logic: LogicController, rng: np.random.Generator,
         c: dict) -> None:
    free, speech, deeds = CONDITIONS[condition]
    if free:
        game.free_seats = frozenset({seat})
    g = group(game, seat)
    by_logic = deeds == "logic" and Role(int(game.roles[seat])) in PARTNER_ROLES
    side = lambda p: "evil" if game.is_evil_seat(p) else "good"  # noqa: E731
    while not game.done:
        acting, chosen = game.acting_players(), {}
        if game.in_discussion:
            p = acting[0]
            ctx = P.message_context(game, p)
            msg = P.forced_message(ctx)
            if msg is None:
                msg = ctrls[side(p)].speak([Request(0, game, p, None, ctx)], rng).messages[0]
            if p == seat and free:
                msg = free_speech(game, seat, np.asarray(msg), speech, rng)
            elif p != seat and game.phase == Phase.DISCUSS and msg[P.H_TEAM] > 0:
                on_it = P.TEAM_MEMBERS[game.team_size][int(msg[P.H_TEAM])][(seat - p) % N_PLAYERS]
                c[g, "others_suggestions"] += 1
                c[g, "others_suggestions_with_seat"] += bool(on_it)
            chosen[p] = msg
        else:
            phase = game.phase
            for s in ("good", "evil"):
                requests = []
                for p in acting:
                    if side(p) != s:
                        continue
                    mask = game.legal_mask(p)
                    legal = np.flatnonzero(mask)
                    if len(legal) == 1:
                        chosen[p] = int(legal[0])
                    elif p == seat and by_logic and phase in (Phase.PROPOSE, Phase.VOTE):
                        chosen[p] = logic._act(game, p, rng)
                    else:
                        requests.append(Request(0, game, p, mask))
                if requests:
                    actions = ctrls[s].act(requests, rng).actions
                    chosen.update({r.player: int(a) for r, a in zip(requests, actions)})
        leader = game.leader
        game.step(chosen)
        if game.phase == Phase.VOTE and leader != seat:  # a team was just proposed by another seat
            c[g, "others_proposals"] += 1
            c[g, "others_proposals_with_seat"] += bool(game.team[seat])
    quests = game.quest_result >= 0
    c[g, "games"] += 1
    c[g, "good_win"] += bool(game.good_won)
    three = game.win_reason in (WinReason.ASSASSIN_HIT, WinReason.ASSASSIN_MISSED)
    c[g, "three_successes"] += three
    c[g, "assassin_hit"] += game.win_reason == WinReason.ASSASSIN_HIT
    c[g, "quests"] += int(quests.sum())
    c[g, "quests_with_seat"] += int(game.quest_team[quests][:, seat].sum())
    for k, name in enumerate(P.AUDIT):
        c[g, "audit_" + name] += int(game.audit[seat, k])
    suspicion = P.public_suspicion(game)
    mates = [p for p in range(N_PLAYERS) if p != seat and game.is_evil_seat(p) == game.is_evil_seat(seat)]
    c[g, "suspicion_seat"] += float(suspicion[seat])
    c[g, "suspicion_teammates"] += float(suspicion[mates].mean())


def summarize(c: dict) -> dict:
    out = {}
    for g in ("servant_percival", "merlin", "evil"):
        n = int(c[g, "games"])
        if not n:
            continue
        rate = lambda a, b: round(c[g, a] / c[g, b], 4) if c[g, b] else None  # noqa: E731
        wins = int(c[g, "good_win"])
        out[g] = {
            "games": n, "good_win": round(wins / n, 4), "good_win_ci95": [round(x, 4) for x in wilson_interval(wins, n)],
            "three_successes": rate("three_successes", "games"),
            "assassin_hit_given_three_successes": rate("assassin_hit", "three_successes"),
            "others_suggestions_with_seat": rate("others_suggestions_with_seat", "others_suggestions"),
            "others_proposals_with_seat": rate("others_proposals_with_seat", "others_proposals"),
            "quests_with_seat": rate("quests_with_seat", "quests"),
            "audit_per_game": {name: rate("audit_" + name, "games") for name in P.AUDIT},
            "final_suspicion_seat": rate("suspicion_seat", "games"),
            "final_suspicion_teammates": rate("suspicion_teammates", "games"),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    comm = comm_config(load_checkpoint(args.checkpoint)["config"])
    ctrls = {t: PolicyController(net, learn=False) for t, net in load_policies(args.checkpoint).items()}
    logic = LogicController()
    result = {"checkpoint": args.checkpoint, "games": args.games, "seed": args.seed, "conditions": {}}
    for condition in args.conditions:
        c: dict = defaultdict(float)
        rng = np.random.default_rng([args.seed, 1])
        for i in range(args.games):
            game = AvalonGame(np.random.default_rng([args.seed, 0, i]), comm=comm)  # the same deal in every condition
            play(game, i % N_PLAYERS, condition, ctrls, logic, rng, c)
        result["conditions"][condition] = summarize(c)
        print(condition, json.dumps(result["conditions"][condition]), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
