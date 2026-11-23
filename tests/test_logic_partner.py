"""The logic partner and the partner training mode."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.agents import ActorCritic, LogicController, PolicyController, Request, RuleController
from src.agents.logic_agent import clean_probability
from src.config import load_config
from src.env import AvalonGame, Role
from src.env import actions as A
from src.runner import play
from src.training import TrajectoryRecorder
from src.training.trainer import sample_matchups, train
from src.comm import protocol as P
from src.training.workers import LEARNER, PARTNER, partner_seats, teacher_seats
from tests.helpers import make_game, play_quest, propose

PILOT = Path(__file__).resolve().parents[1] / "configs" / "pilot_silent.yaml"
# Seats: 0 Merlin, 1 Percival, 2-3 Servants, 4 Morgana, 5 Assassin.


def _act(game, player, seed=0):
    return int(LogicController().act([Request(0, game, player, game.legal_mask(player))],
                                     np.random.default_rng(seed)).actions[0])


def test_never_brings_back_a_failed_team_or_its_superset():
    game = make_game(leader=0)
    play_quest(game, [2, 4], failers=[4])  # servant 2 and Morgana failed: 2 knows 4 is evil
    play_quest(game, [0, 1, 2], failers=[])
    # Servant 2 leads quest 3 (four seats): every team with 4 must be avoided.
    game.leader = 2
    for seed in range(20):
        team = A.decode_team(2, _act(game, 2, seed), game.team_size)
        assert 4 not in team and 2 in team


def test_votes_follow_deduction_and_always_approve_own_and_fifth_proposals():
    game = make_game(leader=0)
    play_quest(game, [3, 4], failers=[4])  # servant 2 cannot tell whether 3 or 4 is evil...
    propose(game, [3, 4, 2])  # ...but a superset of a failed team must hold an evil player
    assert _act(game, 2) == A.REJECT
    game = make_game(leader=2)
    propose(game, [2, 3])
    assert _act(game, 2) == A.APPROVE  # own proposal
    game = make_game(leader=0)
    play_quest(game, [3, 4], failers=[4])
    for _ in range(4):  # four rejections, then the fifth proposal of a team known to hold evil
        propose(game, [3, 4, 0])
        game.step({p: A.REJECT for p in range(6)})
    propose(game, [3, 4, 0])
    assert game.attempt == 4 and _act(game, 2) == A.APPROVE


def test_percival_knows_a_four_seat_team_with_both_or_neither_candidate_must_hold_evil():
    # Percival (seat 1) sees candidates 0 and 4; the others (2, 3, 5) hold exactly one evil player.
    teams = np.array([[s in t for s in range(6)] for t in ([1, 2, 3, 5], [1, 0, 4, 2], [1, 2, 3, 0])])
    p = clean_probability(make_game(), 1, teams)
    assert p[0] == 0 and p[1] == 0 and p[2] > 0


def test_refuses_roles_it_does_not_play():
    game = make_game(leader=0)
    with pytest.raises(ValueError):
        _act(game, 0)  # Merlin


def test_partner_decisions_are_never_training_samples():
    rng = np.random.default_rng(1)
    games = [AvalonGame(rng) for _ in range(16)]
    specs = [(PARTNER, LEARNER)] * 16
    seats = partner_seats(games, specs, rng, LogicController(), max_partners=2)
    for game, s in zip(games, seats):
        assert 1 <= len(s) <= 2 and all(game.roles[p] in (Role.SERVANT, Role.PERCIVAL) for p in s)
    learner = PolicyController(ActorCritic(), learn=True)
    recorder = TrajectoryRecorder()
    play(games, [learner] * 16, [RuleController()] * 16, rng, on_decision=recorder, seat_ctrls=seats)
    assert all(g.done for g in games)
    recorded = {(gi, p) for gi, p in recorder.steps}
    assert recorded and not any(p in seats[gi] for gi, p in recorded)


def test_sample_matchups_without_partner_is_unchanged():
    old = sample_matchups(200, {"good": 3, "evil": 3}, [0.5, 0.25, 0.25], np.random.default_rng(7))
    new = sample_matchups(200, {"good": 3, "evil": 3}, [0.5, 0.25, 0.25, 0.0], np.random.default_rng(7))
    assert old == new
    specs, modes = sample_matchups(400, {"good": 0, "evil": 0}, [0.5, 0, 0, 0.5], np.random.default_rng(7))
    assert {s for s, m in zip(specs, modes) if m == "partner"} == {(PARTNER, LEARNER)}
    assert 150 < modes.count("partner") < 250


COMM_M = {"comm.enabled": True, "comm.claim": False, "comm.team": False, "comm.intent": False}


def test_in_a_communication_game_the_teacher_speaks_for_partner_seats_and_a_share_of_the_others():
    rng = np.random.default_rng(3)
    cfg = load_config(PILOT, {**COMM_M, "opponents.p_self": 0.25, "opponents.p_partner": 0.25})
    games = [AvalonGame(rng, comm=cfg.comm) for _ in range(16)]
    partners = partner_seats(games, [(PARTNER, LEARNER)] * 16, rng, LogicController(), max_partners=2)
    speakers = teacher_seats(cfg, partners, rng)
    assert [set(s) for s in speakers] == [set(p) for p in partners]  # no share: only the partners' seats
    assert teacher_seats(load_config(PILOT), partners, rng) is None  # the silent game has no speakers
    learner = {t: PolicyController(ActorCritic(comm=True), learn=True) for t in ("good", "evil")}
    recorder = TrajectoryRecorder()
    play(games, [learner["good"]] * 16, [learner["evil"]] * 16, rng, on_decision=recorder, seat_ctrls=partners,
         seat_speakers=speakers)
    assert all(g.done for g in games)
    assert not any(p in partners[gi] for gi, p in recorder.steps)  # neither their actions nor their messages
    assert any((g.msg_log[:, list(partners[i]), 0] >= 0).any() for i, g in enumerate(games))  # the seats did speak

    cfg.adapt.teacher_share = 1.0
    games = [AvalonGame(rng, comm=cfg.comm) for _ in range(16)]
    everyone = teacher_seats(cfg, [{}] * 16, rng)
    assert all(set(s) == set(range(6)) for s in everyone)
    recorder = TrajectoryRecorder()
    play(games, [learner["good"]] * 16, [learner["evil"]] * 16, rng, on_decision=recorder, seat_speakers=everyone)
    batches = recorder.build(games, 1.0, 0.95)
    assert not (batches["good"].actions[:, 0] < 0).any()  # no message of a good player is a training sample


@pytest.mark.parametrize("workers", [0, 2])
def test_communication_training_with_partner_games_and_teacher_seats_runs(tmp_path, workers):
    overrides = {
        "out_dir": str(tmp_path), "device": "cpu", "total_games": 64, "games_per_iter": 32, "checkpoint_every": 2,
        "eval.every": 2, "eval.games": 32, "ppo.minibatch_size": 256, "workers": workers,
        "opponents.p_self": 0.25, "opponents.p_partner": 0.25, "adapt.teacher_share": 0.4, **COMM_M,
    }
    run_dir = train(load_config(PILOT, overrides))
    lines = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(e["modes"]["partner"] > 0 and e["partner"]["games"] == e["modes"]["partner"] for e in lines[1:])


@pytest.mark.parametrize("workers", [0, 2])
def test_training_with_partner_games_logs_them(tmp_path, workers):
    overrides = {
        "out_dir": str(tmp_path), "device": "cpu", "total_games": 64, "games_per_iter": 32, "checkpoint_every": 2,
        "eval.every": 2, "eval.games": 32, "ppo.minibatch_size": 256, "workers": workers,
        "opponents.p_self": 0.25, "opponents.p_partner": 0.25,
    }
    run_dir = train(load_config(PILOT, overrides))
    lines = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(e["modes"]["partner"] > 0 and e["partner"]["games"] == e["modes"]["partner"] for e in lines[1:])
