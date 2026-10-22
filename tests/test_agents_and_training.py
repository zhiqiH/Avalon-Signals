import json
from pathlib import Path

import numpy as np
import pytest
import torch

from avalon_rl.agents import ActorCritic, PolicyController, RandomController, Request, RuleController
from avalon_rl.config import load_config
from avalon_rl.env import AvalonGame
from avalon_rl.evaluation import run_matchup
from avalon_rl.runner import play
from avalon_rl.training import TrajectoryRecorder, gae_terminal
from avalon_rl.training.checkpoint import load_policies
from avalon_rl.training.trainer import train

PILOT = Path(__file__).resolve().parents[1] / "configs" / "pilot_silent.yaml"


def test_scripted_players_finish_every_game():
    rng = np.random.default_rng(0)
    rule, rand = RuleController(), RandomController()
    for good, evil in [(rule, rule), (rule, rand), (rand, rule), (rand, rand)]:
        games = [AvalonGame(rng) for _ in range(100)]
        play(games, [good] * 100, [evil] * 100, rng)  # AvalonGame.step rejects illegal actions
        assert all(g.done for g in games)


def test_rule_players_beat_random_ones():
    rule, rand = RuleController(), RandomController()
    rand_vs_rand, _ = run_matchup(rand, rand, 400, seed=3)
    rule_vs_rand, _ = run_matchup(rule, rand, 400, seed=3)
    rand_vs_rule, _ = run_matchup(rand, rule, 400, seed=3)
    assert rule_vs_rand["good_win"]["rate"] > rand_vs_rand["good_win"]["rate"] + 0.1
    assert rand_vs_rule["good_win"]["rate"] < rand_vs_rand["good_win"]["rate"]


def test_policy_only_samples_legal_actions():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    ctrl = PolicyController(ActorCritic(), learn=False)
    games = [AvalonGame(rng) for _ in range(64)]
    play(games, [ctrl] * 64, [ctrl] * 64, rng)
    assert all(g.done for g in games)
    game = AvalonGame(rng)
    reqs = [Request(0, game, game.leader, game.legal_mask(game.leader))] * 500
    actions = ctrl.act(reqs, rng).actions
    assert reqs[0].mask[actions].all()


def test_gae_reduces_to_monte_carlo_and_td0():
    values = np.array([0.2, -0.1, 0.4])
    adv, ret = gae_terminal(values, 1.0, gamma=1.0, lam=1.0)
    assert np.allclose(adv, 1.0 - values) and np.allclose(ret, 1.0)
    adv, _ = gae_terminal(values, -1.0, gamma=1.0, lam=0.0)
    assert np.allclose(adv, [-0.1 - 0.2, 0.4 + 0.1, -1.0 - 0.4])


def test_recorder_assigns_team_rewards():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    learners = {t: PolicyController(ActorCritic(), learn=True) for t in ("good", "evil")}
    games = [AvalonGame(rng) for _ in range(32)]
    recorder = TrajectoryRecorder()
    play(games, [learners["good"]] * 32, [learners["evil"]] * 32, rng, on_decision=recorder)
    batches = recorder.build(games, gamma=1.0, lam=1.0)
    # With gamma = lam = 1 every return equals the final reward of that player's team.
    expected = {"good": [], "evil": []}
    for (gi, player), steps in recorder.steps.items():
        evil = games[gi].is_evil_seat(player)
        reward = 1.0 if games[gi].good_won != evil else -1.0
        expected["evil" if evil else "good"] += [reward] * len(steps)
    for team in ("good", "evil"):
        assert np.allclose(batches[team].returns, expected[team])


def test_config_rejects_unknown_keys():
    with pytest.raises(KeyError):
        load_config(PILOT, {"ppo.learning_rate": 0.1})


def test_short_training_run_checkpoints_and_resumes(tmp_path):
    overrides = {
        "out_dir": str(tmp_path), "device": "cpu", "total_games": 64, "games_per_iter": 32,
        "checkpoint_every": 1, "eval.every": 2, "eval.games": 32, "opponents.snapshot_every": 1,
        "ppo.minibatch_size": 256,
    }
    run_dir = train(load_config(PILOT, overrides))
    lines = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[0]["iteration"] == 0 and "reference_rule_vs_rule" in lines[0]
    assert lines[-1]["iteration"] == 2 and "eval" in lines[-1]
    nets = load_policies(run_dir / "latest.pt")
    assert set(nets) == {"good", "evil"}

    train(load_config(PILOT, {**overrides, "total_games": 96, "resume": True}))
    lines = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[-1]["iteration"] == 3 and lines[-1]["games"] == 96


def test_fresh_start_overwrites_metrics_of_earlier_attempt(tmp_path):
    overrides = {
        "out_dir": str(tmp_path), "device": "cpu", "total_games": 64, "games_per_iter": 32,
        "eval.every": 2, "eval.games": 16, "ppo.minibatch_size": 256,
    }
    run_dir = train(load_config(PILOT, overrides))
    train(load_config(PILOT, overrides))  # same run dir, not a resume
    lines = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [entry["iteration"] for entry in lines] == [0, 1, 2]
