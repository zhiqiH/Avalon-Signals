import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.agents import ActorCritic, PolicyController, RandomController, Request, RuleController
from src.config import load_config
from src.env import AvalonGame
from src.evaluation import run_matchup
from src.runner import play
from src.training import TrajectoryRecorder, gae_terminal
from src.training.checkpoint import load_policies
from src.training.opponent_pool import OpponentPool
from src.training.trainer import refill_pool, train
from src.training.workers import LEARNER, RolloutWorkers, cpu_state

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


def test_resume_refills_the_pool_from_saved_snapshots(tmp_path):
    overrides = {
        "out_dir": str(tmp_path), "device": "cpu", "total_games": 128, "games_per_iter": 32,
        "checkpoint_every": 2, "eval.every": 4, "eval.games": 32, "ppo.minibatch_size": 256,
    }
    run_dir = train(load_config(PILOT, overrides))  # checkpoints at iterations 2 and 4
    pool = OpponentPool(capacity=3)
    refill_pool(pool, run_dir, iteration=4, every=1, size=3)  # snapshots 2, 3, 4; no file for 3
    assert len(pool.snapshots["good"]) == len(pool.snapshots["evil"]) == 2
    newest = load_policies(run_dir / "checkpoint_00004.pt")["good"]
    restored = pool.snapshots["good"][-1].net
    assert all(torch.equal(a, b) for a, b in zip(newest.state_dict().values(), restored.state_dict().values()))


def test_training_with_rollout_workers_is_reproducible_and_resumes(tmp_path):
    overrides = {
        "device": "cpu", "total_games": 96, "games_per_iter": 32, "checkpoint_every": 1, "eval.every": 3,
        "eval.games": 32, "opponents.snapshot_every": 1, "ppo.minibatch_size": 256, "workers": 2,
    }
    runs = []
    for name in ("a", "b"):
        run_dir = train(load_config(PILOT, {**overrides, "out_dir": str(tmp_path / name)}))
        runs.append([json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()])
    timing = ("collect_s", "update_s")
    assert [{k: v for k, v in e.items() if k not in timing} for e in runs[0]] == \
        [{k: v for k, v in e.items() if k not in timing} for e in runs[1]]
    last = runs[0][-1]
    assert last["iteration"] == 3 and last["games"] == 96 and "eval" in last
    assert last["eval"]["policy_good_vs_rule_evil"]["games"] == 32
    assert sum(last["modes"].values()) == 32 and last["modes"]["pool"] > 0  # pool snapshots reached the workers

    run_dir = train(load_config(PILOT, {**overrides, "out_dir": str(tmp_path / "a"), "total_games": 128, "resume": True}))
    lines = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["iteration"] for e in lines[1:]] == [1, 2, 3, 4]


def test_worker_messages_carry_no_torch_tensors():
    # torch tensors would be sent as shared-memory file descriptors and exhaust open files on a pool refill
    state = cpu_state(ActorCritic())
    assert state and all(isinstance(v, np.ndarray) for v in state.values())


def test_a_failing_rollout_worker_raises_in_the_training_process():
    with RolloutWorkers(1, load_config(PILOT)) as workers:
        with pytest.raises(RuntimeError, match="IndexError"):
            workers.play([(0, LEARNER)], np.random.default_rng(0))  # pool index 0 of an empty pool


def test_fresh_start_overwrites_metrics_of_earlier_attempt(tmp_path):
    overrides = {
        "out_dir": str(tmp_path), "device": "cpu", "total_games": 64, "games_per_iter": 32,
        "eval.every": 2, "eval.games": 16, "ppo.minibatch_size": 256,
    }
    run_dir = train(load_config(PILOT, overrides))
    train(load_config(PILOT, overrides))  # same run dir, not a resume
    lines = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [entry["iteration"] for entry in lines] == [0, 1, 2]



def test_networks_without_deduction_ignore_the_deduction_fields():
    from src.env import OBS_DIM
    from src.env.observation import OBS_DIM_NO_DEDUCTION
    from src.training.checkpoint import build_net

    torch.manual_seed(0)
    old = build_net({"model": {"hidden_sizes": [16]}})  # config written before the deduction fields existed
    assert old.obs_dim == OBS_DIM_NO_DEDUCTION and old.actor[0].in_features == OBS_DIM_NO_DEDUCTION
    obs = torch.randn(4, OBS_DIM)
    noisy = obs.clone()
    noisy[:, OBS_DIM_NO_DEDUCTION:] = torch.randn(4, OBS_DIM - OBS_DIM_NO_DEDUCTION)
    assert torch.equal(old.logits(obs), old.logits(noisy)) and torch.equal(old.value(obs), old.value(noisy))
    assert build_net({"model": {"hidden_sizes": [16], "deduction": True}}).actor[0].in_features == OBS_DIM
