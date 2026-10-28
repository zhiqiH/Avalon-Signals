import json
import math
from pathlib import Path

import torch

from avalon_rl.agents.policy import masked_logits
from avalon_rl.config import load_config
from avalon_rl.training import magnet_kl
from avalon_rl.training.best_response import BestResponseConfig, exploitability, train_best_response
from avalon_rl.training.checkpoint import load_checkpoint
from avalon_rl.training.trainer import train

PILOT = Path(__file__).resolve().parents[1] / "configs" / "pilot_silent.yaml"


def _logp(logits, mask):
    return torch.log_softmax(masked_logits(logits, mask), dim=-1)


def test_magnet_kl_is_zero_for_identical_policies_and_positive_otherwise():
    mask = torch.tensor([[True, True, False, True]])
    logits = torch.tensor([[0.3, -1.0, 5.0, 2.0]])
    assert abs(magnet_kl(_logp(logits, mask), logits, mask).item()) < 1e-6
    assert magnet_kl(_logp(logits, mask), torch.zeros_like(logits), mask).item() > 0


def test_uniform_magnet_kl_equals_log_legal_actions_minus_entropy():
    mask = torch.tensor([[True, True, False, True, False]])
    logits = torch.tensor([[1.0, -0.5, 9.0, 0.2, 9.0]])
    logp = _logp(logits, mask)
    entropy = -(logp.exp() * logp).nan_to_num().sum()
    assert math.isclose(magnet_kl(logp, None, mask).item(), math.log(3) - entropy.item(), rel_tol=1e-5)


def test_magnet_kl_stays_finite_when_the_magnet_almost_excludes_a_legal_action():
    mask = torch.tensor([[True, True, True]])
    policy = _logp(torch.tensor([[0.0, 0.0, 0.0]]), mask)
    magnet_logits = torch.tensor([[0.0, 0.0, -500.0]])  # probability underflows to 0 in float32
    kl = magnet_kl(policy, magnet_logits, mask)
    assert torch.isfinite(kl) and kl.item() > 100


def _short_mmd_run(tmp_path, **extra):
    overrides = {
        "out_dir": str(tmp_path), "device": "cpu", "total_games": 64, "games_per_iter": 32,
        "checkpoint_every": 1, "eval.every": 2, "eval.games": 16, "ppo.minibatch_size": 256,
        "mmd.enabled": True, "mmd.alpha": 0.1, "mmd.magnet_update_every": 1, **extra,
    }
    return train(load_config(PILOT, overrides)), overrides


def test_mmd_training_logs_magnet_kl_and_checkpoints_the_magnet(tmp_path):
    run_dir, overrides = _short_mmd_run(tmp_path)
    lines = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "good/magnet_kl" in lines[1] and lines[1].get("magnet_updated")
    ckpt = load_checkpoint(run_dir / "latest.pt")
    assert set(ckpt["magnets"]) == {"good", "evil"}

    train(load_config(PILOT, {**overrides, "total_games": 96, "resume": True}))
    lines = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[-1]["iteration"] == 3


def test_uniform_magnet_needs_no_magnet_network(tmp_path):
    run_dir, _ = _short_mmd_run(tmp_path, **{"mmd.magnet": "uniform"})
    assert "magnets" not in load_checkpoint(run_dir / "latest.pt")


def test_best_response_and_exploitability(tmp_path):
    run_dir, _ = _short_mmd_run(tmp_path)
    cfg = BestResponseConfig(games=64, games_per_iter=32, eval_every=1, eval_games=32)
    results = {}
    for side in ("good", "evil"):
        out = tmp_path / f"br_{side}.json"
        results[side] = train_best_response(run_dir / "latest.pt", side, cfg, out_path=out)
        assert out.exists() and len(results[side]["curve"]) == 3
        assert 0.0 <= results[side]["final_win"] <= 1.0
    e = exploitability(results["good"], results["evil"])
    assert math.isclose(e["exploitability"], results["good"]["final_win"] + results["evil"]["final_win"] - 1)
