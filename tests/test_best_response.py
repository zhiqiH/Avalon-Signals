import math
from pathlib import Path

from src.config import load_config
from src.training.best_response import BestResponseConfig, exploitability, train_best_response
from src.training.trainer import train

PILOT = Path(__file__).resolve().parents[1] / "configs" / "pilot_silent.yaml"


def _short_run(tmp_path):
    overrides = {
        "out_dir": str(tmp_path), "device": "cpu", "total_games": 64, "games_per_iter": 32,
        "checkpoint_every": 1, "eval.every": 2, "eval.games": 16, "ppo.minibatch_size": 256,
    }
    return train(load_config(PILOT, overrides))


def test_best_response_and_exploitability(tmp_path):
    run_dir = _short_run(tmp_path)
    cfg = BestResponseConfig(games=64, games_per_iter=32, eval_every=1, eval_games=32)
    results = {}
    for side in ("good", "evil"):
        out = tmp_path / f"br_{side}.json"
        results[side] = train_best_response(run_dir / "latest.pt", side, cfg, out_path=out)
        assert out.exists() and len(results[side]["curve"]) == 3
        assert 0.0 <= results[side]["final_win"] <= 1.0
    e = exploitability(results["good"], results["evil"])
    assert math.isclose(e["exploitability"], results["good"]["final_win"] + results["evil"]["final_win"] - 1)
