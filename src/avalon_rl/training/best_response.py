"""Approximate best response (BR): train one team against a frozen opponent to estimate exploitability.

    exploitability ~= P(BR evil wins | frozen good) + P(BR good wins | frozen evil) - 1

It is 0 at a Nash equilibrium of the two-team zero-sum game and grows the more a fixed policy can be beaten.
Because the best response is itself learned (finite budget, function approximation), the estimate is a lower
bound on the true exploitability (approximate exploitability, Timbers et al., IJCAI 2022). Compare policies only
under the same BR budget and initialisation.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..agents.policy import PolicyController
from ..env.game import AvalonGame
from ..evaluation.evaluate import run_matchup
from ..runner import play
from .checkpoint import TEAMS, build_net, comm_config, load_checkpoint, load_policies
from .ppo import PPOConfig, ppo_update
from .rollout import TrajectoryRecorder


@dataclass
class BestResponseConfig:
    games: int = 100_000  # BR training budget
    games_per_iter: int = 512
    eval_every: int = 40  # iterations between evaluations of the BR against the frozen target
    eval_games: int = 1000
    init: str = "target"  # target: start from the checkpoint's own network for this side; scratch: fresh network
    seed: int = 0
    device: str = "cpu"
    eval_seed: int = 20261020  # role assignments shared with scripts/evaluate.py


def _win_rate(summary: dict, side: str) -> float:
    good = summary["good_win"]["rate"]
    return good if side == "good" else 1.0 - good


def train_best_response(checkpoint: str | Path, side: str, cfg: BestResponseConfig,
                        out_path: str | Path | None = None) -> dict:
    """Train a `side` best response against the frozen other side of `checkpoint`; return a result dict."""
    if side not in TEAMS:
        raise ValueError(f"side must be one of {TEAMS}")
    other = "evil" if side == "good" else "good"
    device = torch.device(cfg.device)
    rng = np.random.default_rng([cfg.seed, TEAMS.index(side)])
    torch.manual_seed(cfg.seed)

    ckpt = load_checkpoint(checkpoint, "cpu")
    ppo = PPOConfig(**ckpt["config"]["ppo"])  # same PPO settings as the target's own training
    target = load_policies(checkpoint, "cpu")
    frozen = PolicyController(target[other], learn=False)

    comm = comm_config(ckpt["config"])
    net = build_net(ckpt["config"]).to(device)
    if cfg.init == "target":
        net.load_state_dict(ckpt["nets"][side])
    elif cfg.init != "scratch":
        raise ValueError("init must be 'target' or 'scratch'")
    optimizer = torch.optim.Adam(net.parameters(), lr=ppo.lr, eps=1e-5)
    rollout_net = net if device.type == "cpu" else copy.deepcopy(net).to("cpu")
    learner = PolicyController(rollout_net, learn=True)
    frozen_br = PolicyController(rollout_net, learn=False)

    def evaluate() -> float:
        good, evil = (frozen_br, frozen) if side == "good" else (frozen, frozen_br)
        summary, _ = run_matchup(good, evil, cfg.eval_games, cfg.eval_seed, comm=comm)
        return _win_rate(summary, side)

    curve = [(0, evaluate())]
    print(f"BR {side} vs frozen {other} of {checkpoint}: start win rate {curve[0][1]:.3f}", flush=True)
    n_iters = math.ceil(cfg.games / cfg.games_per_iter)
    for it in range(1, n_iters + 1):
        games = [AvalonGame(rng, comm=comm) for _ in range(cfg.games_per_iter)]
        mine, theirs = [learner] * len(games), [frozen] * len(games)
        good_ctrls, evil_ctrls = (mine, theirs) if side == "good" else (theirs, mine)
        recorder = TrajectoryRecorder()
        play(games, good_ctrls, evil_ctrls, rng, on_decision=recorder)
        batch = recorder.build(games, ppo.gamma, ppo.gae_lambda)[side]
        ppo_update(net, optimizer, batch, ppo, device, rng)
        if rollout_net is not net:
            rollout_net.load_state_dict(net.state_dict())
        if it % cfg.eval_every == 0 or it == n_iters:
            curve.append((it * cfg.games_per_iter, evaluate()))
            print(f"  iter {it}/{n_iters} games {it * cfg.games_per_iter}: BR win rate {curve[-1][1]:.3f}", flush=True)

    result = {
        "checkpoint": str(checkpoint),
        "target_iteration": ckpt["iteration"],
        "target_games": ckpt["games_seen"],
        "side": side,
        "start_win": curve[0][1],  # with init=target: this side's own policy vs the frozen target (self-play value)
        "final_win": curve[-1][1],
        "curve": curve,
        "config": asdict(cfg),
    }
    if out_path is not None:
        Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def exploitability(br_good: dict, br_evil: dict) -> dict:
    """Combine the two best-response results for the same checkpoint."""
    if br_good["checkpoint"] != br_evil["checkpoint"]:
        raise ValueError("best responses belong to different checkpoints")
    return {
        "checkpoint": br_good["checkpoint"],
        "target_games": br_good["target_games"],
        "br_good_win": br_good["final_win"],
        "br_evil_win": br_evil["final_win"],
        "exploitability": br_good["final_win"] + br_evil["final_win"] - 1.0,
        "selfplay_good_win": br_good["start_win"] if br_good["config"]["init"] == "target" else None,
    }
