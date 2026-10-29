"""Self-play PPO training: the silent baseline S and the communication configurations (cfg.comm).

Each iteration plays `games_per_iter` games, turns the learning players' decisions into one
batch per team, and updates the good and the evil network separately (so the 4-vs-2 team sizes
never weight each other's losses). Metrics go to <run_dir>/metrics.jsonl, one JSON object per line.
A communication configuration normally starts from a silent checkpoint (`init_from`, the branch point):
the silent weights are embedded so that the network initially plays exactly like S and ignores messages.
The optional MMD regulariser (ppo.py) belongs to an archived experiment.
"""
from __future__ import annotations

import copy
import json
import math
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from ..agents.policy import ActorCritic, PolicyController, load_silent_weights
from ..agents.rule_agent import RuleController
from ..config import TrainConfig, config_to_dict
from ..env.game import AvalonGame
from ..evaluation.evaluate import run_matchup
from ..evaluation.metrics import game_outcome, summarize
from ..runner import play
from .checkpoint import comm_config, load_checkpoint, save_checkpoint
from .gpu_lock import gpu_turn
from .opponent_pool import OpponentPool
from .ppo import ppo_update
from .rollout import TEAMS, TrajectoryRecorder, message_usage

MODES = ("self", "pool", "rule")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_info(device: torch.device) -> dict:
    """Where and with which code a run (or resumption) started; appended to run_info.jsonl."""
    info = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    repo = Path(__file__).resolve().parents[3]
    git = lambda *args: subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    try:
        info["git_commit"] = git("rev-parse", "HEAD")
        info["git_dirty"] = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        info["git_commit"] = info["git_dirty"] = None
    return info


def sample_matchups(n_games, learners, pool, rule, probs, rng):
    """Choose who controls each team in each training game."""
    good, evil, modes = [], [], []
    for _ in range(n_games):
        mode = MODES[rng.choice(3, p=probs)]
        ctrls = dict(learners)
        if mode != "self":
            frozen_team = TEAMS[int(rng.integers(2))]
            opponent = rule if mode == "rule" else pool.sample(frozen_team, rng)
            if opponent is None:  # pool still empty early in training
                mode = "self"
            else:
                ctrls[frozen_team] = opponent
        good.append(ctrls["good"])
        evil.append(ctrls["evil"])
        modes.append(mode)
    return good, evil, modes


def _rate(summary: dict, key: str = "good_win") -> float:
    return summary[key]["rate"]


def train(cfg: TrainConfig) -> Path:
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    nets = {t: ActorCritic(hidden_sizes=cfg.model.hidden_sizes, comm=cfg.comm.enabled).to(device) for t in TEAMS}
    opts = {t: torch.optim.Adam(nets[t].parameters(), lr=cfg.ppo.lr, eps=1e-5) for t in TEAMS}
    iteration = games_seen = 0
    latest = run_dir / "latest.pt"
    if cfg.resume and latest.exists():
        ckpt = load_checkpoint(latest, device)
        for t in TEAMS:
            nets[t].load_state_dict(ckpt["nets"][t])
            opts[t].load_state_dict(ckpt["optimizers"][t])
        iteration, games_seen = ckpt["iteration"], ckpt["games_seen"]
        rng.bit_generator.state = ckpt["numpy_rng"]
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        print(f"resumed from {latest} at iteration {iteration} (opponent pool restarts empty)")
    elif cfg.init_from:
        init_from = cfg.init_from.format(seed=cfg.seed)
        ckpt = load_checkpoint(init_from, device)
        branch = cfg.comm.enabled and not comm_config(ckpt["config"]).enabled
        for t in TEAMS:
            if branch:
                load_silent_weights(nets[t], ckpt["nets"][t])
            else:
                nets[t].load_state_dict(ckpt["nets"][t])
        print(f"initialised network weights from {init_from}" + (" (silent branch point)" if branch else ""))

    # MMD moving magnet: a frozen copy of each team's policy, refreshed every magnet_update_every iterations.
    # It starts as the initial policy (near-uniform for a fresh network) and is restored exactly on resume.
    magnets = None
    if cfg.mmd.enabled and cfg.mmd.magnet == "moving":
        magnets = {t: copy.deepcopy(nets[t]).eval() for t in TEAMS}
        for t in TEAMS:
            magnets[t].requires_grad_(False)
            if cfg.resume and latest.exists() and "magnets" in ckpt:
                magnets[t].load_state_dict(ckpt["magnets"][t])
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False), encoding="utf-8")
    with open(run_dir / "run_info.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"iteration": iteration, **run_info(device)}) + "\n")

    # Rollouts run on CPU copies: per-tick batches are small, so GPU transfers would dominate.
    rollout_nets = nets if device.type == "cpu" else {t: copy.deepcopy(nets[t]).to("cpu") for t in TEAMS}
    learners = {t: PolicyController(rollout_nets[t], learn=True) for t in TEAMS}
    frozen_views = {t: PolicyController(rollout_nets[t], learn=False) for t in TEAMS}
    rule = RuleController(cfg.opponents.rule_fail_prob, cfg.opponents.rule_merlin_knowledge_prob)
    pool = OpponentPool(cfg.opponents.pool_size)
    probs = [cfg.opponents.p_self, cfg.opponents.p_pool, cfg.opponents.p_rule]
    n_iters = math.ceil(cfg.total_games / cfg.games_per_iter)
    eval_seed = 10_000 + cfg.seed  # fixed, so evaluations across iterations are paired

    # A fresh start replaces the metrics of any earlier attempt in this run dir; a resume appends.
    metrics_path = run_dir / "metrics.jsonl"
    if iteration == 0 and metrics_path.exists():
        print(f"warning: starting from scratch, overwriting {metrics_path} (use --resume to continue)")
    with open(metrics_path, "a" if iteration > 0 else "w", encoding="utf-8") as log:
        if iteration == 0:
            ref, _ = run_matchup(rule, rule, cfg.eval.games, eval_seed, comm=cfg.comm)
            log.write(json.dumps({"iteration": 0, "reference_rule_vs_rule": ref}) + "\n")
            print(f"reference: rule vs rule good win rate {_rate(ref):.3f}")

        while iteration < n_iters:
            t0 = time.perf_counter()
            games = [AvalonGame(rng, comm=cfg.comm) for _ in range(cfg.games_per_iter)]
            good_ctrls, evil_ctrls, modes = sample_matchups(len(games), learners, pool, rule, probs, rng)
            recorder = TrajectoryRecorder()
            play(games, good_ctrls, evil_ctrls, rng, on_decision=recorder)
            batches = recorder.build(games, cfg.ppo.gamma, cfg.ppo.gae_lambda)
            t1 = time.perf_counter()

            entry: dict = {}
            with gpu_turn(device):  # runs sharing a GPU take turns (see gpu_lock)
                for t in TEAMS:
                    if len(batches[t]):
                        stats = ppo_update(nets[t], opts[t], batches[t], cfg.ppo, device, rng,
                                           mmd=cfg.mmd, magnet=magnets[t] if magnets else None)
                        entry.update({f"{t}/{k}": v for k, v in stats.items()})
                        entry.update({f"{t}/says_{k}": v for k, v in message_usage(batches[t]).items()})
                    if rollout_nets[t] is not nets[t]:
                        rollout_nets[t].load_state_dict(nets[t].state_dict())
            t2 = time.perf_counter()
            iteration += 1
            games_seen += len(games)

            self_games = [game_outcome(g) for g, m in zip(games, modes) if m == "self"]
            entry = {
                "iteration": iteration,
                "games": games_seen,
                "collect_s": round(t1 - t0, 3),
                "update_s": round(t2 - t1, 3),
                "decisions": int(sum(len(b) for b in batches.values())),
                "modes": {m: modes.count(m) for m in MODES},
                "selfplay": summarize(self_games) if self_games else None,
                **entry,
            }
            if iteration % cfg.opponents.snapshot_every == 0:
                for t in TEAMS:
                    pool.add(t, rollout_nets[t])
            if magnets and iteration % cfg.mmd.magnet_update_every == 0:
                for t in TEAMS:
                    magnets[t].load_state_dict(nets[t].state_dict())
                entry["magnet_updated"] = True
            if iteration % cfg.eval.every == 0 or iteration == n_iters:
                good_vs_rule, _ = run_matchup(frozen_views["good"], rule, cfg.eval.games, eval_seed, comm=cfg.comm)
                rule_vs_evil, _ = run_matchup(rule, frozen_views["evil"], cfg.eval.games, eval_seed, comm=cfg.comm)
                entry["eval"] = {"policy_good_vs_rule_evil": good_vs_rule, "rule_good_vs_policy_evil": rule_vs_evil}
            if iteration % cfg.checkpoint_every == 0 or iteration == n_iters:
                state = dict(
                    iteration=iteration, games_seen=games_seen, config=config_to_dict(cfg),
                    nets={t: nets[t].state_dict() for t in TEAMS},
                    optimizers={t: opts[t].state_dict() for t in TEAMS},
                    numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                )
                if magnets:
                    state["magnets"] = {t: magnets[t].state_dict() for t in TEAMS}
                with gpu_turn(device):  # saving copies the weights off the GPU
                    save_checkpoint(run_dir / f"checkpoint_{iteration:05d}.pt", **state)
                    save_checkpoint(latest, **state)
            log.write(json.dumps(entry) + "\n")
            log.flush()

            line = f"iter {iteration}/{n_iters} games {games_seen} collect {t1 - t0:.1f}s update {t2 - t1:.1f}s"
            if entry["selfplay"]:
                line += f" | self-play good win {_rate(entry['selfplay']):.3f}"
            if "eval" in entry:
                line += (f" | good-vs-rule {_rate(entry['eval']['policy_good_vs_rule_evil']):.3f}"
                         f" evil-vs-rule {1 - _rate(entry['eval']['rule_good_vs_policy_evil']):.3f}")
            print(line, flush=True)
    return run_dir
