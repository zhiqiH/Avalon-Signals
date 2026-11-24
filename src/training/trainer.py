"""Self-play PPO training: the silent baseline S and the communication configurations (cfg.comm).

Each iteration plays `games_per_iter` games, turns the learning players' decisions into one
batch per team, and updates the good and the evil network separately (so the 4-vs-2 team sizes
never weight each other's losses). Metrics go to <run_dir>/metrics.jsonl, one JSON object per line.
A communication configuration normally starts from a silent checkpoint (`init_from`, the branch point):
the silent weights are embedded so that the network initially plays exactly like S and ignores messages.
"""
from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from ..agents.logic_agent import LogicController
from ..agents.policy import ActorCritic, PolicyController, is_legacy_state, load_silent_weights
from ..agents.rule_agent import RuleController
from ..config import TrainConfig, config_to_dict
from ..env.game import AvalonGame
from ..evaluation.evaluate import run_matchup
from ..evaluation.metrics import game_outcome, summarize
from ..runner import play
from .adapt import adapter_state, apply_lora, plain_state_dict, trainable_parameters
from .checkpoint import comm_config, load_checkpoint, load_policies, save_checkpoint
from .opponent_pool import OpponentPool
from .ppo import ppo_update
from .rollout import TEAMS, TrajectoryRecorder, message_usage
from .workers import (LEARNER, PARTNER, RULE, RolloutWorkers, Spec, frozen_speakers, partner_seats, resolve,
                      teacher_seats)

MODES = ("self", "pool", "rule", "partner")


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
    repo = Path(__file__).resolve().parents[2]
    git = lambda *args: subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    try:
        info["git_commit"] = git("rev-parse", "HEAD")
        info["git_dirty"] = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        info["git_commit"] = info["git_dirty"] = None
    return info


def sample_matchups(n_games: int, pool_sizes: dict[str, int], probs, rng) -> tuple[list[tuple[Spec, Spec]], list[str]]:
    """Choose who controls each team in each training game: (good spec, evil spec) per game, and its mode.

    A spec is LEARNER, RULE, PARTNER (good team only) or an index into that team's opponent pool
    (see `workers.resolve`).
    """
    specs, modes = [], []
    probs = list(probs) + [0.0] * (len(MODES) - len(probs))  # callers predating the partner mode
    for _ in range(n_games):
        mode = MODES[rng.choice(len(MODES), p=probs)]
        spec = {"good": LEARNER, "evil": LEARNER}
        if mode == "partner":
            spec["good"] = PARTNER
        elif mode != "self":
            frozen_team = TEAMS[int(rng.integers(2))]
            if mode == "rule":
                spec[frozen_team] = RULE
            elif pool_sizes[frozen_team]:
                spec[frozen_team] = int(rng.integers(pool_sizes[frozen_team]))
            else:  # pool still empty early in training
                mode = "self"
        specs.append((spec["good"], spec["evil"]))
        modes.append(mode)
    return specs, modes


def refill_pool(pool: OpponentPool, run_dir: Path, iteration: int, every: int, size: int) -> list[dict]:
    """On a resume, refill the opponent pool with the saved checkpoints among the last `size` snapshots.

    Snapshots are taken more often than checkpoints are saved, so only the snapshots that have a checkpoint
    file come back (half of them with the default settings); the pool fills up again as training continues.
    Returns the restored networks, oldest first.
    """
    restored = []
    for it in range(iteration - (iteration % every), max(0, iteration - size * every), -every):
        path = run_dir / f"checkpoint_{it:05d}.pt"
        if it > 0 and path.exists():
            restored.append((it, load_policies(path)))
    for it, nets in reversed(restored):
        for t in TEAMS:
            pool.add(t, nets[t])
    its = [it for it, _ in restored]
    print(f"opponent pool refilled with {len(its)} snapshots" + (f" (iterations {min(its)}-{max(its)})" if its else ""))
    return [nets for _, nets in reversed(restored)]


def _rate(summary: dict, key: str = "good_win") -> float:
    return summary[key]["rate"]


def train(cfg: TrainConfig) -> Path:
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    nets = {t: ActorCritic(hidden_sizes=cfg.model.hidden_sizes, comm=cfg.comm.enabled,
                           deduction=cfg.model.deduction).to(device) for t in TEAMS}
    iteration = games_seen = 0
    latest = run_dir / "latest.pt"
    ckpt = None
    legacy = False  # the starting network predates the reply and promise slots: its adapters do not fit
    if cfg.resume and latest.exists():
        ckpt = load_checkpoint(latest, device)
        for t in TEAMS:
            nets[t].load_state_dict(ckpt["nets"][t])
        iteration, games_seen = ckpt["iteration"], ckpt["games_seen"]
        rng.bit_generator.state = ckpt["numpy_rng"]
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        print(f"resumed from {latest} at iteration {iteration}")
    elif cfg.init_from:
        init_from = cfg.init_from.format(seed=cfg.seed)
        ckpt = load_checkpoint(init_from, device)
        branch = cfg.comm.enabled and not comm_config(ckpt["config"]).enabled
        legacy = cfg.comm.enabled and is_legacy_state(ckpt["nets"]["good"])
        for t in TEAMS:
            if branch:
                load_silent_weights(nets[t], ckpt["nets"][t])
            else:
                nets[t].load_state_dict(ckpt["nets"][t])
        print(f"initialised network weights from {init_from}" + (" (silent branch point)" if branch else "")
              + (" (before the reply and promise slots: new parts start at zero)" if legacy else ""))
    if cfg.adapt.lora_rank > 0:
        for t in TEAMS:
            apply_lora(nets[t], cfg.adapt.lora_rank, cfg.adapt.lora_scale).to(device)
            if ckpt is not None and ckpt.get("adapter") and not legacy:  # same adapters as the run it resumes or continues
                nets[t].load_state_dict(ckpt["adapter"][t])
        n_train = sum(p.numel() for p in trainable_parameters(nets["good"]))
        print(f"LoRA rank {cfg.adapt.lora_rank}: {n_train:,} trainable parameters per team"
              + (" (adapters continued from the starting checkpoint)" if ckpt is not None and ckpt.get("adapter")
                 and not legacy else ""))
    opts = {t: torch.optim.Adam(trainable_parameters(nets[t]), lr=cfg.ppo.lr, eps=1e-5) for t in TEAMS}
    if cfg.resume and iteration > 0:
        for t in TEAMS:
            opts[t].load_state_dict(ckpt["optimizers"][t])
    anchors = None
    if cfg.adapt.msg_kl_coef > 0:
        anchors = load_policies(cfg.adapt.anchor_from.format(seed=cfg.seed), device)
        for net in anchors.values():
            net.eval().requires_grad_(False)

    (run_dir / "config.yaml").write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False), encoding="utf-8")
    with open(run_dir / "run_info.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"iteration": iteration, **run_info(device)}) + "\n")

    # Rollouts run on plain CPU copies: per-tick batches are small, so GPU transfers would dominate, and
    # adapted networks are merged once per update instead of on every forward pass.
    if device.type == "cpu" and cfg.adapt.lora_rank == 0:
        rollout_nets = nets
    else:
        rollout_nets = {t: ActorCritic(hidden_sizes=cfg.model.hidden_sizes, comm=cfg.comm.enabled,
                                       deduction=cfg.model.deduction) for t in TEAMS}
        for t in TEAMS:
            rollout_nets[t].load_state_dict({k: v.cpu() for k, v in plain_state_dict(nets[t]).items()})
    speakers = frozen_speakers(cfg)
    if speakers:
        print(f"listener stage: every learned player speaks with the frozen networks of {cfg.adapt.speaker_from.format(seed=cfg.seed)}")
    learners = {t: PolicyController(rollout_nets[t], learn=True, speaker=speakers.get(t)) for t in TEAMS}
    frozen_views = {t: PolicyController(rollout_nets[t], learn=False, speaker=speakers.get(t)) for t in TEAMS}
    rule = RuleController(cfg.opponents.rule_fail_prob, cfg.opponents.rule_merlin_knowledge_prob)
    partner = LogicController(cfg.opponents.partner_slack)
    pool = OpponentPool(cfg.opponents.pool_size, speakers)
    restored = refill_pool(pool, run_dir, iteration, cfg.opponents.snapshot_every, cfg.opponents.pool_size) if iteration else []
    with RolloutWorkers(cfg.workers, cfg) if cfg.workers else nullcontext() as workers:
        if workers:
            print(f"{len(workers)} rollout worker processes")
            for nets_then in restored:
                workers.pool_add(nets_then)
            workers.set_weights(nets)
        _train_loop(cfg, run_dir, device, rng, nets, opts, rollout_nets, learners, frozen_views, rule, pool, workers,
                    iteration, games_seen, anchors, partner)
    return run_dir


def msg_kl_coef(cfg: TrainConfig, iteration: int) -> float:
    """Weight of the message KL anchor at an iteration: constant, or falling linearly over `msg_kl_anneal`
    iterations to `msg_kl_floor` (0 by default), where it stays."""
    a = cfg.adapt
    if a.msg_kl_anneal <= 0:
        return a.msg_kl_coef
    return max(a.msg_kl_floor, a.msg_kl_coef * max(0.0, 1.0 - iteration / a.msg_kl_anneal))


def _train_loop(cfg, run_dir, device, rng, nets, opts, rollout_nets, learners, frozen_views, rule, pool, workers,
                iteration, games_seen, anchors=None, partner=None) -> None:
    latest = run_dir / "latest.pt"
    probs = [cfg.opponents.p_self, cfg.opponents.p_pool, cfg.opponents.p_rule, cfg.opponents.p_partner]
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
            pool_sizes = {t: len(pool.snapshots[t]) for t in TEAMS}
            if workers:
                specs, modes = sample_matchups(cfg.games_per_iter, pool_sizes, probs, rng)
                batches, outcomes = workers.play(specs, rng)
            else:
                games = [AvalonGame(rng, comm=cfg.comm) for _ in range(cfg.games_per_iter)]
                specs, modes = sample_matchups(len(games), pool_sizes, probs, rng)
                seats = partner_seats(games, specs, rng, partner, cfg.opponents.partner_max)
                recorder = TrajectoryRecorder()
                play(games, [resolve(g, "good", learners, pool, rule) for g, _ in specs],
                     [resolve(e, "evil", learners, pool, rule) for _, e in specs], rng, on_decision=recorder,
                     seat_ctrls=seats, seat_speakers=teacher_seats(cfg, seats, rng))
                batches = recorder.build(games, cfg.ppo.gamma, cfg.ppo.gae_lambda)
                outcomes = [game_outcome(g) for g in games]
            t1 = time.perf_counter()

            entry: dict = {}
            kl_coef = msg_kl_coef(cfg, iteration) if anchors else 0.0
            if anchors:
                entry["msg_kl_coef"] = round(kl_coef, 6)
            for t in TEAMS:
                if len(batches[t]):
                    stats = ppo_update(nets[t], opts[t], batches[t], cfg.ppo, device, rng,
                                       ref_net=anchors[t] if anchors else None, msg_kl_coef=kl_coef)
                    entry.update({f"{t}/{k}": v for k, v in stats.items()})
                    entry.update({f"{t}/says_{k}": v for k, v in message_usage(batches[t]).items()})
                if rollout_nets[t] is not nets[t]:
                    rollout_nets[t].load_state_dict({k: v.cpu() for k, v in plain_state_dict(nets[t]).items()})
            if workers:
                workers.set_weights(nets)
            t2 = time.perf_counter()
            iteration += 1
            games_seen += len(specs)

            self_games = [o for o, m in zip(outcomes, modes) if m == "self"]
            partner_games = [o for o, m in zip(outcomes, modes) if m == "partner"]
            entry = {
                "iteration": iteration,
                "games": games_seen,
                "collect_s": round(t1 - t0, 3),
                "update_s": round(t2 - t1, 3),
                "decisions": int(sum(len(b) for b in batches.values())),
                "modes": {m: modes.count(m) for m in MODES},
                "selfplay": summarize(self_games) if self_games else None,
                **({"partner": summarize(partner_games)} if partner_games else {}),
                **entry,
            }
            if iteration % cfg.opponents.snapshot_every == 0:
                for t in TEAMS:
                    pool.add(t, rollout_nets[t])
                if workers:
                    workers.pool_add_current()
            if iteration % cfg.eval.every == 0 or iteration == n_iters:
                if workers:
                    good_vs_rule = workers.evaluate(LEARNER, RULE, cfg.eval.games, eval_seed)
                    rule_vs_evil = workers.evaluate(RULE, LEARNER, cfg.eval.games, eval_seed)
                else:
                    good_vs_rule, _ = run_matchup(frozen_views["good"], rule, cfg.eval.games, eval_seed, comm=cfg.comm)
                    rule_vs_evil, _ = run_matchup(rule, frozen_views["evil"], cfg.eval.games, eval_seed, comm=cfg.comm)
                entry["eval"] = {"policy_good_vs_rule_evil": good_vs_rule, "rule_good_vs_policy_evil": rule_vs_evil}
            if iteration % cfg.checkpoint_every == 0 or iteration == n_iters:
                state = dict(
                    iteration=iteration, games_seen=games_seen, config=config_to_dict(cfg),
                    nets={t: plain_state_dict(nets[t]) for t in TEAMS},  # merged weights: every tool reads these
                    optimizers={t: opts[t].state_dict() for t in TEAMS},
                    numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                )
                if cfg.adapt.lora_rank > 0:
                    state["adapter"] = {t: adapter_state(nets[t]) for t in TEAMS}
                save_checkpoint(run_dir / f"checkpoint_{iteration:05d}.pt", **state)
                save_checkpoint(latest, **state)
            log.write(json.dumps(entry) + "\n")
            log.flush()

            line = f"iter {iteration}/{n_iters} games {games_seen} collect {t1 - t0:.1f}s update {t2 - t1:.1f}s"
            if entry["selfplay"]:
                line += f" | self-play good win {_rate(entry['selfplay']):.3f}"
            if partner_games:
                line += f" | partner games good win {_rate(entry['partner']):.3f}"
            if "eval" in entry:
                line += (f" | good-vs-rule {_rate(entry['eval']['policy_good_vs_rule_evil']):.3f}"
                         f" evil-vs-rule {1 - _rate(entry['eval']['rule_good_vs_policy_evil']):.3f}")
            print(line, flush=True)
