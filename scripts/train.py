"""Train a configuration with self-play PPO.

    python scripts/train.py --config configs/baseline_silent.yaml --seed 0
    python scripts/train.py --config configs/comm_F.yaml --seed 0 1 2 --device cuda
    python scripts/train.py --config configs/comm_M.yaml --seed 0 1 2 --resume
    python scripts/train.py --config configs/pilot_silent.yaml --device cpu
    python scripts/train.py --config configs/baseline_silent.yaml --set ppo.lr=1e-4 --set ppo.ent_coef=0.02

With several seeds, every seed trains in its own process at the same time and writes its console output to
<run dir>/train.log.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root, so `import src` works

from src.config import load_config
from src.training.trainer import train


def parse_set(items: list[str]) -> dict:
    overrides = {}
    for item in items:
        key, _, raw = item.partition("=")
        if not key or not _:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        overrides[key] = yaml.safe_load(raw)
    return overrides


def overrides_of(args: argparse.Namespace, seed: int | None) -> dict:
    return {
        "seed": seed, "device": args.device, "total_games": args.total_games, "name": args.name,
        "out_dir": args.out_dir, "init_from": args.init_from, "resume": args.resume or None,
        **parse_set(args.set),
    }


def train_seeds_in_parallel(args: argparse.Namespace) -> int:
    """One process per seed, all at once; returns the number of seeds that failed."""
    threads = args.threads or max(1, (os.cpu_count() or 2) // (2 * len(args.seed)))  # share the physical cores
    shared = ["--config", args.config, "--threads", str(threads)]
    for flag, value in (("--device", args.device), ("--total-games", args.total_games), ("--name", args.name),
                        ("--out-dir", args.out_dir), ("--init-from", args.init_from)):
        if value is not None:
            shared += [flag, str(value)]
    shared += ["--resume"] if args.resume else []
    for item in args.set:
        shared += ["--set", item]

    jobs = []
    for seed in args.seed:
        run_dir = load_config(args.config, overrides_of(args, seed)).run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        log = open(run_dir / "train.log", "a", encoding="utf-8")
        cmd = [sys.executable, "-u", os.path.abspath(__file__), *shared, "--seed", str(seed)]
        jobs.append((seed, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        print(f"seed {seed}: pid {jobs[-1][1].pid}, log {run_dir / 'train.log'}", flush=True)
    failed = 0
    try:
        for seed, proc, log in jobs:
            code = proc.wait()
            log.close()
            failed += code != 0
            print(f"seed {seed}: {'finished' if code == 0 else f'failed (exit code {code})'}", flush=True)
    except KeyboardInterrupt:
        for _, proc, _ in jobs:
            proc.terminate()
        raise
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/baseline_silent.yaml")
    parser.add_argument("--seed", type=int, nargs="+", help="one or more seeds; several seeds train in parallel")
    parser.add_argument("--device", help="auto | cpu | cuda | mps")
    parser.add_argument("--total-games", type=int)
    parser.add_argument("--name")
    parser.add_argument("--out-dir")
    parser.add_argument("--init-from", help="checkpoint whose network weights initialise training")
    parser.add_argument("--resume", action="store_true", help="continue from the run's latest.pt")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override any config key")
    parser.add_argument("--threads", type=int,
                        help="torch CPU threads per run (default with several seeds: an even share of the cores)")
    args = parser.parse_args()

    if args.seed and len(args.seed) > 1:
        sys.exit(train_seeds_in_parallel(args))
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = load_config(args.config, overrides_of(args, args.seed[0] if args.seed else None))
    print(f"run dir: {cfg.run_dir}")
    train(cfg)


if __name__ == "__main__":
    main()
