"""Train the silent baseline S with self-play PPO.

    python scripts/train.py --config configs/baseline_silent.yaml --seed 0
    python scripts/train.py --config configs/pilot_silent.yaml --device cpu
    python scripts/train.py --config configs/baseline_silent.yaml --seed 0 --resume
    python scripts/train.py --config configs/baseline_silent.yaml --set ppo.lr=1e-4 --set ppo.ent_coef=0.02
"""
from __future__ import annotations

import argparse

import torch
import yaml

from avalon_rl.config import load_config
from avalon_rl.training.trainer import train


def parse_set(items: list[str]) -> dict:
    overrides = {}
    for item in items:
        key, _, raw = item.partition("=")
        if not key or not _:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        overrides[key] = yaml.safe_load(raw)
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/baseline_silent.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", help="auto | cpu | cuda | mps")
    parser.add_argument("--total-games", type=int)
    parser.add_argument("--name")
    parser.add_argument("--out-dir")
    parser.add_argument("--init-from", help="checkpoint whose network weights initialise training")
    parser.add_argument("--resume", action="store_true", help="continue from the run's latest.pt")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override any config key")
    parser.add_argument("--threads", type=int, help="torch CPU threads; use 1-2 when running several seeds in parallel")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    overrides = {
        "seed": args.seed, "device": args.device, "total_games": args.total_games, "name": args.name,
        "out_dir": args.out_dir, "init_from": args.init_from, "resume": args.resume or None,
        **parse_set(args.set),
    }
    cfg = load_config(args.config, overrides)
    print(f"run dir: {cfg.run_dir}")
    train(cfg)


if __name__ == "__main__":
    main()
