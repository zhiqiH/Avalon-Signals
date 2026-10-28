"""Cross-seed play: every seed's good policy against every seed's evil policy.

    python scripts/crossplay.py runs/baseline_silent/seed0/latest.pt runs/baseline_silent/seed1/latest.pt runs/baseline_silent/seed2/latest.pt

Cell (i, j) is the good win rate of good_i vs evil_j over the same paired role assignments. The diagonal is
ordinary self-play within one training run; the off-diagonal pairs policies that never trained together.
A large gap between the two means measures co-adaptation; for policies near an equilibrium it should be small.
Writes the matrix and per-cell summaries to --out (default crossplay.json next to the first checkpoint).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from avalon_rl.agents import PolicyController
from avalon_rl.evaluation import run_matchup
from avalon_rl.training.checkpoint import load_policies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261020, help="role-assignment seed shared by all cells")
    parser.add_argument("--out")
    args = parser.parse_args()

    policies = [{t: PolicyController(net, learn=False) for t, net in load_policies(c).items()} for c in args.checkpoints]
    n = len(policies)
    win = np.zeros((n, n))
    cells = {}
    for i in range(n):
        for j in range(n):
            summary, _ = run_matchup(policies[i]["good"], policies[j]["evil"], args.games, args.seed)
            win[i, j] = summary["good_win"]["rate"]
            cells[f"good{i}_evil{j}"] = summary

    print("good win rate, rows = good policy, columns = evil policy")
    print("        " + "".join(f"evil{j:<4d}" for j in range(n)))
    for i in range(n):
        print(f"good{i:<4d}" + "".join(f"{win[i, j]:<8.3f}" for j in range(n)))
    off = ~np.eye(n, dtype=bool)
    diag_mean, off_mean = float(np.diag(win).mean()), float(win[off].mean()) if n > 1 else float("nan")
    print(f"same-seed mean {diag_mean:.3f}  cross-seed mean {off_mean:.3f}  all {win.mean():.3f}")

    out = Path(args.out) if args.out else Path(args.checkpoints[0]).with_name("crossplay.json")
    out.write_text(json.dumps({
        "checkpoints": args.checkpoints, "games": args.games, "seed": args.seed,
        "good_win": win.tolist(), "same_seed_mean": diag_mean, "cross_seed_mean": off_mean, "cells": cells,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
