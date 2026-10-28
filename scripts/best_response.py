"""Train an approximate best response against one side of a checkpoint (half of an exploitability estimate).

    python scripts/best_response.py --checkpoint runs/baseline_eq/seed0/latest.pt --side evil --games 100000
    python scripts/best_response.py --checkpoint runs/baseline_eq/seed0/latest.pt --side good --games 100000

`--side evil` trains an evil team against the frozen good policy and vice versa. Each run writes
br_<side>_<checkpoint stem>_<games>.json next to the checkpoint; combine the two sides with
scripts/exploitability.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from avalon_rl.training.best_response import BestResponseConfig, train_best_response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--side", required=True, choices=["good", "evil"])
    parser.add_argument("--games", type=int, default=100_000)
    parser.add_argument("--init", default="target", choices=["target", "scratch"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-games", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    cfg = BestResponseConfig(games=args.games, init=args.init, seed=args.seed, eval_games=args.eval_games, device=args.device)
    ckpt = Path(args.checkpoint)
    out = Path(args.out) if args.out else ckpt.with_name(f"br_{args.side}_{ckpt.stem}_{args.games // 1000}k.json")
    result = train_best_response(ckpt, args.side, cfg, out_path=out)
    print(f"BR {args.side}: {result['start_win']:.3f} -> {result['final_win']:.3f}; wrote {out}")


if __name__ == "__main__":
    main()
