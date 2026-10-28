"""Measure simulation and update throughput on the current machine before choosing budget B.

    python scripts/benchmark.py --games 2048 --device cuda
    python scripts/benchmark.py --games 512 --comm F --device cuda   # full five-slot communication

Reports (1) random-vs-random games per second (pure engine), (2) self-play rollout speed with the
policy network, and (3) PPO update time for one iteration's batch on the chosen device. --comm picks a
communication configuration: S (silent, default), M (alignment + rationale) or F (all five slots).
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch

from avalon_rl.agents import ActorCritic, PolicyController, RandomController
from avalon_rl.comm import CommConfig
from avalon_rl.env.game import AvalonGame
from avalon_rl.runner import play
from avalon_rl.training import PPOConfig, TrajectoryRecorder, ppo_update


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=2048)
    parser.add_argument("--device", default="cpu", help="device for the PPO update")
    parser.add_argument("--comm", default="S", choices=["S", "M", "F"])
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    comm = {"S": CommConfig(), "M": CommConfig(enabled=True, claim=False, team=False, intent=False),
            "F": CommConfig(enabled=True)}[args.comm]
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    games = [AvalonGame(rng, comm=comm) for _ in range(args.games)]
    rand = RandomController()
    t = time.perf_counter()
    play(games, [rand] * len(games), [rand] * len(games), rng)
    dt = time.perf_counter() - t
    print(f"engine (random play):  {args.games / dt:8.0f} games/s")

    nets = {"good": ActorCritic(comm=comm.enabled), "evil": ActorCritic(comm=comm.enabled)}
    learners = {k: PolicyController(v, learn=True) for k, v in nets.items()}
    games = [AvalonGame(rng, comm=comm) for _ in range(args.games)]
    recorder = TrajectoryRecorder()
    t = time.perf_counter()
    play(games, [learners["good"]] * len(games), [learners["evil"]] * len(games), rng, on_decision=recorder)
    batches = recorder.build(games, gamma=1.0, lam=0.95)
    dt = time.perf_counter() - t
    n_dec = sum(len(b) for b in batches.values())
    print(f"self-play rollout:     {args.games / dt:8.0f} games/s  ({n_dec / dt:.0f} decisions/s, "
          f"{n_dec / args.games:.1f} decisions/game)")

    device = torch.device(args.device)
    cfg = PPOConfig()
    for team, batch in batches.items():
        net = copy.deepcopy(nets[team]).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
        t = time.perf_counter()
        ppo_update(net, opt, batch, cfg, device, rng)
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"PPO update ({team:4s}, {device.type}): {time.perf_counter() - t:6.2f} s for {len(batch)} samples")


if __name__ == "__main__":
    main()
