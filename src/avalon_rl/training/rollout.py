"""Turns recorded decisions into per-team PPO batches with GAE advantages."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, fields
from typing import Sequence

import numpy as np

from ..agents.base import ActResult, Request
from ..env.actions import N_ACTIONS
from ..env.game import AvalonGame
from ..env.observation import OBS_DIM

TEAMS = ("good", "evil")


@dataclass
class Batch:
    obs: np.ndarray
    masks: np.ndarray
    actions: np.ndarray
    logps: np.ndarray
    values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray

    def __len__(self) -> int:
        return len(self.actions)

    @classmethod
    def empty(cls) -> "Batch":
        return cls(
            obs=np.zeros((0, OBS_DIM), np.float32), masks=np.zeros((0, N_ACTIONS), bool),
            actions=np.zeros(0, np.int64), logps=np.zeros(0, np.float32), values=np.zeros(0, np.float32),
            advantages=np.zeros(0, np.float32), returns=np.zeros(0, np.float32),
        )

    @classmethod
    def concat(cls, parts: Sequence["Batch"]) -> "Batch":
        if not parts:
            return cls.empty()
        return cls(**{f.name: np.concatenate([getattr(p, f.name) for p in parts]) for f in fields(cls)})


def gae_terminal(values: np.ndarray, final_reward: float, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """GAE for one player's trajectory whose only reward arrives at its last decision.

    The game ends after the last decision, so the value after it is 0. With gamma = lam = 1 the
    advantage reduces to the Monte Carlo return minus the value estimate.
    """
    n = len(values)
    rewards = np.zeros(n, dtype=np.float64)
    rewards[-1] = final_reward
    next_values = np.append(values[1:], 0.0)
    deltas = rewards + gamma * next_values - values
    advantages = np.zeros(n, dtype=np.float64)
    running = 0.0
    for t in range(n - 1, -1, -1):
        running = deltas[t] + gamma * lam * running
        advantages[t] = running
    return advantages, advantages + values


class TrajectoryRecorder:
    """Decision hook for `runner.play` that stores each learning player's trajectory."""

    def __init__(self) -> None:
        self.steps: dict[tuple[int, int], list[tuple]] = defaultdict(list)

    def __call__(self, req: Request, result: ActResult, row: int) -> None:
        self.steps[(req.game_idx, req.player)].append(
            (result.obs[row], result.masks[row], result.actions[row], result.logps[row], result.values[row])
        )

    def build(self, games: Sequence[AvalonGame], gamma: float, lam: float) -> dict[str, Batch]:
        parts: dict[str, list[Batch]] = {team: [] for team in TEAMS}
        for (game_idx, player), steps in self.steps.items():
            game = games[game_idx]
            evil = game.is_evil_seat(player)
            reward = 1.0 if game.good_won != evil else -1.0
            obs, masks, actions, logps, values = (np.array(x) for x in zip(*steps))
            advantages, returns = gae_terminal(values.astype(np.float64), reward, gamma, lam)
            parts["evil" if evil else "good"].append(Batch(
                obs=obs.astype(np.float32), masks=masks.astype(bool), actions=actions.astype(np.int64),
                logps=logps.astype(np.float32), values=values.astype(np.float32),
                advantages=advantages.astype(np.float32), returns=returns.astype(np.float32),
            ))
        return {team: Batch.concat(parts[team]) for team in TEAMS}
