"""Turns recorded decisions into per-team PPO batches with GAE advantages."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, fields
from typing import Sequence

import numpy as np

from ..agents.base import ActResult, Request
from ..comm.protocol import H_ALIGN, H_CLAIM, H_INTENT, H_NAME, H_RTYPE, H_TEAM, MSG_HEAD_OFFSETS, MSG_HEAD_SIZES, N_OTHERS
from ..env.actions import N_ACTIONS
from ..env.game import AvalonGame
from ..env.observation import OBS_DIM

TEAMS = ("good", "evil")


@dataclass
class Batch:
    obs: np.ndarray
    comm_obs: np.ndarray  # int8 communication features; zero columns for silent networks
    masks: np.ndarray  # legal values of every network head, concatenated
    actions: np.ndarray  # chosen value per network head, -1 where unused (one column for silent networks)
    logps: np.ndarray
    values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray

    def __len__(self) -> int:
        return len(self.actions)

    @classmethod
    def empty(cls) -> "Batch":
        return cls(
            obs=np.zeros((0, OBS_DIM), np.float32), comm_obs=np.zeros((0, 0), np.int8),
            masks=np.zeros((0, N_ACTIONS), bool), actions=np.zeros((0, 1), np.int8),
            logps=np.zeros(0, np.float32), values=np.zeros(0, np.float32),
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
        comm = result.comm_obs[row] if result.comm_obs is not None else np.zeros(0, np.int8)
        self.steps[(req.game_idx, req.player)].append(
            (result.obs[row], comm, result.masks[row], result.heads[row], result.logps[row], result.values[row])
        )

    def build(self, games: Sequence[AvalonGame], gamma: float, lam: float) -> dict[str, Batch]:
        parts: dict[str, list[Batch]] = {team: [] for team in TEAMS}
        for (game_idx, player), steps in self.steps.items():
            game = games[game_idx]
            evil = game.is_evil_seat(player)
            reward = 1.0 if game.good_won != evil else -1.0
            obs, comm, masks, actions, logps, values = (np.array(x) for x in zip(*steps))
            advantages, returns = gae_terminal(values.astype(np.float64), reward, gamma, lam)
            parts["evil" if evil else "good"].append(Batch(
                obs=obs.astype(np.float32), comm_obs=comm.astype(np.int8), masks=masks.astype(bool),
                actions=actions.astype(np.int8),
                logps=logps.astype(np.float32), values=values.astype(np.float32),
                advantages=advantages.astype(np.float32), returns=returns.astype(np.float32),
            ))
        return {team: Batch.concat(parts[team]) for team in TEAMS}


USAGE_HEADS = {
    "claim": [H_CLAIM], "alignment": list(range(H_ALIGN, H_ALIGN + N_OTHERS)),
    "rationale": list(range(H_RTYPE, H_RTYPE + N_OTHERS)), "team": [H_TEAM], "intent": [H_INTENT], "name": [H_NAME],
}


def message_usage(batch: Batch) -> dict[str, float]:
    """Share of message heads that said something, among heads that had a real choice (two or more legal values).

    For rationale, a real choice exists only for judged players; for the team head, the leader (whose suggestion
    is fixed to its proposal) is excluded automatically.
    """
    out = {}
    if batch.actions.shape[1] == 1:
        return out
    msgs = batch.actions[:, 0] < 0
    for slot, heads in USAGE_HEADS.items():
        said = choices = 0
        for h in heads:
            lo = N_ACTIONS + MSG_HEAD_OFFSETS[h]
            free = msgs & (batch.masks[:, lo:lo + MSG_HEAD_SIZES[h]].sum(axis=1) >= 2)
            said += int((batch.actions[free, 1 + h] > 0).sum())
            choices += int(free.sum())
        if choices:
            out[slot] = round(said / choices, 4)
    return out
