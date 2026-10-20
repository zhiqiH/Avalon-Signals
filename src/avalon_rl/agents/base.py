"""Controller interface: a controller chooses actions for a batch of decision requests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..env.game import AvalonGame


@dataclass
class Request:
    game_idx: int
    game: AvalonGame
    player: int
    mask: np.ndarray  # legal actions, shape (N_ACTIONS,)


@dataclass
class ActResult:
    actions: np.ndarray
    # Filled in by learning policies so the rollout can store training samples.
    obs: np.ndarray | None = None
    masks: np.ndarray | None = None
    logps: np.ndarray | None = None
    values: np.ndarray | None = None


class Controller(Protocol):
    learn: bool  # whether decisions made by this controller become PPO training samples

    def act(self, requests: list[Request], rng: np.random.Generator) -> ActResult:
        ...
