"""Controller interface: a controller chooses actions (and, in discussions, messages) for a batch of requests."""
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
    mask: np.ndarray  # act: legal actions (N_ACTIONS,); speak: legal static message values (N_MSG_LOGITS,)
    # speak only: rationale validity tables from comm.protocol.message_mask
    rtype_valid: np.ndarray | None = None
    rround_valid: np.ndarray | None = None


@dataclass
class ActResult:
    actions: np.ndarray | None = None  # act: one action per request
    messages: np.ndarray | None = None  # speak: (requests, N_MSG_HEADS) message heads
    # Filled in by learning policies so the rollout can store training samples.
    obs: np.ndarray | None = None
    comm_obs: np.ndarray | None = None  # communication features (int8), communication networks only
    masks: np.ndarray | None = None  # legal values of every head of the network, concatenated
    heads: np.ndarray | None = None  # chosen value per network head, -1 where a head was not used
    logps: np.ndarray | None = None
    values: np.ndarray | None = None


class Controller(Protocol):
    learn: bool  # whether decisions made by this controller become PPO training samples

    def act(self, requests: list[Request], rng: np.random.Generator) -> ActResult:
        ...

    def speak(self, requests: list[Request], rng: np.random.Generator) -> ActResult:
        ...
