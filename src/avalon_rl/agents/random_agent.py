"""Uniformly random legal play: the weakest sanity baseline."""
from __future__ import annotations

import numpy as np

from ..comm.protocol import sample_messages, stack_contexts
from .base import ActResult, Request


class RandomController:
    learn = False

    def act(self, requests: list[Request], rng: np.random.Generator) -> ActResult:
        return ActResult(np.array([rng.choice(np.flatnonzero(r.mask)) for r in requests], dtype=np.int64))

    def speak(self, requests: list[Request], rng: np.random.Generator) -> ActResult:
        """Each message head uniformly among its legal values."""
        return ActResult(messages=sample_messages(stack_contexts([r.ctx for r in requests]), rng))
