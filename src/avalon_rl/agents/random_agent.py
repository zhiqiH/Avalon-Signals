"""Uniformly random legal play: the weakest sanity baseline."""
from __future__ import annotations

import numpy as np

from ..comm.protocol import sample_message
from .base import ActResult, Request


class RandomController:
    learn = False

    def act(self, requests: list[Request], rng: np.random.Generator) -> ActResult:
        return ActResult(np.array([rng.choice(np.flatnonzero(r.mask)) for r in requests], dtype=np.int64))

    def speak(self, requests: list[Request], rng: np.random.Generator) -> ActResult:
        """Each message head uniformly among its legal values."""
        return ActResult(messages=np.stack([sample_message(r.mask, r.rtype_valid, r.rround_valid, rng) for r in requests]))
