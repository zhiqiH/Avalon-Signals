"""Uniformly random legal play: the weakest sanity baseline."""
from __future__ import annotations

import numpy as np

from .base import ActResult, Request


class RandomController:
    learn = False

    def act(self, requests: list[Request], rng: np.random.Generator) -> ActResult:
        return ActResult(np.array([rng.choice(np.flatnonzero(r.mask)) for r in requests], dtype=np.int64))
