"""Frozen past versions of each team's policy, used as self-play opponents."""
from __future__ import annotations

import copy
from collections import deque

from ..agents.policy import ActorCritic, PolicyController


class OpponentPool:
    def __init__(self, capacity: int, speakers: dict[str, PolicyController] | None = None):
        self.speakers = speakers or {}  # listener stage: snapshots speak through the frozen speaker too
        self.snapshots: dict[str, deque[PolicyController]] = {
            "good": deque(maxlen=capacity), "evil": deque(maxlen=capacity),
        }

    def add(self, team: str, net: ActorCritic) -> None:
        frozen = copy.deepcopy(net).to("cpu").eval()
        for p in frozen.parameters():
            p.requires_grad_(False)
        self.snapshots[team].append(PolicyController(frozen, learn=False, speaker=self.speakers.get(team)))

    def __len__(self) -> int:
        return sum(len(v) for v in self.snapshots.values())
