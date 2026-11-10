"""Saving and loading training checkpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..agents.policy import ActorCritic
from ..comm.protocol import CommConfig

TEAMS = ("good", "evil")


def save_checkpoint(path: Path, **state: Any) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic, so an interrupted save never corrupts latest.pt


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    # Our own files: they also hold config and RNG state, so full unpickling is needed.
    return torch.load(path, map_location=device, weights_only=False)


def comm_config(config: dict[str, Any]) -> CommConfig:
    """Communication configuration stored in a checkpoint's config (silent for checkpoints that predate it)."""
    return CommConfig(**config.get("comm", {}))


def build_net(config: dict[str, Any]) -> ActorCritic:
    return ActorCritic(hidden_sizes=config["model"]["hidden_sizes"], comm=comm_config(config).enabled)


def load_policies(path: str | Path, device: str | torch.device = "cpu") -> dict[str, ActorCritic]:
    """Rebuild both teams' networks from a checkpoint, with the architecture it was trained with."""
    ckpt = load_checkpoint(path, device)
    nets = {}
    for team in TEAMS:
        net = build_net(ckpt["config"]).to(device)
        net.load_state_dict(ckpt["nets"][team])
        nets[team] = net.eval()
    return nets
