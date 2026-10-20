"""Actor-critic network and the controller that plays with it.

One network per team; all seats of that team share its parameters. The actor and critic are
separate MLPs over the information-state vector. Illegal actions are masked before the softmax.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ..env.actions import N_ACTIONS
from ..env.observation import OBS_DIM, encode
from .base import ActResult, Request

MASKED_LOGIT = -1e9


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, MASKED_LOGIT)


def _layer(in_dim: int, out_dim: int, std: float = float(np.sqrt(2))) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, std)
    nn.init.zeros_(layer.bias)
    return layer


def _mlp(in_dim: int, hidden_sizes: Sequence[int], out_dim: int, out_std: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for h in hidden_sizes:
        layers += [_layer(in_dim, h), nn.Tanh()]
        in_dim = h
    layers.append(_layer(in_dim, out_dim, std=out_std))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden_sizes: Sequence[int] = (256, 256)):
        super().__init__()
        self.actor = _mlp(obs_dim, hidden_sizes, n_actions, out_std=0.01)
        self.critic = _mlp(obs_dim, hidden_sizes, 1, out_std=1.0)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def distribution(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        return Categorical(logits=masked_logits(self.actor(obs), mask)), self.value(obs)


class PolicyController:
    def __init__(self, net: ActorCritic, learn: bool, device: str | torch.device = "cpu", greedy: bool = False):
        self.net = net
        self.learn = learn
        self.device = torch.device(device)
        self.greedy = greedy

    @torch.no_grad()
    def act(self, requests: list[Request], rng: np.random.Generator) -> ActResult:
        obs = np.stack([encode(r.game, r.player) for r in requests])
        masks = np.stack([r.mask for r in requests])
        dist, value = self.net.distribution(
            torch.from_numpy(obs).to(self.device), torch.from_numpy(masks).to(self.device)
        )
        actions = dist.probs.argmax(-1) if self.greedy else dist.sample()
        return ActResult(
            actions=actions.cpu().numpy(),
            obs=obs,
            masks=masks,
            logps=dist.log_prob(actions).cpu().numpy(),
            values=value.cpu().numpy(),
        )
