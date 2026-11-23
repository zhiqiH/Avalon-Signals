"""LoRA adaptation of a branched network: the frozen silent baseline changes only through low-rank updates.

With `adapt.lora_rank > 0` the actor's three linear layers are re-parametrized as

    W = W0 + F * free + scale * (B @ A) * (1 - F)

- W0: the starting weights, frozen.
- F: marks the entries that are new in a communication network (the message-input columns of the first
  layer and the message-head rows of the output layer); those entries train freely.
- B @ A: the rank-r update of everything that came from the baseline (B starts at zero, so the network
  starts exactly where it was).

Biases are frozen, except those of new output rows. The message encoders, the stage-B conditioning map and
the whole critic train normally: the critic only serves training and is not part of the evaluated policy.
A silent network has no new entries, so the same code gives the silent LoRA control.

Rollouts, the opponent pool, rollout workers and checkpoints use plain merged weights (`plain_state_dict`),
so every evaluation tool reads an adapted network like any other. Checkpoints also keep the adapter state
(`adapter_state`), so training can resume, or continue from a stage-2 run, with the same adapters.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import parametrize

from ..agents.policy import ActorCritic
from ..env.actions import N_ACTIONS


class LoRAWeight(nn.Module):
    def __init__(self, shape: torch.Size, free_mask: torch.Tensor, rank: int, scale: float):
        super().__init__()
        out_dim, in_dim = shape
        dev = free_mask.device
        self.register_buffer("free_mask", free_mask.float())
        self.free = nn.Parameter(torch.zeros(shape, device=dev))
        self.lora_a = nn.Parameter(torch.randn(rank, in_dim, device=dev) / in_dim ** 0.5)
        self.lora_b = nn.Parameter(torch.zeros(out_dim, rank, device=dev))
        self.scale = scale

    def forward(self, w0: torch.Tensor) -> torch.Tensor:
        low_rank = self.scale * (self.lora_b @ self.lora_a)
        return w0 + self.free * self.free_mask + low_rank * (1 - self.free_mask)


class FreeBias(nn.Module):
    def __init__(self, free_mask: torch.Tensor):
        super().__init__()
        self.register_buffer("free_mask", free_mask.float())
        self.free = nn.Parameter(torch.zeros(free_mask.shape, device=free_mask.device))

    def forward(self, b0: torch.Tensor) -> torch.Tensor:
        return b0 + self.free * self.free_mask


def is_adapted(net: nn.Module) -> bool:
    return any(parametrize.is_parametrized(m) for m in net.modules())


def apply_lora(net: ActorCritic, rank: int, scale: float = 1.0) -> ActorCritic:
    """Re-parametrize `net` in place (see the module docstring) and freeze what must stay fixed."""
    layers = [m for m in net.actor if isinstance(m, nn.Linear)]
    for i, layer in enumerate(layers):
        mask = torch.zeros_like(layer.weight, dtype=torch.bool)
        bias_mask = torch.zeros_like(layer.bias, dtype=torch.bool)
        if net.comm and i == 0:
            mask[:, net.obs_dim:] = True  # message inputs
        if net.comm and i == len(layers) - 1:
            mask[N_ACTIONS:] = True  # message heads
            bias_mask[N_ACTIONS:] = True
        parametrize.register_parametrization(layer, "weight", LoRAWeight(layer.weight.shape, mask, rank, scale))
        parametrize.register_parametrization(layer, "bias", FreeBias(bias_mask))
        layer.parametrizations.weight.original.requires_grad_(False)
        layer.parametrizations.bias.original.requires_grad_(False)
    return net


def plain_state_dict(net: nn.Module) -> dict[str, torch.Tensor]:
    """Merged weights under the plain ActorCritic keys (the state dict itself for an unadapted network)."""
    if not is_adapted(net):
        return net.state_dict()
    out = {}
    with torch.no_grad():
        for prefix, m in net.named_modules():
            names = m.parametrizations.keys() if parametrize.is_parametrized(m) else ()
            for name in names:  # the parametrized tensor, computed from the frozen original and the adapters
                out[f"{prefix}.{name}"] = getattr(m, name).detach().clone()
        for k, v in net.state_dict().items():
            if ".parametrizations." not in k:
                out[k] = v
    return out


def adapter_state(net: nn.Module) -> dict[str, torch.Tensor] | None:
    return net.state_dict() if is_adapted(net) else None


def trainable_parameters(net: nn.Module) -> list[nn.Parameter]:
    return [p for p in net.parameters() if p.requires_grad]
