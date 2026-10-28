"""Actor-critic network and the controller that plays with it.

One network per team; all seats of that team share its parameters. The actor and critic are
separate MLPs over the information-state vector. Illegal actions are masked before the softmax.

The actor has one categorical head for game actions and, in networks of communication configurations
(`comm=True`), the 19 message heads of `comm.protocol`; a decision's log-probability is the sum over the
heads it used. Communication networks also read the message history: each message slot (quest x attempt
x speaker) goes through a shared linear encoder, and the per-slot codes are concatenated with the silent
observation and the other communication features.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

from ..comm.protocol import (
    COMM_GLOBAL_DIM,
    H_ALIGN,
    H_RROUND,
    H_RTYPE,
    MSG_FEAT_DIM,
    MSG_HEAD_OFFSETS,
    MSG_HEAD_SIZES,
    N_MSG_HEADS,
    N_MSG_SLOTS,
    N_OTHERS,
    RECORD_TYPES,
    STATIC_HEADS,
    encode_comm,
)
from ..env.actions import N_ACTIONS
from ..env.observation import OBS_DIM, encode
from .base import ActResult, Request

MASKED_LOGIT = -1e9
MSG_EMBED = 32  # code size per message slot


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, MASKED_LOGIT)


def _generator(rng: np.random.Generator, device: torch.device) -> torch.Generator:
    """Torch sampling stream drawn from the caller's numpy generator, so play is reproducible from its seed."""
    return torch.Generator(device=device).manual_seed(int(rng.integers(2**63 - 1)))


def _sample(logits: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    return torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=gen).squeeze(1)


def heads_logp_entropy(logits: torch.Tensor, masks: torch.Tensor, heads: torch.Tensor,
                       head_sizes: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-probability and entropy of each decision, summed over the heads it used (heads[:, h] >= 0)."""
    logp = logits.new_zeros(len(logits))
    entropy = logits.new_zeros(len(logits))
    zero = logits.new_zeros(len(logits))
    off = 0
    for h, size in enumerate(head_sizes):
        lp = torch.log_softmax(masked_logits(logits[:, off:off + size], masks[:, off:off + size]), dim=-1)
        idx = heads[:, h].long()
        used = idx >= 0
        logp = logp + torch.where(used, lp.gather(1, idx.clamp(min=0).unsqueeze(1)).squeeze(1), zero)
        entropy = entropy + torch.where(used, -(lp.exp() * lp).sum(-1), zero)
        off += size
    return logp, entropy


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
    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden_sizes: Sequence[int] = (256, 256),
                 comm: bool = False):
        super().__init__()
        self.comm = comm
        self.head_sizes: tuple[int, ...] = (n_actions,) + (MSG_HEAD_SIZES if comm else ())
        in_dim = obs_dim + (COMM_GLOBAL_DIM + N_MSG_SLOTS * MSG_EMBED if comm else 0)
        self.actor = _mlp(in_dim, hidden_sizes, sum(self.head_sizes), out_std=0.01)
        self.critic = _mlp(in_dim, hidden_sizes, 1, out_std=1.0)
        if comm:
            self.actor_msg = _layer(MSG_FEAT_DIM, MSG_EMBED, std=1.0)
            self.critic_msg = _layer(MSG_FEAT_DIM, MSG_EMBED, std=1.0)

    def _inputs(self, obs: torch.Tensor, comm_obs: torch.Tensor | None, encoder: nn.Module | None) -> torch.Tensor:
        if not self.comm:
            return obs
        comm_obs = comm_obs.float()
        msgs = comm_obs[:, COMM_GLOBAL_DIM:].reshape(len(obs), N_MSG_SLOTS, MSG_FEAT_DIM)
        return torch.cat([obs, comm_obs[:, :COMM_GLOBAL_DIM], torch.tanh(encoder(msgs)).flatten(1)], dim=-1)

    def logits(self, obs: torch.Tensor, comm_obs: torch.Tensor | None = None) -> torch.Tensor:
        return self.actor(self._inputs(obs, comm_obs, getattr(self, "actor_msg", None)))

    def value(self, obs: torch.Tensor, comm_obs: torch.Tensor | None = None) -> torch.Tensor:
        return self.critic(self._inputs(obs, comm_obs, getattr(self, "critic_msg", None))).squeeze(-1)

    def evaluate(self, obs, comm_obs, masks, heads):
        """(log-prob, entropy, value, logits) of recorded decisions."""
        logits = self.logits(obs, comm_obs)
        logp, entropy = heads_logp_entropy(logits, masks, heads, self.head_sizes)
        return logp, entropy, self.value(obs, comm_obs), logits


def load_silent_weights(net: ActorCritic, silent_state: dict[str, torch.Tensor]) -> None:
    """Initialise a communication network from a silent one so that it plays exactly like it.

    The silent weights fill the leading block of every layer. Weights from the new communication inputs
    and into the new message heads are zero, so messages are ignored and every message head starts uniform;
    the message encoders keep their random initialisation (their outputs meet zero weights).
    """
    state = net.state_dict()
    for key, value in silent_state.items():
        target = state[key]
        if target.shape != value.shape:
            target.zero_()
        target[tuple(slice(0, n) for n in value.shape)] = value
    net.load_state_dict(state)


class PolicyController:
    def __init__(self, net: ActorCritic, learn: bool, device: str | torch.device = "cpu", greedy: bool = False):
        self.net = net
        self.learn = learn
        self.device = torch.device(device)
        self.greedy = greedy

    def _inputs(self, requests: list[Request]):
        obs = np.stack([encode(r.game, r.player) for r in requests])
        comm = np.stack([encode_comm(r.game, r.player) for r in requests]) if self.net.comm else None
        t_comm = torch.from_numpy(comm).to(self.device) if comm is not None else None
        return obs, comm, torch.from_numpy(obs).to(self.device), t_comm

    def _result(self, logits, masks, heads, obs, comm, t_obs, t_comm, **decisions) -> ActResult:
        logp, _ = heads_logp_entropy(logits, masks, heads, self.net.head_sizes)
        return ActResult(
            obs=obs, comm_obs=comm, masks=masks.cpu().numpy(), heads=heads.cpu().numpy().astype(np.int8),
            logps=logp.cpu().numpy(), values=self.net.value(t_obs, t_comm).cpu().numpy(), **decisions,
        )

    @torch.no_grad()
    def act(self, requests: list[Request], rng: np.random.Generator) -> ActResult:
        obs, comm, t_obs, t_comm = self._inputs(requests)
        logits = self.net.logits(t_obs, t_comm)
        masks = torch.zeros(logits.shape, dtype=torch.bool, device=self.device)
        masks[:, :N_ACTIONS] = torch.from_numpy(np.stack([r.mask for r in requests]))
        action_logits = masked_logits(logits[:, :N_ACTIONS], masks[:, :N_ACTIONS])
        actions = action_logits.argmax(-1) if self.greedy else _sample(action_logits, _generator(rng, self.device))
        heads = torch.full((len(requests), len(self.net.head_sizes)), -1, dtype=torch.long, device=self.device)
        heads[:, 0] = actions
        return self._result(logits, masks, heads, obs, comm, t_obs, t_comm, actions=actions.cpu().numpy())

    @torch.no_grad()
    def speak(self, requests: list[Request], rng: np.random.Generator) -> ActResult:
        """Sample a message head by head: static heads, then rationale types of judged players, then rounds."""
        if not self.net.comm:
            raise ValueError("a silent network cannot send messages")
        obs, comm, t_obs, t_comm = self._inputs(requests)
        n = len(requests)
        logits = self.net.logits(t_obs, t_comm)
        masks = torch.zeros(logits.shape, dtype=torch.bool, device=self.device)
        static = torch.from_numpy(np.stack([r.mask for r in requests])).to(self.device)
        rtype_valid = torch.from_numpy(np.stack([r.rtype_valid for r in requests])).to(self.device)
        rround_valid = torch.from_numpy(np.stack([r.rround_valid for r in requests])).to(self.device)
        msg = torch.full((n, N_MSG_HEADS), -1, dtype=torch.long, device=self.device)
        gen = _generator(rng, self.device)

        def choose(h: int, legal: torch.Tensor) -> None:
            lo = N_ACTIONS + MSG_HEAD_OFFSETS[h]
            masks[:, lo:lo + MSG_HEAD_SIZES[h]] = legal
            head_logits = masked_logits(logits[:, lo:lo + MSG_HEAD_SIZES[h]], legal)
            pick = head_logits.argmax(-1) if self.greedy else _sample(head_logits, gen)
            msg[:, h] = torch.where(legal.any(-1), pick, torch.full_like(pick, -1))

        for h in STATIC_HEADS:
            choose(h, static[:, MSG_HEAD_OFFSETS[h]:MSG_HEAD_OFFSETS[h] + MSG_HEAD_SIZES[h]])
        for r in range(N_OTHERS):
            choose(H_RTYPE + r, rtype_valid[:, r] & (msg[:, H_ALIGN + r] > 0).unsqueeze(1))
        rows = torch.arange(n, device=self.device)
        for r in range(N_OTHERS):
            t = msg[:, H_RTYPE + r]
            cites = (t >= min(RECORD_TYPES)) & (t <= max(RECORD_TYPES))
            choose(H_RROUND + r, rround_valid[rows, r, t.clamp(min=0)] & cites.unsqueeze(1))

        heads = torch.cat([torch.full((n, 1), -1, dtype=torch.long, device=self.device), msg], dim=1)
        return self._result(logits, masks, heads, obs, comm, t_obs, t_comm,
                            messages=msg.cpu().numpy().astype(np.int8))
