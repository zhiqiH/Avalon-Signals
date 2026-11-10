"""Actor-critic network and the controller that plays with it.

One network per team; all seats of that team share its parameters. The actor and critic are
separate MLPs over the information-state vector. Illegal actions are masked before the softmax.

The actor has one categorical head for game actions and, in networks of communication configurations
(`comm=True`), the 22 message heads of `comm.protocol`; a decision's log-probability is the sum over the
heads it used. A message is sampled in two stages: the claim and judgments first, then the rationale, team
suggestion and action intents, whose logits are conditioned on the stage-A choices (`actor_cond`, zero at the
branch point) on top of the consistency masks. Communication networks also read the message history: each
message slot (quest x speaker, plus the evil talk) goes through a shared linear encoder, and the per-slot
codes are concatenated with the silent observation and the other communication features.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

from ..comm.protocol import (
    CLAIM_VALUES,
    COMM_GLOBAL_DIM,
    COND_DIM,
    H_CLAIM,
    H_JUDGE,
    JUDGE_VALUES,
    MSG_FEAT_DIM,
    MSG_HEAD_OFFSETS,
    MSG_HEAD_SIZES,
    N_MSG_HEADS,
    N_MSG_SLOTS,
    N_OTHERS,
    STAGE_A,
    STAGE_B,
    encode_comm,
    head_mask,
    stack_contexts,
)
from ..env.actions import N_ACTIONS
from ..env.observation import OBS_DIM, encode
from .base import ActResult, Request

MASKED_LOGIT = -1e9
MSG_EMBED = 32  # code size per message slot
# Logit columns of the stage-B message heads (conditioned on the claim and judgments).
_STAGE_B_COLS = torch.zeros(N_ACTIONS + sum(MSG_HEAD_SIZES), dtype=torch.bool)
for _h in STAGE_B:
    _STAGE_B_COLS[N_ACTIONS + MSG_HEAD_OFFSETS[_h]:N_ACTIONS + MSG_HEAD_OFFSETS[_h] + MSG_HEAD_SIZES[_h]] = True


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
            self.actor_cond = nn.Linear(COND_DIM, hidden_sizes[-1], bias=False)
            nn.init.zeros_(self.actor_cond.weight)

    def _inputs(self, obs: torch.Tensor, comm_obs: torch.Tensor | None, encoder: nn.Module | None) -> torch.Tensor:
        if not self.comm:
            return obs
        comm_obs = comm_obs.float()
        msgs = comm_obs[:, COMM_GLOBAL_DIM:].reshape(len(obs), N_MSG_SLOTS, MSG_FEAT_DIM)
        return torch.cat([obs, comm_obs[:, :COMM_GLOBAL_DIM], torch.tanh(encoder(msgs)).flatten(1)], dim=-1)

    def hidden(self, obs: torch.Tensor, comm_obs: torch.Tensor | None = None) -> torch.Tensor:
        return self.actor[:-1](self._inputs(obs, comm_obs, getattr(self, "actor_msg", None)))

    def head_logits(self, hidden: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Logits of every head; with `cond` (stage-A one-hots) the stage-B message heads are conditioned on it."""
        out = self.actor[-1](hidden)
        if cond is None or not self.comm:
            return out
        conditioned = self.actor[-1](hidden + self.actor_cond(cond))
        return torch.where(_STAGE_B_COLS.to(out.device), conditioned, out)

    def logits(self, obs: torch.Tensor, comm_obs: torch.Tensor | None = None, cond: torch.Tensor | None = None):
        return self.head_logits(self.hidden(obs, comm_obs), cond)

    def value(self, obs: torch.Tensor, comm_obs: torch.Tensor | None = None) -> torch.Tensor:
        return self.critic(self._inputs(obs, comm_obs, getattr(self, "critic_msg", None))).squeeze(-1)

    def evaluate(self, obs, comm_obs, masks, heads):
        """(log-prob, entropy, value, logits) of recorded decisions."""
        logits = self.logits(obs, comm_obs, stage_a_onehot(heads[:, 1:]) if self.comm else None)
        logp, entropy = heads_logp_entropy(logits, masks, heads, self.head_sizes)
        return logp, entropy, self.value(obs, comm_obs), logits


def stage_a_onehot(msg: torch.Tensor) -> torch.Tensor:
    """(batch, COND_DIM) one-hots of the claim and the five judgments of message heads `msg` (-1 gives zeros)."""
    parts = [msg[:, H_CLAIM:H_CLAIM + 1], msg[:, H_JUDGE:H_JUDGE + N_OTHERS]]
    sizes = [len(CLAIM_VALUES), len(JUDGE_VALUES)]
    out = []
    for values, size in zip(parts, sizes):
        values = values.long()
        onehot = torch.nn.functional.one_hot(values.clamp(min=0), size).float() * (values >= 0).unsqueeze(-1)
        out.append(onehot.flatten(1))
    return torch.cat(out, dim=1)


def load_silent_weights(net: ActorCritic, silent_state: dict[str, torch.Tensor]) -> None:
    """Initialise a communication network from a silent one so that it plays exactly like it.

    The silent weights fill the leading block of every layer. Weights from the new communication inputs
    and into the new message heads are zero, so messages are ignored and every message head starts uniform;
    the message encoders keep their random initialisation (their outputs meet zero weights), and the stage-B
    conditioning starts at zero.
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
        """Sample messages head by head under the consistency masks: claim and judgments, then the rest conditioned on them."""
        if not self.net.comm:
            raise ValueError("a silent network cannot send messages")
        obs, comm, t_obs, t_comm = self._inputs(requests)
        n = len(requests)
        ctx = stack_contexts([r.ctx for r in requests])
        hidden = self.net.hidden(t_obs, t_comm)
        logits = self.net.head_logits(hidden)
        masks = torch.zeros(logits.shape, dtype=torch.bool, device=self.device)
        msg = np.full((n, N_MSG_HEADS), -1, dtype=np.int64)
        gen = _generator(rng, self.device)

        def choose(h: int, logits: torch.Tensor) -> None:
            legal = torch.from_numpy(head_mask(h, msg, ctx)).to(self.device)
            lo = N_ACTIONS + MSG_HEAD_OFFSETS[h]
            masks[:, lo:lo + MSG_HEAD_SIZES[h]] = legal
            head_logits = masked_logits(logits[:, lo:lo + MSG_HEAD_SIZES[h]], legal)
            pick = head_logits.argmax(-1) if self.greedy else _sample(head_logits, gen)
            msg[:, h] = np.where(legal.any(-1).cpu().numpy(), pick.cpu().numpy(), -1)

        for h in STAGE_A:
            choose(h, logits)
        t_msg = torch.from_numpy(msg).to(self.device)
        logits = self.net.head_logits(hidden, stage_a_onehot(t_msg))  # stage-A columns are unchanged
        for h in STAGE_B:
            choose(h, logits)

        heads = torch.cat([torch.full((n, 1), -1, dtype=torch.long, device=self.device),
                           torch.from_numpy(msg).to(self.device)], dim=1)
        return self._result(logits, masks, heads, obs, comm, t_obs, t_comm, messages=msg.astype(np.int8))
