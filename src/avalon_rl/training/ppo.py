"""Clipped PPO update for one team's network, optionally with magnetic mirror descent (MMD).

MMD (Sokota et al., ICLR 2023) adds alpha * KL(pi || magnet) to the loss. With a magnet that is a frozen copy
of the policy refreshed every K iterations, the regularised fixed points move towards a Nash equilibrium of the
two-team zero-sum game; with a uniform magnet the update converges to an entropy-regularised (quantal response)
equilibrium instead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..agents.policy import ActorCritic, masked_logits
from .rollout import Batch


@dataclass
class MMDConfig:
    enabled: bool = False
    alpha: float = 0.1  # weight of KL(pi || magnet)
    magnet: str = "moving"  # moving: frozen policy copy refreshed every `magnet_update_every`; uniform: uniform over legal actions
    magnet_update_every: int = 50  # iterations between magnet refreshes (moving magnet only)


@dataclass
class PPOConfig:
    lr: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 2048
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    gamma: float = 1.0
    gae_lambda: float = 0.95
    norm_adv: bool = True


def magnet_kl(policy_logp: torch.Tensor, magnet_logits: torch.Tensor | None, mask: torch.Tensor) -> torch.Tensor:
    """Mean KL(pi || magnet) over legal actions; `policy_logp` are normalised log-probabilities.

    Computed from log-softmax rather than torch's Categorical KL so that a magnet probability that underflows
    to zero on a legal action gives a large finite penalty instead of inf/NaN. `magnet_logits=None` means the
    uniform distribution over legal actions.
    """
    if magnet_logits is None:
        magnet_logits = torch.zeros_like(policy_logp)
    magnet_logp = torch.log_softmax(masked_logits(magnet_logits, mask), dim=-1)
    p = policy_logp.exp()  # exactly 0 on illegal actions, so they contribute nothing
    return (p * (policy_logp - magnet_logp)).sum(-1).mean()


def ppo_update(
    net: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    cfg: PPOConfig,
    device: torch.device,
    rng: np.random.Generator,
    mmd: MMDConfig | None = None,
    magnet: ActorCritic | None = None,
) -> dict[str, float]:
    """Run `cfg.epochs` passes of minibatch PPO over `batch` and return mean diagnostics.

    Returns lie in [-1, 1] (terminal +/-1 reward, gamma = 1), so the value target needs no normalization.
    With `mmd.enabled`, adds mmd.alpha * KL(pi || magnet); `magnet` is the frozen magnet network for the
    moving magnet and is ignored for the uniform one.
    """
    use_mmd = mmd is not None and mmd.enabled
    if use_mmd and mmd.magnet == "moving" and magnet is None:
        raise ValueError("moving-magnet MMD needs a magnet network")
    t = lambda x: torch.as_tensor(x, device=device)  # noqa: E731
    obs, masks, actions = t(batch.obs), t(batch.masks), t(batch.actions)
    old_logps, advantages, returns = t(batch.logps), t(batch.advantages), t(batch.returns)

    n = len(batch)
    keys = ["policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac"] + (["magnet_kl"] if use_mmd else [])
    stats: dict[str, list[float]] = {k: [] for k in keys}
    net.train()
    for _ in range(cfg.epochs):
        order = rng.permutation(n)
        for start in range(0, n, cfg.minibatch_size):
            idx = torch.as_tensor(order[start:start + cfg.minibatch_size], device=device)
            dist, value = net.distribution(obs[idx], masks[idx])
            new_logps = dist.log_prob(actions[idx])
            log_ratio = new_logps - old_logps[idx]
            ratio = log_ratio.exp()

            adv = advantages[idx]
            if cfg.norm_adv and len(idx) > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            policy_loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
            value_loss = 0.5 * (value - returns[idx]).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss - cfg.ent_coef * entropy + cfg.vf_coef * value_loss
            if use_mmd:
                magnet_logits = None
                if mmd.magnet == "moving":
                    with torch.no_grad():
                        magnet_logits = magnet.actor(obs[idx])
                kl_magnet = magnet_kl(dist.logits, magnet_logits, masks[idx])
                loss = loss + mmd.alpha * kl_magnet

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.item())
                stats["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                stats["clipfrac"].append(((ratio - 1).abs() > cfg.clip_coef).float().mean().item())
                if use_mmd:
                    stats["magnet_kl"].append(kl_magnet.item())

    out = {k: float(np.mean(v)) for k, v in stats.items()}
    var_returns = float(np.var(batch.returns))
    out["explained_variance"] = float("nan") if var_returns == 0 else 1 - float(np.var(batch.returns - batch.values)) / var_returns
    out["samples"] = n
    return out
