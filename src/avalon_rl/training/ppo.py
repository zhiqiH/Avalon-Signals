"""Clipped PPO update for one team's network."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..agents.policy import ActorCritic
from .rollout import Batch


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


def ppo_update(
    net: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    cfg: PPOConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Run `cfg.epochs` passes of minibatch PPO over `batch` and return mean diagnostics.

    Returns lie in [-1, 1] (terminal +/-1 reward, gamma = 1), so the value target needs no normalization.
    """
    t = lambda x: torch.as_tensor(x, device=device)  # noqa: E731
    obs, masks, actions = t(batch.obs), t(batch.masks), t(batch.actions)
    old_logps, advantages, returns = t(batch.logps), t(batch.advantages), t(batch.returns)

    n = len(batch)
    stats: dict[str, list[float]] = {k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac")}
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

    out = {k: float(np.mean(v)) for k, v in stats.items()}
    var_returns = float(np.var(batch.returns))
    out["explained_variance"] = float("nan") if var_returns == 0 else 1 - float(np.var(batch.returns - batch.values)) / var_returns
    out["samples"] = n
    return out
