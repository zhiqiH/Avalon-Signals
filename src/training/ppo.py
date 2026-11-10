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
    # The int8 communication features are the bulk of a batch: keep them in host memory and move one
    # minibatch at a time, so several runs can share one GPU.
    comm_obs = torch.as_tensor(batch.comm_obs) if net.comm else None
    old_logps, advantages, returns = t(batch.logps), t(batch.advantages), t(batch.returns)
    is_msg = actions[:, 0] < 0  # message decisions use no game-action head

    n = len(batch)
    keys = ["policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac"]
    keys += ["msg_entropy"] if net.comm else []
    stats: dict[str, list[float]] = {k: [] for k in keys}
    net.train()
    for _ in range(cfg.epochs):
        order = rng.permutation(n)
        for start in range(0, n, cfg.minibatch_size):
            idx = torch.as_tensor(order[start:start + cfg.minibatch_size], device=device)
            mb_comm = comm_obs[idx.cpu()].to(device, non_blocking=True) if comm_obs is not None else None
            new_logps, entropies, value, _ = net.evaluate(obs[idx], mb_comm, masks[idx], actions[idx])
            log_ratio = new_logps - old_logps[idx]
            ratio = log_ratio.exp()

            adv = advantages[idx]
            if cfg.norm_adv and len(idx) > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            policy_loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
            value_loss = 0.5 * (value - returns[idx]).pow(2).mean()
            loss = policy_loss - cfg.ent_coef * entropies.mean() + cfg.vf_coef * value_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                msg_rows = is_msg[idx]
                if (~msg_rows).any():  # entropy of game-action decisions, comparable with the silent baseline
                    stats["entropy"].append(entropies[~msg_rows].mean().item())
                if net.comm and msg_rows.any():
                    stats["msg_entropy"].append(entropies[msg_rows].mean().item())
                stats["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                stats["clipfrac"].append(((ratio - 1).abs() > cfg.clip_coef).float().mean().item())

    out = {k: float(np.mean(v)) if v else float("nan") for k, v in stats.items()}
    var_returns = float(np.var(batch.returns))
    out["explained_variance"] = float("nan") if var_returns == 0 else 1 - float(np.var(batch.returns - batch.values)) / var_returns
    out["samples"] = n
    return out
