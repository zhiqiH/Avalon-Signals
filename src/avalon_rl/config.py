"""Training configuration: dataclasses loaded from YAML with optional dotted-key overrides."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .training.ppo import MMDConfig, PPOConfig


@dataclass
class ModelConfig:
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])


@dataclass
class OpponentConfig:
    # Per training game: both teams use the current policies, or one team (chosen uniformly)
    # is replaced by a past snapshot or by the rule-based player. Only current policies learn.
    p_self: float = 0.5
    p_pool: float = 0.25
    p_rule: float = 0.25
    snapshot_every: int = 10  # iterations between snapshots added to the pool
    pool_size: int = 20  # snapshots kept per team
    rule_fail_prob: float = 0.9
    rule_merlin_knowledge_prob: float = 0.5


@dataclass
class EvalConfig:
    every: int = 20  # iterations between evaluations against the rule-based players
    games: int = 512


@dataclass
class TrainConfig:
    name: str = "baseline_silent"
    seed: int = 0
    device: str = "auto"  # auto | cpu | cuda | mps; rollouts always run on CPU
    total_games: int = 100_000  # training budget B
    games_per_iter: int = 512
    checkpoint_every: int = 20
    out_dir: str = "runs"
    init_from: str | None = None  # load network weights only (warm start / branch point)
    resume: bool = False  # continue from <out_dir>/<name>/seed<seed>/latest.pt if it exists
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    mmd: MMDConfig = field(default_factory=MMDConfig)
    opponents: OpponentConfig = field(default_factory=OpponentConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.name / f"seed{self.seed}"


def _from_dict(cls, data: dict[str, Any]):
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise KeyError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        default = getattr(cls(), name)
        kwargs[name] = _from_dict(type(default), value) if dataclasses.is_dataclass(default) else value
    return cls(**kwargs)


def _set_dotted(data: dict[str, Any], key: str, value: Any) -> None:
    *parents, leaf = key.split(".")
    for p in parents:
        data = data.setdefault(p, {})
    data[leaf] = value


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> TrainConfig:
    data: dict[str, Any] = {}
    if path is not None:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for key, value in (overrides or {}).items():
        if value is not None:
            _set_dotted(data, key, value)
    cfg = _from_dict(TrainConfig, data)
    probs = (cfg.opponents.p_self, cfg.opponents.p_pool, cfg.opponents.p_rule)
    if abs(sum(probs) - 1.0) > 1e-6:
        raise ValueError(f"opponent probabilities must sum to 1, got {probs}")
    return cfg


def config_to_dict(cfg: TrainConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
