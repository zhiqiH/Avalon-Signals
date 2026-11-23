"""Training configuration: dataclasses loaded from YAML with optional dotted-key overrides."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .comm.protocol import CommConfig
from .training.ppo import PPOConfig


@dataclass
class ModelConfig:
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    deduction: bool = True  # read the deduction fields of the observation (checkpoints before 2026-11-12: false)


@dataclass
class OpponentConfig:
    # Per training game: both teams use the current policies, or one team (chosen uniformly)
    # is replaced by a past snapshot or by the rule-based player, or (partner) both teams use the
    # current policies but one or two good seats are played by the logic partner (agents/logic_agent.py;
    # in a communication game the rule teacher speaks for those seats).
    # Only current policies learn.
    p_self: float = 0.5
    p_pool: float = 0.25
    p_rule: float = 0.25
    p_partner: float = 0.0
    partner_max: int = 2  # partner games put 1..partner_max logic partners in Servant/Percival seats
    partner_slack: float = 0.1  # the partner approves teams within this chance-of-clean of the best team
    snapshot_every: int = 10  # iterations between snapshots added to the pool
    pool_size: int = 20  # snapshots kept per team
    rule_fail_prob: float = 0.9
    rule_merlin_knowledge_prob: float = 0.5


@dataclass
class EvalConfig:
    every: int = 20  # iterations between evaluations against the rule-based players
    games: int = 512


@dataclass
class AdaptConfig:
    """How a run adapts a starting network (communication pilot; src/training/adapt.py). Defaults: train everything."""
    lora_rank: int = 0  # >0: the actor changes only through rank-r updates; new message parts and the critic train fully
    lora_scale: float = 1.0
    speaker_from: str | None = None  # listener stage: every learned player speaks with this checkpoint's frozen network
    msg_kl_coef: float = 0.0  # weight of KL(message heads || anchor) on message decisions
    msg_kl_anneal: int = 0  # iterations over which that weight falls linearly to msg_kl_floor (0: constant)
    msg_kl_floor: float = 0.0  # the weight never falls below this: talk stays tied to the anchor's meaning
    # Share of seats per training game (either team, drawn per seat) whose public messages come from the rule
    # teacher (agents/teacher.py) instead of their own policy; they still act for themselves. Listeners keep
    # meeting talk with the teacher's meaning, and a private code cannot rely on who is speaking.
    teacher_share: float = 0.0
    anchor_from: str | None = None  # checkpoint whose message heads are the KL anchor ("{seed}" is filled in)


@dataclass
class TrainConfig:
    name: str = "baseline_silent"
    seed: int = 0
    device: str = "auto"  # auto | cpu | cuda | mps; rollouts always run on CPU
    total_games: int = 100_000  # training budget B
    games_per_iter: int = 512
    checkpoint_every: int = 20
    workers: int = 0  # rollout worker processes per run (0: play in the training process; src/training/workers.py)
    out_dir: str = "runs"
    init_from: str | None = None  # load network weights only (warm start / branch point); "{seed}" is filled in
    resume: bool = False  # continue from <out_dir>/<name>/seed<seed>/latest.pt if it exists
    model: ModelConfig = field(default_factory=ModelConfig)
    comm: CommConfig = field(default_factory=CommConfig)  # disabled: the silent game
    ppo: PPOConfig = field(default_factory=PPOConfig)
    opponents: OpponentConfig = field(default_factory=OpponentConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    adapt: AdaptConfig = field(default_factory=AdaptConfig)

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
    probs = (cfg.opponents.p_self, cfg.opponents.p_pool, cfg.opponents.p_rule, cfg.opponents.p_partner)
    if abs(sum(probs) - 1.0) > 1e-6:
        raise ValueError(f"opponent probabilities must sum to 1, got {probs}")
    if cfg.opponents.p_partner > 0 and not 1 <= cfg.opponents.partner_max <= 3:
        raise ValueError("partner_max must be 1-3 (there are three Servant/Percival seats)")
    if not 0.0 <= cfg.adapt.teacher_share <= 1.0:
        raise ValueError("adapt.teacher_share must lie in [0, 1]")
    return cfg


def config_to_dict(cfg: TrainConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
