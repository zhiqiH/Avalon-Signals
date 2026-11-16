"""Rollout worker processes: each plays a share of an iteration's games on the CPU.

The training process keeps the learner networks (on any device), decides who controls each game
(`trainer.sample_matchups`) and sends every worker a contiguous share of the games. A worker holds
CPU copies of the current networks, a mirror of the opponent pool and the rule-based players, plays
its games with its own random stream, and returns the PPO batches (GAE already computed) and the game
outcomes. Results are combined in worker order, so a run is reproducible for a fixed number of workers;
it differs from a run with another number of workers, or with none, only by random-number streams.

Evaluation games are split the same way and keep their paired role assignments (game i always gets
`evaluate.paired_games`' seating for game i); only the policies' sampling streams differ from the
single-process `run_matchup`.
"""
from __future__ import annotations

import multiprocessing as mp
import traceback
from typing import Sequence

import numpy as np
import torch

from ..agents.policy import ActorCritic, PolicyController
from ..agents.rule_agent import RuleController
from ..config import TrainConfig
from ..env.game import AvalonGame
from ..evaluation.evaluate import paired_games
from ..evaluation.metrics import game_outcome, summarize
from ..runner import play
from .opponent_pool import OpponentPool
from .rollout import TEAMS, Batch, TrajectoryRecorder

# Who controls a team in one game: the current network, the rule-based players, or (an int) a pool snapshot.
LEARNER, RULE = "learner", "rule"
Spec = str | int


def resolve(spec: Spec, team: str, learners: dict, pool: OpponentPool, rule: RuleController):
    if spec == LEARNER:
        return learners[team]
    if spec == RULE:
        return rule
    return pool.snapshots[team][spec]


def _spec_key(spec: Spec) -> tuple[int, int]:
    return (0, 0) if spec == LEARNER else (1, 0) if spec == RULE else (2, int(spec))


def cpu_state(net: torch.nn.Module) -> dict[str, np.ndarray]:
    """Weights as plain numpy arrays: torch tensors would travel as shared-memory file descriptors, one per
    tensor, and a pool refill to many workers runs out of open files."""
    return {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}


def _load(net: torch.nn.Module, state: dict[str, np.ndarray]) -> None:
    net.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})


def concat_batches(parts: Sequence[dict[str, Batch]]) -> dict[str, Batch]:
    # A share without decisions of one team yields an empty batch whose columns may not match.
    return {t: Batch.concat([p[t] for p in parts if len(p[t])]) for t in TEAMS}


def _new_net(cfg: TrainConfig) -> ActorCritic:
    return ActorCritic(hidden_sizes=cfg.model.hidden_sizes, comm=cfg.comm.enabled, deduction=cfg.model.deduction)


def _worker_main(conn, cfg: TrainConfig) -> None:
    torch.set_num_threads(1)  # many small batches: one core per worker is fastest
    try:
        nets = {t: _new_net(cfg) for t in TEAMS}
        scratch = _new_net(cfg)
        learners = {t: PolicyController(nets[t], learn=True) for t in TEAMS}
        frozen = {t: PolicyController(nets[t], learn=False) for t in TEAMS}
        rule = RuleController(cfg.opponents.rule_fail_prob, cfg.opponents.rule_merlin_knowledge_prob)
        pool = OpponentPool(cfg.opponents.pool_size)
        while True:
            cmd, *args = conn.recv()
            if cmd == "weights":
                for t, state in args[0].items():
                    _load(nets[t], state)
            elif cmd == "pool_add_current":
                for t in TEAMS:
                    pool.add(t, nets[t])
            elif cmd == "pool_add":
                for t, state in args[0].items():
                    _load(scratch, state)
                    pool.add(t, scratch)
            elif cmd == "play":
                seed, specs = args
                rng = np.random.default_rng(seed)
                games = [AvalonGame(rng, comm=cfg.comm) for _ in specs]
                good = [resolve(g, "good", learners, pool, rule) for g, _ in specs]
                evil = [resolve(e, "evil", learners, pool, rule) for _, e in specs]
                recorder = TrajectoryRecorder()
                play(games, good, evil, rng, on_decision=recorder)
                batches = recorder.build(games, cfg.ppo.gamma, cfg.ppo.gae_lambda)
                conn.send(("ok", (batches, [game_outcome(g) for g in games])))
            elif cmd == "eval":
                good_spec, evil_spec, seed, start, stop = args
                games = paired_games(seed, start, stop, cfg.comm)
                good = frozen["good"] if good_spec == LEARNER else rule
                evil = frozen["evil"] if evil_spec == LEARNER else rule
                play(games, [good] * len(games), [evil] * len(games), np.random.default_rng([seed, 1_000_003, start]))
                conn.send(("ok", [game_outcome(g) for g in games]))
            elif cmd == "close":
                return
            else:
                raise ValueError(f"unknown worker command {cmd!r}")
    except Exception:  # report to the training process instead of leaving it waiting
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


class RolloutWorkers:
    """A fixed set of worker processes; use as a context manager so they always shut down."""

    def __init__(self, n: int, cfg: TrainConfig):
        ctx = mp.get_context("spawn")  # never fork a process that may hold CUDA state
        self.conns, self.procs = [], []
        for _ in range(n):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker_main, args=(child, cfg), daemon=True)
            proc.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(proc)

    def __len__(self) -> int:
        return len(self.conns)

    def __enter__(self) -> "RolloutWorkers":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _broadcast(self, msg: tuple) -> None:
        for conn in self.conns:
            conn.send(msg)

    @staticmethod
    def _recv(conn):
        status, payload = conn.recv()
        if status == "error":
            raise RuntimeError(f"rollout worker failed:\n{payload}")
        return payload

    def set_weights(self, nets: dict[str, torch.nn.Module]) -> None:
        self._broadcast(("weights", {t: cpu_state(nets[t]) for t in TEAMS}))

    def pool_add_current(self) -> None:
        """Every worker adds its current networks to its pool (mirrors `OpponentPool.add` in the trainer)."""
        self._broadcast(("pool_add_current",))

    def pool_add(self, nets: dict[str, torch.nn.Module]) -> None:
        self._broadcast(("pool_add", {t: cpu_state(nets[t]) for t in TEAMS}))

    def _shares(self, n: int) -> list[tuple[int, int]]:
        bounds = np.linspace(0, n, len(self) + 1).round().astype(int)
        return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]

    def play(self, specs: Sequence[tuple[Spec, Spec]], rng: np.random.Generator) -> tuple[dict[str, Batch], list[dict]]:
        """Play one game per (good spec, evil spec); returns the per-team batches and the outcomes in game order.

        Games with the same controllers go to the same worker where possible, so each worker makes few,
        large network calls per tick instead of one tiny call per pool snapshot.
        """
        order = sorted(range(len(specs)), key=lambda i: (_spec_key(specs[i][0]), _spec_key(specs[i][1])))
        shares = self._shares(len(specs))
        seeds = rng.integers(2**63, size=len(shares))
        for conn, (a, b), seed in zip(self.conns, shares, seeds):
            conn.send(("play", int(seed), [specs[i] for i in order[a:b]]))
        results = [self._recv(conn) for conn in self.conns[:len(shares)]]
        outcomes: list[dict] = [{}] * len(specs)
        for i, outcome in zip(order, (o for r in results for o in r[1])):
            outcomes[i] = outcome
        return concat_batches([r[0] for r in results]), outcomes

    def evaluate(self, good_spec: Spec, evil_spec: Spec, n_games: int, seed: int) -> dict:
        """Frozen matchup of the current networks and/or the rule players with paired role assignments."""
        shares = self._shares(n_games)
        for conn, (a, b) in zip(self.conns, shares):
            conn.send(("eval", good_spec, evil_spec, seed, a, b))
        return summarize([o for conn in self.conns[:len(shares)] for o in self._recv(conn)])

    def close(self) -> None:
        for conn in self.conns:
            try:
                conn.send(("close",))
            except (BrokenPipeError, OSError):
                pass
        for proc in self.procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
        self.conns, self.procs = [], []
