"""Team dialect probe: does a single public message give away its speaker's team, and how?

    python scripts/dialect_probe.py runs/comm_v3/stage3b_M/seed*/checkpoint_01000.pt --out results/tables/dialect.json

Each checkpoint plays same-seed self-play with paired seating. Every message of the public discussions is one
sample; a logistic probe predicts "the speaker is evil" and is scored by held-out log-loss gain over the base
rate (nats; the base rate's entropy is 0.64) and by accuracy (always guessing "good" gives 0.67).

- `form`: how many judgments of each value and rationales of each type the message holds, not whom they are
  about. Nothing a human listener would read as an argument, so a gain here is a code.
- `full`: every head's value, seats included.
- `first`: the same two probes on the first discussion only, where no record exists yet. A Servant knows
  nothing then, so whatever separates the teams there is not evidence.

Next to the probes: what each role says (accusations and endorsements per message, and how often they are
right), and the public audit counts each role has collected by the end of a game.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root, so `import src` works

from src.agents.policy import PolicyController  # noqa: E402
from src.comm import protocol as P  # noqa: E402
from src.env.rules import N_QUESTS, Role  # noqa: E402
from src.evaluation.evaluate import paired_games  # noqa: E402
from src.runner import play  # noqa: E402
from src.training.checkpoint import build_net, comm_config, load_checkpoint  # noqa: E402
from src.training.rollout import TEAMS  # noqa: E402

SEED = 20261020
_GROUPS = (range(P.H_JUDGE, P.H_RTYPE), range(P.H_RTYPE, P.H_RROUND), range(P.H_RROUND, P.H_TEAM))


def play_games(path: str, n: int) -> list:
    ck = load_checkpoint(path)
    nets = {t: build_net(ck["config"]) for t in TEAMS}
    for t in TEAMS:
        nets[t].load_state_dict(ck["nets"][t])
    ctrl = {t: PolicyController(nets[t], learn=False) for t in TEAMS}
    rng = np.random.default_rng([SEED, 1_000_003])
    games = []
    for start in range(0, n, 500):
        batch = paired_games(SEED, start, min(start + 500, n), comm_config(ck["config"]))
        play(batch, [ctrl["good"]] * len(batch), [ctrl["evil"]] * len(batch), rng)
        games += batch
    return games


def _probe(x: np.ndarray, y: np.ndarray, gid: np.ndarray) -> dict:
    test = gid % 5 == 0  # split by game
    xt, yt = torch.tensor(x[~test]), torch.tensor(y[~test])
    w, b = torch.zeros(x.shape[1], requires_grad=True), torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=300)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(xt @ w + b, yt) + 1e-3 * w.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    xv, yv = torch.tensor(x[test]), torch.tensor(y[test])
    with torch.no_grad():
        p = torch.sigmoid(xv @ w + b).clamp(1e-6, 1 - 1e-6)
        ll = -(yv * p.log() + (1 - yv) * (1 - p).log()).mean().item()
        base = yt.mean().item()
        ll0 = -(yv * np.log(base) + (1 - yv) * np.log(1 - base)).mean().item()
        acc = ((p > 0.5).float() == yv).float().mean().item()
    return {"info_gain_nats": round(ll0 - ll, 4), "accuracy": round(acc, 3), "messages": int(len(y))}


def analyze(job: tuple[str, int]) -> tuple[str, dict]:
    path, n = job
    torch.set_num_threads(4)
    games = play_games(path, n)
    form, full, y, gid, quest = [], [], [], [], []
    said = {r.name.lower(): np.zeros(5) for r in Role}  # messages, accusations, right ones, endorsements, right ones
    audit = {r.name.lower(): [] for r in Role}
    for i, game in enumerate(games):
        evil = np.array([game.is_evil_seat(s) for s in range(len(game.roles))])
        for s, role in enumerate(game.roles):
            audit[Role(int(role)).name.lower()].append(game.audit[s])
        for q in range(N_QUESTS):
            for s in range(len(game.roles)):
                m = game.msg_log[q, s].astype(int)
                if m[0] < 0:
                    continue
                onehots = []
                for h, size in enumerate(P.MSG_HEAD_SIZES):
                    v = np.zeros(size + 1)
                    v[m[h] + 1] = 1
                    onehots.append(v)
                counts = [np.bincount(m[list(g)] + 1, minlength=P.MSG_HEAD_SIZES[g[0]] + 1) for g in _GROUPS]
                form.append(np.concatenate(counts + [np.eye(N_QUESTS)[q]]))
                full.append(np.concatenate(onehots + [np.eye(N_QUESTS)[q]]))
                y.append(float(evil[s]))
                gid.append(i)
                quest.append(q)
                direction = P.JUDGE_DIR[np.maximum(m[P.H_JUDGE:P.H_RTYPE], 0)]
                target_evil = evil[P._others(s)]
                said[Role(int(game.roles[s])).name.lower()] += (
                    1, (direction == 1).sum(), ((direction == 1) & target_evil).sum(),
                    (direction == 0).sum(), ((direction == 0) & ~target_evil).sum())
    form, full = np.array(form, np.float32), np.array(full, np.float32)
    y, gid, first = np.array(y, np.float32), np.array(gid), np.array(quest) == 0
    res = {"games": len(games), "good_win": round(float(np.mean([g.good_won for g in games])), 4),
           "all": {"form": _probe(form, y, gid), "full": _probe(full, y, gid)},
           "first": {"form": _probe(form[first], y[first], gid[first]), "full": _probe(full[first], y[first], gid[first])},
           "says": {}, "audit_per_game": {}}
    for role, (n_msg, acc, acc_ok, end, end_ok) in said.items():
        res["says"][role] = {"accusations_per_message": round(acc / n_msg, 3),
                             "accusations_right": round(acc_ok / acc, 3) if acc else None,
                             "endorsements_per_message": round(end / n_msg, 3),
                             "endorsements_right": round(end_ok / end, 3) if end else None}
    for role, rows in audit.items():
        rows = np.array(rows)
        res["audit_per_game"][role] = {**{name: round(float(rows[:, k].mean()), 3) for k, name in enumerate(P.AUDIT)},
                                       "any": round(float((rows.sum(1) > 0).mean()), 3)}
    return path, res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--out")
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=min(len(args.checkpoints), 6)) as ex:
        result = dict(ex.map(analyze, [(c, args.games) for c in args.checkpoints]))
    for path, r in result.items():
        print(f"== {path}: good win {r['good_win']:.3f}")
        for scope in ("all", "first"):
            f, a = r[scope]["form"], r[scope]["full"]
            print(f"  {scope:5s} form {f['info_gain_nats']:.3f} nats, acc {f['accuracy']:.3f} | "
                  f"full {a['info_gain_nats']:.3f} nats, acc {a['accuracy']:.3f} | {f['messages']} messages")
        for role, s in r["says"].items():
            print(f"  {role:9s} accuses {s['accusations_per_message']:.2f} (right {s['accusations_right']}), "
                  f"endorses {s['endorsements_per_message']:.2f} (right {s['endorsements_right']}), "
                  f"audit {r['audit_per_game'][role]}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
