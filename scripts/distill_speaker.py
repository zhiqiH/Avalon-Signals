"""Distil the rule teacher's messages into the message heads of a branched network (communication stage 1).

    python scripts/distill_speaker.py --baseline baseline/seed0/checkpoint_19532.pt --config configs/comm.yaml \
        --seed 0 --out runs/comm/distill/seed0

1. Branch both teams' networks from the frozen silent checkpoint exactly as training does (message inputs and
   heads at zero weight, so the network plays exactly like the baseline). The starting checkpoint can also be a
   communication network trained before the reply and promise slots: it keeps its weights, and the parts for
   those slots start at zero (`policy.upgrade_legacy_state`).
2. Play self-play games: actions come from these networks, every message from the rule teacher
   (src/agents/teacher.py). Record each speaking decision.
3. Fit only the message heads: the rows of the actor's output layer that belong to message heads, and the
   stage-B conditioning map. The trunk and the game-action rows stay fixed, so the network still plays exactly
   like the baseline, and listeners still ignore messages (their input weights stay zero). Loss: cross-entropy of the
   teacher's value at every head the message used, under the same consistency masks as sampling.
4. Save a checkpoint that training can start from (`init_from`), plus distill.json with held-out fit statistics.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root, so `import src` works

from src.agents.base import ActResult  # noqa: E402
from src.agents.policy import ActorCritic, PolicyController, load_silent_weights, masked_logits, stage_a_onehot  # noqa: E402
from src.agents.teacher import TeacherSpeaker  # noqa: E402
from src.comm import protocol as P  # noqa: E402
from src.config import config_to_dict, load_config  # noqa: E402
from src.env.actions import N_ACTIONS  # noqa: E402
from src.env.game import AvalonGame  # noqa: E402
from src.runner import play  # noqa: E402
from src.training.checkpoint import comm_config, load_checkpoint, save_checkpoint  # noqa: E402
from src.training.rollout import TEAMS  # noqa: E402


class Recorder:
    """Speaks as the teacher and records (game index, team, obs, comm obs, context, message) per decision."""

    def __init__(self, team: str, policy: PolicyController, teacher: TeacherSpeaker, store: list):
        self.team, self.policy, self.teacher, self.store, self.learn = team, policy, teacher, store, False

    def act(self, requests, rng):
        return self.policy.act(requests, rng)

    def speak(self, requests, rng):
        msgs = self.teacher.speak(requests, rng).messages
        obs, comm, _, _ = self.policy._inputs(requests)
        for i, r in enumerate(requests):
            self.store.append((r.game_idx + self.offset, self.team, obs[i], comm[i], r.ctx, msgs[i]))
        return ActResult(messages=msgs)


def collect(nets, comm, n_games: int, seed: int, merlin_knowledge_prob: float) -> list:
    store: list = []
    rec = {t: Recorder(t, PolicyController(nets[t], learn=False),
                       TeacherSpeaker(merlin_knowledge_prob=merlin_knowledge_prob, seed=seed + i), store)
           for i, t in enumerate(TEAMS)}
    rng = np.random.default_rng([seed, 1])
    for start in range(0, n_games, 500):
        for r in rec.values():
            r.offset = start
        games = [AvalonGame(np.random.default_rng([seed, 2, i]), comm=comm) for i in range(start, min(start + 500, n_games))]
        play(games, [rec["good"]] * len(games), [rec["evil"]] * len(games), rng)
    return store


def head_masks(msgs: np.ndarray, ctx: dict) -> np.ndarray:
    """(n, N_MSG_LOGITS) legal values of every head given the earlier heads of each recorded message."""
    out = np.zeros((len(msgs), P.N_MSG_LOGITS), dtype=bool)
    for h in P.STAGE_A + P.STAGE_B:
        lo = P.MSG_HEAD_OFFSETS[h]
        out[:, lo:lo + P.MSG_HEAD_SIZES[h]] = P.head_mask(h, msgs, ctx)
    return out


def head_nll(logits: torch.Tensor, masks: torch.Tensor, msgs: torch.Tensor, heads) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed negative log-likelihood over the used heads in `heads`, and the number of used heads, per row."""
    nll = logits.new_zeros(len(logits))
    used_n = logits.new_zeros(len(logits))
    for h in heads:
        lo, size = P.MSG_HEAD_OFFSETS[h], P.MSG_HEAD_SIZES[h]
        lp = torch.log_softmax(masked_logits(logits[:, lo:lo + size], masks[:, lo:lo + size]), -1)
        idx = msgs[:, h]
        used = idx >= 0
        nll = nll - torch.where(used, lp.gather(1, idx.clamp(min=0)[:, None])[:, 0], torch.zeros_like(nll))
        used_n = used_n + used.float()
    return nll, used_n


class MessageHeads(torch.nn.Module):
    """Trainable copy of a network's message-head rows and stage-B conditioning map."""

    def __init__(self, net: ActorCritic):
        super().__init__()
        out = net.actor[-1]
        self.weight = torch.nn.Parameter(out.weight[N_ACTIONS:].detach().clone())
        self.bias = torch.nn.Parameter(out.bias[N_ACTIONS:].detach().clone())
        self.cond = torch.nn.Parameter(net.actor_cond.weight.detach().clone())

    def logits(self, hidden: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        """Message logits in the protocol's head order (stage-B columns conditioned when `cond` is given)."""
        h = hidden if cond is None else hidden + cond @ self.cond.T
        return h @ self.weight.T + self.bias

    def write_into(self, net: ActorCritic) -> None:
        with torch.no_grad():
            net.actor[-1].weight[N_ACTIONS:] = self.weight
            net.actor[-1].bias[N_ACTIONS:] = self.bias
            net.actor_cond.weight.copy_(self.cond)


def fit(net: ActorCritic, rows: list, epochs: int, device: torch.device, log_prefix: str) -> dict:
    games = np.array([r[0] for r in rows])
    test = np.isin(games, np.unique(games)[::5])  # a fifth of the games held out
    obs = torch.tensor(np.stack([r[2] for r in rows]))
    comm = torch.tensor(np.stack([r[3] for r in rows]))
    msgs_np = np.stack([r[5] for r in rows]).astype(np.int64)
    ctx = P.stack_contexts([r[4] for r in rows])
    masks = torch.tensor(head_masks(msgs_np, ctx))
    msgs = torch.tensor(msgs_np)
    with torch.no_grad():  # the trunk is fixed: compute its features once
        hidden = torch.cat([net.hidden(obs[i:i + 4096], comm[i:i + 4096]) for i in range(0, len(obs), 4096)])
    a_onehot = stage_a_onehot(msgs)
    heads = MessageHeads(net).to(device)
    hidden, masks, msgs, a_onehot = hidden.to(device), masks.to(device), msgs.to(device), a_onehot.to(device)
    tr, te = torch.tensor(~test, device=device), torch.tensor(test, device=device)
    opt = torch.optim.Adam(heads.parameters(), lr=3e-3)

    def nll_of(idx):
        la = heads.logits(hidden[idx], None)
        lb = heads.logits(hidden[idx], a_onehot[idx])
        na, ca = head_nll(la, masks[idx], msgs[idx], P.STAGE_A)
        nb, cb = head_nll(lb, masks[idx], msgs[idx], P.STAGE_B)
        return na + nb, ca + cb

    tr_idx = torch.nonzero(tr)[:, 0]
    history = []
    for epoch in range(epochs):
        perm = tr_idx[torch.randperm(len(tr_idx), device=device)]
        for i in range(0, len(perm), 4096):
            nll, _ = nll_of(perm[i:i + 4096])
            loss = nll.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            nll_te, _ = nll_of(torch.nonzero(te)[:, 0])
        history.append(round(float(nll_te.mean()), 4))
        print(f"{log_prefix} epoch {epoch + 1}/{epochs}: held-out NLL per message {history[-1]:.3f}")
    heads.write_into(net)
    with torch.no_grad():
        idx = torch.nonzero(te)[:, 0]
        la = heads.logits(hidden[idx], None)
        judge_agree, judge_n = 0, 0
        by_value = {v: [0.0, 0] for v in ("none", "good", "evil", "merlin")}
        codes = {"none": P.J_NONE, "good": P.J_GOOD, "evil": P.J_EVIL, "merlin": P.J_MERLIN}
        for h in range(P.H_JUDGE, P.H_JUDGE + P.N_OTHERS):
            lo, size = P.MSG_HEAD_OFFSETS[h], P.MSG_HEAD_SIZES[h]
            probs = torch.softmax(masked_logits(la[:, lo:lo + size], masks[idx, lo:lo + size]), -1)
            pick = probs.argmax(-1)
            target = msgs[idx, h]
            used = target >= 0
            judge_agree += int(((pick == target) & used).sum())
            judge_n += int(used.sum())
            for name, code in codes.items():  # mean probability the sampled message repeats the teacher's value
                sel = target == code
                by_value[name][0] += float(probs[sel, code].sum())
                by_value[name][1] += int(sel.sum())
        uniform = torch.zeros_like(la)
        nll_u, _ = head_nll(uniform, masks[idx], msgs[idx], P.STAGE_A + P.STAGE_B)
    return {"messages": len(rows), "held_out": int(test.sum()), "held_out_nll_uniform": round(float(nll_u.mean()), 4),
            "held_out_nll_by_epoch": history, "judgment_agreement_greedy": round(judge_agree / max(judge_n, 1), 4),
            "p_repeat_teacher_judgment": {k: (round(a / n, 4) if n else None) for k, (a, n) in by_value.items()},
            "teacher_judgment_counts": {k: n for k, (_, n) in by_value.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True, help="checkpoint to branch from (silent, or a communication one)")
    parser.add_argument("--config", required=True, help="communication config (its comm block and model)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--games", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--merlin-knowledge-prob", type=float, default=0.5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    t0 = time.time()
    cfg = load_config(args.config, {"seed": args.seed, "init_from": args.baseline})
    silent = load_checkpoint(args.baseline)
    nets = {}
    for t in TEAMS:
        nets[t] = ActorCritic(hidden_sizes=cfg.model.hidden_sizes, comm=True, deduction=cfg.model.deduction)
        if comm_config(silent["config"]).enabled:
            nets[t].load_state_dict(silent["nets"][t])
        else:
            load_silent_weights(nets[t], silent["nets"][t])
    torch.set_num_threads(8)
    rows = collect(nets, cfg.comm, args.games, args.seed * 1000 + 7, args.merlin_knowledge_prob)
    print(f"collected {len(rows)} speaking decisions from {args.games} games in {time.time() - t0:.0f}s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = {t: fit(nets[t], [r for r in rows if r[1] == t], args.epochs, device, t) for t in TEAMS}
    for t in TEAMS:
        nets[t].cpu()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_checkpoint(out / "checkpoint_00000.pt", iteration=0, games_seen=0, config=config_to_dict(cfg),
                    nets={t: nets[t].state_dict() for t in TEAMS}, distilled_from="src/agents/teacher.py")
    info = {"baseline": args.baseline, "config": args.config, "games": args.games, "epochs": args.epochs,
            "merlin_knowledge_prob": args.merlin_knowledge_prob, "seconds": round(time.time() - t0), "fit": stats}
    (out / "distill.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
