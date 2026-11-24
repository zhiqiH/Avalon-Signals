"""Communication diagnostics: is talk informative, is it used, and does the learning signal reach it?

    python scripts/comm_diagnostics.py --checkpoint runs/comm/talk/seed0/checkpoint_01000.pt --out d.json
    python scripts/comm_diagnostics.py --branch baseline/seed0/checkpoint_19532.pt --config configs/comm.yaml --out d0.json

`--branch` builds the communication network exactly as training does at the branch point (silent weights,
zero weights on every message input and output). `--speaker` lets another checkpoint's frozen networks do all
the talking, as in the listener stage. All games are same-seed self-play with paired seating.

1. Speaker: for every (speaker, other player) pair in the public discussions, by the speaker's role:
   - judgment use and accusation precision (judged evil and truly evil) against the base rate;
   - information beyond public evidence: held-out log-loss gain (nats per pair) of a logistic probe for
     "this player is evil" when the speaker's judgment is added to public features (public suspicion,
     publicly certain evil / good, mission number);
   - the same two numbers for a babbler that speaks uniformly within the same consistency masks in the
     same states, the reference for what the masks alone make look informative.
2. Listener: total-variation distance between a listener's vote or proposal distribution with the
   current mission's real messages and with the babbler's messages in their place.
3. Play: good win rate with real talk, with babbler talk (masks kept) and with every slot silenced
   (silence is out of distribution for a talking network, so it is the weakest of the three); with
   --teacher also with every player speaking as the rule teacher (src/agents/teacher.py).
4. Learning signal: policy-gradient directions of the message heads (and of the listeners' message
   inputs) from two independent 256-game batches. A cosine near 0 means the gradient is noise. The
   entropy term's gradient norm on the message heads is reported next to the policy gradient's.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root, so `import src` works

from src.agents.base import ActResult  # noqa: E402
from src.agents.policy import ActorCritic, PolicyController, load_silent_weights, masked_logits  # noqa: E402
from src.agents.rule_agent import public_suspicion  # noqa: E402
from src.agents.teacher import TeacherSpeaker  # noqa: E402
from src.comm import protocol as P  # noqa: E402
from src.config import load_config  # noqa: E402
from src.env import actions as A  # noqa: E402
from src.env.observation import _certain, possible_evil_pairs  # noqa: E402
from src.env.rules import N_PLAYERS, Phase, Role  # noqa: E402
from src.evaluation.evaluate import paired_games  # noqa: E402
from src.evaluation.metrics import game_outcome, summarize  # noqa: E402
from src.runner import play  # noqa: E402
from src.training.checkpoint import build_net, comm_config, load_checkpoint  # noqa: E402
from src.training.rollout import TEAMS, TrajectoryRecorder  # noqa: E402

ROLE_NAMES = {int(r): r.name.lower() for r in Role}
SEED = 20261020
GRAD_GAMES = 256  # games per batch for the gradient comparison


# ---------------------------------------------------------------------- loading
def speakers_for(args) -> dict[str, PolicyController]:
    """--speaker: a checkpoint whose frozen networks do all the talking (the listener stage); else none."""
    if not args.speaker:
        return {}
    from src.training.checkpoint import load_policies
    return {t: PolicyController(net, learn=False) for t, net in load_policies(args.speaker).items()}


def load_nets(args) -> tuple[dict[str, ActorCritic], P.CommConfig, dict]:
    if args.checkpoint:
        ck = load_checkpoint(args.checkpoint)
        nets = {t: build_net(ck["config"]) for t in TEAMS}
        for t in TEAMS:
            nets[t].load_state_dict(ck["nets"][t])
        return nets, comm_config(ck["config"]), ck["config"]
    cfg = load_config(args.config)
    silent = load_checkpoint(args.branch)
    nets = {}
    for t in TEAMS:
        nets[t] = ActorCritic(hidden_sizes=cfg.model.hidden_sizes, comm=True, deduction=cfg.model.deduction)
        load_silent_weights(nets[t], silent["nets"][t])
    return nets, cfg.comm, {"ppo": vars(cfg.ppo)}


# ---------------------------------------------------------------------- speaker / listener instrumentation
class Instrumented:
    """Wraps a team's policy: logs every message next to a mask-matched babbler message, can speak as the
    babbler, and measures listeners' sensitivity to the current mission's messages."""

    def __init__(self, inner: PolicyController, babble: bool, listen_prob: float, seed: int, teacher=None):
        self.inner, self.babble, self.listen_prob, self.learn = inner, babble, listen_prob, False
        self.teacher = teacher  # speak as the rule teacher instead of the network
        self.rng = np.random.default_rng(seed)
        self.pairs: list[tuple] = []  # (game key, speaker role, judged evil, public features, dir, babbler dir)
        self.babble_feat: dict[tuple[int, int], dict[int, np.ndarray]] = defaultdict(dict)
        self.listen: dict[str, list[float]] = defaultdict(list)
        self.evil_talk: list[tuple[int, int]] = []  # (judged Merlin correctly, judged anyone)

    def speak(self, requests, rng):
        ctx = P.stack_contexts([r.ctx for r in requests])
        babbled = P.sample_messages(ctx, self.rng).astype(np.int64)
        if self.babble:
            msgs = babbled
        elif self.teacher is not None:
            msgs = np.asarray(self.teacher.speak(requests, rng).messages, dtype=np.int64)
        else:
            msgs = np.asarray(self.inner.speak(requests, rng).messages, dtype=np.int64)
        for r, m, b in zip(requests, msgs, babbled):
            self._log(r.game, r.player, m, b)
        return ActResult(messages=msgs.astype(np.int8))

    def _log(self, game, speaker: int, msg: np.ndarray, bab: np.ndarray) -> None:
        others = P._others(speaker)
        if game.phase == Phase.EVIL_DISCUSS:
            for r, seat in enumerate(others):
                if msg[P.H_JUDGE + r] == P.J_MERLIN:
                    self.evil_talk.append((int(game.roles[seat] == Role.MERLIN), 1))
            return
        q = game.quest
        self.babble_feat[(id(game), q)][speaker] = P.message_features(speaker, bab, game.team_size)
        susp = public_suspicion(game)
        ce, cg = _certain(possible_evil_pairs(game))
        role = ROLE_NAMES[int(game.roles[speaker])]
        for r, seat in enumerate(others):
            feats = (susp[seat], float(ce[seat]), float(cg[seat]), q)
            self.pairs.append((id(game), role, int(game.is_evil_seat(seat)), feats,
                               int(P.JUDGE_DIR[msg[P.H_JUDGE + r]]), int(P.JUDGE_DIR[bab[P.H_JUDGE + r]])))

    @torch.no_grad()
    def _probs(self, requests) -> np.ndarray:
        _, _, t_obs, t_comm = self.inner._inputs(requests)
        logits = self.inner.net.logits(t_obs, t_comm)[:, :A.N_ACTIONS]
        mask = torch.from_numpy(np.stack([r.mask for r in requests]))
        return torch.softmax(masked_logits(logits, mask), -1).numpy()

    def act(self, requests, rng):
        if self.listen_prob > 0:
            sel = [r for r in requests if r.game.phase in (Phase.PROPOSE, Phase.VOTE)
                   and self.rng.random() < self.listen_prob and (id(r.game), r.game.quest) in self.babble_feat]
            if sel:
                p_real = self._probs(sel)
                saved = {}
                for r in sel:
                    g = r.game
                    if id(g) not in saved:
                        saved[id(g)] = (g, g.msg_feat[g.quest].copy())
                        for s, f in self.babble_feat[(id(g), g.quest)].items():
                            g.msg_feat[g.quest, s] = f
                p_bab = self._probs(sel)
                for g, feat in saved.values():
                    g.msg_feat[g.quest] = feat
                for r, a, b in zip(sel, p_real, p_bab):
                    kind = "vote" if r.game.phase == Phase.VOTE else "propose"
                    self.listen[f"{ROLE_NAMES[int(r.game.roles[r.player])]}/{kind}"].append(0.5 * float(np.abs(a - b).sum()))
        return self.inner.act(requests, rng)


# ---------------------------------------------------------------------- speaker information
def _design(feats: np.ndarray, dirs: np.ndarray | None) -> np.ndarray:
    susp, ce, cg, q = feats.T
    cols = [np.ones(len(feats)), susp, susp ** 2, ce, cg] + [(q == k).astype(float) for k in range(1, 5)]
    if dirs is not None:
        cols += [(dirs == 0).astype(float), (dirs == 1).astype(float)]  # judged good / judged evil (none is the base)
    return np.stack(cols, 1)


def _logloss(x_tr, y_tr, x_te, y_te) -> float:
    x_tr, x_te = torch.tensor(x_tr, dtype=torch.float64), torch.tensor(x_te, dtype=torch.float64)
    y_tr, y_te = torch.tensor(y_tr, dtype=torch.float64), torch.tensor(y_te, dtype=torch.float64)
    w = torch.zeros(x_tr.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x_tr @ w, y_tr) + 1e-4 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return float(torch.nn.functional.binary_cross_entropy_with_logits(x_te @ w, y_te))


def speaker_info(pairs: list[tuple]) -> dict:
    out = {}
    by_role = defaultdict(list)
    for p in pairs:
        by_role[p[1]].append(p)
    for role, rows in sorted(by_role.items()):
        keys = np.array([r[0] for r in rows])
        test = np.isin(keys, np.unique(keys)[1::2])  # split by game
        y = np.array([r[2] for r in rows], float)
        feats = np.array([r[3] for r in rows], float)
        res = {"pairs": len(rows), "base_rate_evil": round(float(y.mean()), 4)}
        for name, col in (("talk", 4), ("babbler", 5)):
            d = np.array([r[col] for r in rows])
            acc = d == 1
            gain = _logloss(_design(feats[~test], None), y[~test], _design(feats[test], None), y[test]) - \
                _logloss(_design(feats[~test], d[~test]), y[~test], _design(feats[test], d[test]), y[test])
            res[name] = {"judged_share": round(float((d >= 0).mean()), 4), "accused_share": round(float(acc.mean()), 4),
                         "accusation_precision": round(float(y[acc].mean()), 4) if acc.any() else None,
                         "info_gain_nats": round(gain, 5)}
        out[role] = res
    return out


# ---------------------------------------------------------------------- play conditions
def run_condition(job) -> tuple[str, dict]:
    args, mode = job
    torch.set_num_threads(2)
    nets, comm, _ = load_nets(args)
    if mode == "silence":
        comm = P.CommConfig(**{**vars(comm), "silence": [s for s in P.SLOTS if comm.is_open(s)]})
    modes = ("talk", "babble", "silence", "teacher")
    speakers = speakers_for(args)
    ctrls = {t: Instrumented(PolicyController(nets[t], learn=False, speaker=speakers.get(t)), babble=(mode == "babble"),
                             listen_prob=0.25 if mode in ("talk", "teacher") else 0.0,
                             seed=SEED + 10 * modes.index(mode) + TEAMS.index(t),
                             teacher=TeacherSpeaker(seed=SEED + TEAMS.index(t)) if mode == "teacher" else None)
             for t in TEAMS}
    rng = np.random.default_rng([SEED, 1_000_003])
    outcomes = []
    for start in range(0, args.games, 500):
        games = paired_games(SEED, start, min(start + 500, args.games), comm)
        play(games, [ctrls["good"]] * len(games), [ctrls["evil"]] * len(games), rng)
        outcomes += [game_outcome(g) for g in games]
        for c in ctrls.values():
            c.babble_feat.clear()
    s = summarize(outcomes)
    res = {k: round(s[k]["rate"], 4) for k in ("good_win", "three_successes", "assassin_hit_given_three_successes")}
    if mode in ("talk", "teacher"):
        res["speaker"] = speaker_info(ctrls["good"].pairs + ctrls["evil"].pairs)
        lis = {k: v for c in ctrls.values() for k, v in c.listen.items()}
        res["listener_tv"] = {k: {"mean": round(float(np.mean(v)), 4), "share_over_0.1": round(float(np.mean(np.array(v) > 0.1)), 4),
                                  "n": len(v)} for k, v in sorted(lis.items())}
        et = ctrls["evil"].evil_talk
        res["evil_talk_merlin_named_correct"] = round(float(np.mean([e[0] for e in et])), 4) if et else None
    return mode, res


# ---------------------------------------------------------------------- learning signal
def grad_signal(job) -> tuple[str, dict]:
    args, _ = job
    torch.set_num_threads(4)
    nets, comm, config = load_nets(args)
    ppo = config["ppo"]
    grads = {t: [] for t in TEAMS}
    for b in range(2):
        rng = np.random.default_rng([SEED, 77, b])
        games = paired_games(SEED + 1 + b, 0, GRAD_GAMES, comm)
        rec = TrajectoryRecorder()
        speakers = speakers_for(args)
        ctrl = {t: PolicyController(nets[t], learn=True, speaker=speakers.get(t)) for t in TEAMS}
        play(games, [ctrl["good"]] * GRAD_GAMES, [ctrl["evil"]] * GRAD_GAMES, rng, on_decision=rec)
        batches = rec.build(games, ppo["gamma"], ppo["gae_lambda"])
        for t in TEAMS:
            net, bt = nets[t], batches[t]
            obs, comm_obs = torch.as_tensor(bt.obs), torch.as_tensor(bt.comm_obs)
            masks, actions = torch.as_tensor(bt.masks), torch.as_tensor(bt.actions)
            adv = torch.as_tensor(bt.advantages)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            is_msg = (actions[:, 0] < 0).float()
            logp, ent, _, _ = net.evaluate(obs, comm_obs, masks, actions)
            n = len(bt)
            terms = {"msg_pg": -(adv * logp * is_msg).sum() / n, "msg_entropy": -ppo["ent_coef"] * (ent * is_msg).sum() / n,
                     "game_pg": -(adv * logp * (1 - is_msg)).sum() / n}
            out = {}
            for name, loss in terms.items():
                net.zero_grad()
                loss.backward(retain_graph=True)
                out[name + "/heads"] = net.actor[-1].weight.grad[A.N_ACTIONS:].flatten().clone()
                out[name + "/listen"] = net.actor[0].weight.grad[:, net.obs_dim:].flatten().clone()
            grads[t].append(out)
    res = {}
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=0)) if a.norm() > 0 and b.norm() > 0 else None  # noqa: E731
    for t in TEAMS:
        g0, g1 = grads[t]
        res[t] = {
            "msg_heads_pg_cosine_between_batches": cos(g0["msg_pg/heads"], g1["msg_pg/heads"]),
            "listen_inputs_game_pg_cosine_between_batches": cos(g0["game_pg/listen"], g1["game_pg/listen"]),
            "msg_heads_norm_pg": round(float(g0["msg_pg/heads"].norm()), 6),
            "msg_heads_norm_entropy": round(float(g0["msg_entropy/heads"].norm()), 6),
            "listen_inputs_norm_game_pg": round(float(g0["game_pg/listen"].norm()), 6),
        }
    return "gradients", res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="a communication checkpoint")
    src.add_argument("--branch", help="a silent checkpoint, branched as at the start of training (needs --config)")
    parser.add_argument("--config", help="communication config for --branch")
    parser.add_argument("--speaker", help="checkpoint whose frozen networks speak for everyone (listener stage)")
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--teacher", action="store_true", help="also play with every player speaking as the rule teacher")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    jobs = [(args, "talk"), (args, "babble"), (args, "silence")] + ([(args, "teacher")] if args.teacher else [])
    with ProcessPoolExecutor(max_workers=5) as ex:
        futures = [ex.submit(run_condition, j) for j in jobs] + [ex.submit(grad_signal, (args, None))]
        result = dict(f.result() for f in futures)
    result["source"] = args.checkpoint or f"branch of {args.branch} with {args.config}"
    result["speaker"] = args.speaker
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
