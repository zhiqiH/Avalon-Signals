"""Play games with one checkpoint and write a full trajectory of one of them, with what lies behind each decision.

    python scripts/trace_game.py --checkpoint runs/comm_F/stage3b_F/seed0/checkpoint_01000.pt --out trace.md

Every seat is played by the checkpoint's own networks (self-play). For each decision the trace keeps
- what the player knows: its role, its private knowledge, and what exact deduction leaves it (the evil pairs
  still possible for it, and the players that are therefore certainly evil or certainly good);
- what it says: the message, each judgment with the record it rests on;
- what it does: the policy's probabilities (its three preferred teams, its chance to approve, to fail a quest,
  whom to stab), next to the same probabilities without the binding of words, and whether its own words or
  the vote guard forced the action (a leader proposes one of the cleanest teams, the Assassin stabs whom it named).
The game written out is the one with the richest talk among `--games` (or `--game i`). `--json` also saves
the raw trace. The text is in Chinese.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root, so `import src` works

from src.agents.base import Request  # noqa: E402
from src.agents.policy import PolicyController, masked_logits  # noqa: E402
from src.comm import protocol as P  # noqa: E402
from src.env import actions as A  # noqa: E402
from src.env.game import AvalonGame  # noqa: E402
from src.env.observation import EVIL_PAIRS, _certain, possible_evil_pairs  # noqa: E402
from src.env.rules import MAX_PROPOSALS, QUEST_SIZES, WINS_NEEDED, Phase, Role  # noqa: E402
from src.training.checkpoint import comm_config, load_checkpoint, load_policies  # noqa: E402


# ---------------------------------------------------------------------- playing with a trace
def knowledge(game: AvalonGame, player: int) -> dict:
    known_evil, candidates = game.private_knowledge(player)
    possible = possible_evil_pairs(game, player)
    evil, good = _certain(possible)
    return {"role": Role(int(game.roles[player])).name,
            "known_evil": [int(s) for s in np.flatnonzero(known_evil) if s != player],
            "merlin_candidates": [int(s) for s in np.flatnonzero(candidates)],
            "possible_pairs": [list(EVIL_PAIRS[i]) for i in np.flatnonzero(possible)],
            "certain_evil": [int(s) for s in np.flatnonzero(evil) if s != player],
            "certain_good": [int(s) for s in np.flatnonzero(good) if s != player]}


@torch.no_grad()
def action_probs(ctrl: PolicyController, game: AvalonGame, player: int, mask: np.ndarray) -> np.ndarray:
    _, _, t_obs, t_comm = ctrl._inputs([Request(0, game, player, mask)])
    logits = ctrl.net.logits(t_obs, t_comm)[:, :A.N_ACTIONS]
    return torch.softmax(masked_logits(logits, torch.from_numpy(mask)[None]), -1)[0].numpy()


def free_mask(game: AvalonGame, player: int) -> np.ndarray:
    """The legal actions if words did not bind."""
    game.word_binding = False
    try:
        return game.legal_mask(player)
    finally:
        del game.word_binding  # back to the class attribute


def decide(ctrl: PolicyController, game: AvalonGame, player: int, rng: np.random.Generator) -> tuple[int, dict]:
    mask, free = game.legal_mask(player), free_mask(game, player)
    probs, probs_free = action_probs(ctrl, game, player, mask), action_probs(ctrl, game, player, free)
    action = int(rng.choice(len(probs), p=probs / probs.sum()))
    info: dict = {"forced": int(mask.sum()) == 1}
    if game.phase == Phase.PROPOSE:
        teams = lambda p: [{"team": list(A.decode_team(player, int(a), game.team_size)), "p": round(float(p[a]), 3)}  # noqa: E731
                           for a in np.argsort(-p)[:3] if p[a] > 0]
        info.update(team=list(A.decode_team(player, action, game.team_size)), top=teams(probs), top_free=teams(probs_free),
                    allowed=int(mask.sum()), of=int(free.sum()))
    elif game.phase == Phase.VOTE:
        word = P.word_vote(game, player) if game.comm.enabled else None
        guard = None
        if not game.is_evil_seat(player) and game.attempt == MAX_PROPOSALS - 1:
            guard = "fifth"
        elif not game.is_evil_seat(player) and game.fails == WINS_NEEDED - 1 and game.certainly_dirty(player):
            guard = "dirty"
        banned, required = P.word_bounds(game, player) if game.comm.enabled else (np.zeros(6, bool), np.zeros(6, bool))
        info.update(approve=action == A.APPROVE, p_approve=round(float(probs[A.APPROVE]), 3),
                    p_approve_free=round(float(probs_free[A.APPROVE]), 3), word=word, guard=guard,
                    banned_on_team=[int(s) for s in np.flatnonzero(banned & game.team)],
                    required_missing=[int(s) for s in np.flatnonzero(required & ~game.team)],
                    own_suggestion=bool(game.comm.enabled and P.is_own_suggestion(game, player)))
    elif game.phase == Phase.QUEST:
        info.update(fail=action == A.FAIL, p_fail=round(float(probs[A.FAIL]), 3))
    else:
        info.update(target=A.to_absolute(player, action - A.ASSASSIN_OFFSET),
                    targets={int(A.to_absolute(player, a - A.ASSASSIN_OFFSET)): round(float(probs_free[a]), 3)
                             for a in np.flatnonzero(free)})
    return action, info


def play_traced(game: AvalonGame, ctrls: dict[str, PolicyController], rng: np.random.Generator) -> list[dict]:
    steps = []
    while not game.done:
        step = {"phase": game.phase.name, "quest": game.quest, "attempt": game.attempt, "leader": game.leader,
                "team": [int(s) for s in np.flatnonzero(game.team)], "decisions": []}
        chosen: dict = {}
        for p in game.acting_players():
            ctrl = ctrls["evil" if game.is_evil_seat(p) else "good"]
            entry = {"player": p, **knowledge(game, p)}
            if game.in_discussion:
                ctx = P.message_context(game, p)
                msg = P.forced_message(ctx)
                if msg is None:
                    msg = ctrl.speak([Request(0, game, p, None, ctx)], rng).messages[0]
                size = QUEST_SIZES[0] if game.phase == Phase.EVIL_DISCUSS else game.team_size
                entry["message"] = P.decode_message(msg, p, size)
                chosen[p] = msg
            else:
                chosen[p], info = decide(ctrl, game, p, rng)
                entry.update(info)
            step["decisions"].append(entry)
        game.step(chosen)
        steps.append(step)
    return steps


def richness(record: dict) -> float:
    msgs = [m for d in record.get("discussions", []) for m in d["messages"]]
    accusing = sum(any(j["judgment"] == "evil" for j in m.get("judgments", {}).values()) for m in msgs)
    return (2 * min(accusing, 4) + min(sum("intents" in m for m in msgs), 3) + min(sum("team" in m for m in msgs), 6) / 2
            + 2 * any(m.get("claim") == "percival" for m in msgs) + len(record["quest_results"])
            + 3 * (record["win_reason"] in ("ASSASSIN_MISSED", "ASSASSIN_HIT")))


# ---------------------------------------------------------------------- text (Chinese)
ROLE = {"MERLIN": "梅林", "MORGANA": "莫甘娜", "SERVANT": "忠臣", "PERCIVAL": "派西维尔", "ASSASSIN": "刺客"}
CLAIM = {"servant": "忠臣", "percival": "派西维尔", "merlin": "梅林"}
JUDGE = {"good": "好人", "evil": "坏人", "servant": "忠臣", "merlin": "梅林", "percival": "派西维尔"}
BASIS = {"quest": "任务", "vote": "投票", "proposal": "组队"}
DISPUTE = {"defense": "他指认了我", "judgment": "我们对同一个人的判断相反", "team": "他推荐的队里有我指认的人", None: "无依据"}
RESULT = {"ASSASSIN_MISSED": "三次任务成功，刺客没有刺中梅林", "ASSASSIN_HIT": "三次任务成功，但刺客刺中了梅林",
          "THREE_FAILS": "三次任务失败", "FIVE_REJECTIONS": "连续五次否决"}


def name(seat: int) -> str:
    return f"{seat + 1}号"


def names(seats) -> str:
    return "、".join(name(int(s)) for s in seats) if len(seats) else "无"


def knows(d: dict) -> str:
    """What the player knows in private and what deduction leaves it."""
    role = d["role"]
    if role in ("MORGANA", "ASSASSIN"):
        return f"知道队友是 {names(d['known_evil'])}"
    if role == "MERLIN":
        return f"看到坏人是 {names(d['known_evil'])}"
    parts = []
    if role == "PERCIVAL":
        parts.append(f"看到 {names(d['merlin_candidates'])} 一个是梅林一个是莫甘娜")
    parts.append(f"按任务结果还剩 {len(d['possible_pairs'])} 种坏人组合")
    if d["certain_evil"]:
        parts.append(f"能确定 {names(d['certain_evil'])} 是坏人")
    if d["certain_good"]:
        parts.append(f"能确定 {names(d['certain_good'])} 是好人")
    return "；".join(parts)


def says(m: dict, roles: list[str], evil: list[bool]) -> str:
    parts = []
    if "claim" in m:
        parts.append(f"“我是{CLAIM[m['claim']]}”" + ("" if m["claim"].upper() == roles[m["speaker"]] else "（假话）"))
    for seat, e in m.get("judgments", {}).items():
        seat = int(seat)
        if e["judgment"] in ("good", "evil"):
            right = "说对了" if (e["judgment"] == "evil") == evil[seat] else "说错了"
        else:
            right = "说对了" if e["judgment"].upper() == roles[seat] else "说错了"
        basis = f"依据第{e['quest']}轮的{BASIS[e['basis']]}记录，" if e.get("basis") in BASIS else ""
        parts.append(f"“{name(seat)}是{JUDGE[e['judgment']]}”（{basis}{right}）")
    if "team" in m:
        parts.append(f"推荐 [{names(m['team'])}]")
    for seat, intent in m.get("intents", {}).items():
        parts.append(f"“{'没有' if intent == 'require' else '有'}{name(int(seat))}的队我不赞成”")
    for seat, e in m.get("replies", {}).items():
        if e["reply"] == "endorse":
            parts.append(f"“我同意{name(int(seat))}，他推荐的队我赞成”")
        else:
            parts.append(f"“我不同意{name(int(seat))}（{DISPUTE[e.get('basis')]}），他推荐的队我不赞成”")
    for seat, kind in m.get("promises", {}).items():
        outcome, call = ("失败", "坏人") if kind == "fail_evil" else ("成功", "好人")
        parts.append(f"“{name(int(seat))}若上这轮任务且任务{outcome}，下一轮我说他是{call}”")
    return "；".join(parts) if parts else "不发言"


def vote_line(d: dict, roles: list[str]) -> str:
    who = f"{name(d['player'])}{ROLE[roles[d['player']]]}"
    act = "赞成" if d["approve"] else "反对"
    if d["guard"] == "fifth":
        why = "第五次提名，好人必须赞成"
    elif d["guard"] == "dirty":
        why = "已两次失败，按他的推理这支队必有坏人，必须反对"
    elif not d["forced"]:
        why = f"自己决定，赞成概率 {d['p_approve']:.0%}"
    elif d["word"] is None:
        why = "只有这一种选择"
    elif d["banned_on_team"]:
        why = f"他指控或否决过 {names(d['banned_on_team'])}，必须反对"
    elif d["required_missing"]:
        why = f"他说过没有 {names(d['required_missing'])} 的队不赞成，必须反对"
    elif d["own_suggestion"]:
        why = "这正是他推荐的队，必须赞成"
    elif d["word"] is True and "leader" in d:
        why = "自己的提名，必须赞成"
    else:
        why = "本轮他把队里其他人都说成好人，必须赞成"
    if d["forced"] and d["guard"] is None:
        why += f"（不受约束时赞成概率 {d['p_approve_free']:.0%}）"
    return f"    - {who} **{act}**：{why}"


def render(record: dict, steps: list[dict], header: str) -> str:
    roles = record["roles"]
    evil = [r in ("MORGANA", "ASSASSIN") for r in roles]
    who = lambda s: f"{name(s)}{ROLE[roles[s]]}"  # noqa: E731
    out = [f"# 对局轨迹：{header}", "", "座位：" + "，".join(who(s) for s in range(6)), "",
           "每条发言前是这名玩家当时私下知道的事和逻辑推理能确定的事；每次行动后是策略给出的概率，"
           "以及这个行动是不是被他自己说过的话（或投票保护）定下来的。", ""]
    score = [0, 0]
    shown_quest = -1
    for step in steps:
        phase, q = step["phase"], step["quest"]
        if phase in ("DISCUSS", "PROPOSE") and q != shown_quest:
            shown_quest = q
            out += ["", f"## 第 {q + 1} 轮（{QUEST_SIZES[q]} 人队，比分 好人 {score[0]} : {score[1]} 坏人）", ""]
        if phase == "DISCUSS":
            d = step["decisions"][0]
            tag = "（队长）" if d["player"] == step["leader"] else ""
            out += [f"- **{who(d['player'])}{tag}**", f"    - 知道：{knows(d)}", f"    - 说：{says(d['message'], roles, evil)}"]
        elif phase == "PROPOSE":
            d = step["decisions"][0]
            why = "策略的前三选择：" + "，".join(f"[{names(t['team'])}] {t['p']:.0%}" for t in d["top"])
            if d["allowed"] != d["of"]:
                why += f"；按公开记录和他自己的话，{d['of']} 支队里只有 {d['allowed']} 支可提"
                if d["top_free"] and d["top_free"][0]["team"] != d["top"][0]["team"]:
                    why += f"，不受约束时首选 [{names(d['top_free'][0]['team'])}]"
            out += [f"- **提名 {step['attempt'] + 1}**：队长 {who(d['player'])} 提 **[{names(d['team'])}]**", f"    - {why}"]
        elif phase == "VOTE":
            for d in step["decisions"]:
                if d["player"] == step["leader"]:
                    d["leader"] = True
                out.append(vote_line(d, roles))
            approvals = sum(d["approve"] for d in step["decisions"])
            out.append(f"    - 结果：{approvals} 票赞成，{'通过' if approvals >= 4 else '否决'}")
        elif phase == "QUEST":
            cards = []
            for d in step["decisions"]:
                if evil[d["player"]]:
                    cards.append(f"{who(d['player'])} 出**{'失败' if d['fail'] else '成功'}**（出失败的概率 {d['p_fail']:.0%}）")
                else:
                    cards.append(f"{who(d['player'])} 出成功")
            failed = any(d.get("fail") for d in step["decisions"])
            score[1 if failed else 0] += 1
            out += [f"- **任务{'失败' if failed else '成功'}**：" + "；".join(cards)]
        elif phase == "EVIL_DISCUSS":
            if step["decisions"][0]["player"] == [s for s in range(6) if roles[s] == "MORGANA"][0]:
                out += ["", "## 刺杀前坏人商议", ""]
            d = step["decisions"][0]
            out.append(f"- **{who(d['player'])}** 说：{says(d['message'], roles, evil)}")
        elif phase == "ASSASSINATE":
            d = step["decisions"][0]
            dist = "，".join(f"{who(s)} {p:.0%}" for s, p in sorted(d["targets"].items(), key=lambda kv: -kv[1]))
            why = f"他商议时点名 {name(d['target'])} 是梅林，必须刺他；不受约束时" if d["forced"] else ""
            out += ["", f"- **刺杀**：{who(d['player'])} 刺 **{who(d['target'])}**", f"    - {why}策略对各目标的概率：{dist}"]
    out += ["", "## 结局", "", f"**{'好人胜' if record['good_won'] else '坏人胜'}**：{RESULT[record['win_reason']]}。"]
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=200, help="games to play; the one with the richest talk is written out")
    parser.add_argument("--game", type=int, help="write out this game instead")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="Markdown file for the trajectory")
    parser.add_argument("--json", help="also save the raw trace here")
    args = parser.parse_args()
    torch.set_num_threads(4)
    comm = comm_config(load_checkpoint(args.checkpoint)["config"])
    ctrls = {t: PolicyController(net, learn=False) for t, net in load_policies(args.checkpoint).items()}
    rng = np.random.default_rng([args.seed, 5])
    played = []
    for i in range(args.games):
        game = AvalonGame(np.random.default_rng([args.seed, i]), comm=comm)
        steps = play_traced(game, ctrls, rng)
        played.append((game.to_record(), steps))
    best = args.game if args.game is not None else int(np.argmax([richness(r) for r, _ in played]))
    record, steps = played[best]
    wins = np.mean([r["good_won"] for r, _ in played])
    header = f"{Path(args.checkpoint).parent.name}，第 {best} 局（共 {args.games} 局，好人胜率 {wins:.2f}）"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(render(record, steps, header), encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps({"record": record, "steps": steps}, ensure_ascii=False, default=bool), encoding="utf-8")
    print(f"wrote game {best} of {args.games} to {args.out} (good win rate over these games {wins:.3f})")


if __name__ == "__main__":
    main()
