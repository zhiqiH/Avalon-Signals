"""Evaluate a trained checkpoint with frozen policies and paired role assignments.

    python scripts/evaluate.py --checkpoint baseline/seed0/checkpoint_19532.pt --games 1000
    python scripts/evaluate.py --checkpoint runs/comm/talk/seed0/latest.pt --silence all        # nobody speaks
    python scripts/evaluate.py --checkpoint runs/comm/talk/seed0/latest.pt --silence rationale  # without slot (3)

Matchups: self-play, policy against rule-based players on each side, and the rule-vs-rule and
random-vs-random reference lines, all with the checkpoint's communication configuration. --silence
forces the listed slots (or all) to say nothing at test time. Writes a JSON summary next to the
checkpoint unless --out is given; --save-games also dumps every self-play game record (hidden roles
and messages included) as JSON lines. --partners adds self-play with the logic partner in one, two and all
three of the good team's Servant/Percival seats (the rule teacher speaks for them in a communication game).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root, so `import src` works

from src.agents import PolicyController, RandomController, RuleController
from src.evaluation import run_matchup
from src.comm import SLOTS
from src.env.game import AvalonGame
from src.training.checkpoint import comm_config, load_checkpoint, load_policies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261020, help="role-assignment seed shared by all matchups")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--greedy", action="store_true", help="take the most likely action instead of sampling")
    parser.add_argument("--silence", default="", help="comma-separated slots to force silent at test time, or 'all'")
    parser.add_argument("--out")
    parser.add_argument("--save-games", help="JSONL file for the self-play game records")
    parser.add_argument("--no-guard", action="store_true", help="lift the good players' vote guard (reference only)")
    parser.add_argument("--partners", action="store_true", help="also play with 1-3 logic partners in good seats")
    args = parser.parse_args()
    AvalonGame.good_guard = not args.no_guard

    nets = load_policies(args.checkpoint, args.device)
    policy = {t: PolicyController(nets[t], learn=False, device=args.device, greedy=args.greedy) for t in nets}
    config = load_checkpoint(args.checkpoint)["config"]
    opp = config["opponents"]  # same rule player as in training
    comm = comm_config(config)
    comm.silence = list(SLOTS) if args.silence == "all" else [s for s in args.silence.split(",") if s]
    comm.__post_init__()
    rule = RuleController(opp["rule_fail_prob"], opp["rule_merlin_knowledge_prob"])
    rand = RandomController()
    matchups = {
        "selfplay": (policy["good"], policy["evil"]),
        "policy_good_vs_rule_evil": (policy["good"], rule),
        "rule_good_vs_policy_evil": (rule, policy["evil"]),
        "rule_vs_rule": (rule, rule),
        "random_vs_random": (rand, rand),
    }

    results = {"checkpoint": args.checkpoint, "games": args.games, "seed": args.seed, "greedy": args.greedy,
               "good_guard": AvalonGame.good_guard,
               "comm": {s: comm.is_open(s) for s in SLOTS}, "silence": comm.silence}
    partners = {name: 0 for name in matchups}
    for k in (1, 2, 3) if args.partners else ():
        matchups[f"selfplay_with_{k}_logic_partners"] = matchups["selfplay"]
        partners[f"selfplay_with_{k}_logic_partners"] = k
    for name, (good, evil) in matchups.items():
        keep = name == "selfplay" and args.save_games is not None
        summary, records = run_matchup(good, evil, args.games, args.seed, keep_records=keep, comm=comm,
                                       partners=partners[name])
        results[name] = summary
        win = summary["good_win"]
        print(f"{name:28s} good win {win['rate']:.3f}  95% CI [{win['ci95'][0]:.3f}, {win['ci95'][1]:.3f}]")
        if keep:
            with open(args.save_games, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r) + "\n" for r in records)

    suffix = f".silence-{'-'.join(comm.silence)}" if comm.silence else ""
    out = Path(args.out) if args.out else Path(args.checkpoint).with_suffix(f"{suffix}.eval.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
