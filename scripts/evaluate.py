"""Evaluate a trained checkpoint with frozen policies and paired role assignments.

    python scripts/evaluate.py --checkpoint runs/baseline_silent/seed0/latest.pt --games 1000

Matchups: self-play (S -> S), policy against rule-based players on each side, and the rule-vs-rule
and random-vs-random reference lines. Writes a JSON summary next to the checkpoint unless --out is
given; --save-games also dumps every self-play game record (hidden roles included) as JSON lines.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from avalon_rl.agents import PolicyController, RandomController, RuleController
from avalon_rl.evaluation import run_matchup
from avalon_rl.training.checkpoint import load_checkpoint, load_policies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261020, help="role-assignment seed shared by all matchups")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--greedy", action="store_true", help="take the most likely action instead of sampling")
    parser.add_argument("--out")
    parser.add_argument("--save-games", help="JSONL file for the self-play game records")
    args = parser.parse_args()

    nets = load_policies(args.checkpoint, args.device)
    policy = {t: PolicyController(nets[t], learn=False, device=args.device, greedy=args.greedy) for t in nets}
    opp = load_checkpoint(args.checkpoint)["config"]["opponents"]  # same rule player as in training
    rule = RuleController(opp["rule_fail_prob"], opp["rule_merlin_knowledge_prob"])
    rand = RandomController()
    matchups = {
        "selfplay": (policy["good"], policy["evil"]),
        "policy_good_vs_rule_evil": (policy["good"], rule),
        "rule_good_vs_policy_evil": (rule, policy["evil"]),
        "rule_vs_rule": (rule, rule),
        "random_vs_random": (rand, rand),
    }

    results = {"checkpoint": args.checkpoint, "games": args.games, "seed": args.seed, "greedy": args.greedy}
    for name, (good, evil) in matchups.items():
        keep = name == "selfplay" and args.save_games is not None
        summary, records = run_matchup(good, evil, args.games, args.seed, keep_records=keep)
        results[name] = summary
        win = summary["good_win"]
        print(f"{name:28s} good win {win['rate']:.3f}  95% CI [{win['ci95'][0]:.3f}, {win['ci95'][1]:.3f}]")
        if keep:
            with open(args.save_games, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in records)

    out = Path(args.out) if args.out else Path(args.checkpoint).with_suffix(".eval.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
