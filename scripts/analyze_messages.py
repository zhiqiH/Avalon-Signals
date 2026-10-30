"""Message statistics (RQ4) from saved self-play game records, per file and pooled.

    python scripts/evaluate.py --checkpoint runs/comm_M/seed0/latest.pt --save-games runs/comm_M/seed0/games.jsonl
    python scripts/analyze_messages.py runs/comm_M/seed*/games.jsonl --out results/tables/comm_M_messages.json

Reports slot use by team and role, the precision of accusations and endorsements against guessing, the
share and precision of each rationale basis, whether leaders include accused players and voters approve
teams with accused players, and how the evil talk relates to the assassination. Observational only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from avalon_rl.evaluation.messages import message_stats


def _fmt(r: dict | None) -> str:
    return "   -   " if not r or not r["n"] else f"{r['rate']:.3f} (n={r['n']})"


def report(name: str, s: dict) -> None:
    print(f"== {name}: {s['games']} games, {s['messages']} messages")
    for team in ("good", "evil"):
        u = s["usage"].get(team)
        if u:
            print(f"  {team:5s} speakers judge someone {_fmt(u['judges_someone'])}, {u['mean_judgments']:.2f} judgments per message")
    for key in ("good", "evil", "MERLIN", "PERCIVAL", "SERVANT", "MORGANA", "ASSASSIN"):
        a, e = s["accusation_precision"].get(key), s["endorsement_precision"].get(key)
        if a or e:
            print(f"  {key:9s} accusations hit evil {_fmt(a)}   endorsements hit good {_fmt(e)}")
    print(f"  chance that a judged player is evil: good speaker {s['chance_evil']['good']:.2f}, evil speaker {s['chance_evil']['evil']:.2f}")
    for team, shares in s["basis_share"].items():
        print(f"  {team:5s} basis share " + "  ".join(f"{b} {v:.2f}" for b, v in shares.items()))
    for key, r in s["leader_includes"].items():
        print(f"  {key:30s} included {_fmt(r)}")
    for key, r in s["first_proposal_approval"].items():
        print(f"  {key:30s} approves {_fmt(r)}")
    talk = s["evil_talk"]
    for key, r in talk["assassin_follows"].items():
        print(f"  evil talk: {key} {_fmt(r)}; hit when following {_fmt(talk['hit_when_following'].get('hit'))}, "
              f"otherwise {_fmt(talk['hit_otherwise'].get('hit'))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("games", nargs="+", help="JSONL game records, e.g. one file per seed")
    parser.add_argument("--out")
    args = parser.parse_args()

    files = {path: [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()] for path in args.games}
    results = {path: message_stats(records) for path, records in files.items()}
    if len(files) > 1:
        results["pooled"] = message_stats([r for records in files.values() for r in records])
    for name, stats in results.items():
        report(name, stats)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
