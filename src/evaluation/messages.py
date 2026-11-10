"""Message statistics from game records (RQ4): slot use, truthfulness, and how listeners propose and vote.

Works on the self-play records that `scripts/evaluate.py --save-games` writes (hidden roles included).
Everything here is observational: it says what messages contain and what goes with them, not what they
cause; causal effects come from test-time silencing (`scripts/evaluate.py --silence`).

Accusations are judgments that point to evil (evil, Morgana, Assassin); endorsements point to good. Their
precision is compared with guessing: among the five players a speaker can judge, two are evil for a good
speaker and one (its partner) for an evil speaker.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from ..env.rules import EVIL_ROLES, N_PLAYERS
from .metrics import _rate

EVIL_NAMES = frozenset(r.name for r in EVIL_ROLES)
ACCUSATIONS = frozenset({"evil", "morgana", "assassin"})
ROLE_JUDGMENTS = ("servant", "merlin", "percival", "morgana", "assassin")
BASES = ("none", "quest", "vote", "proposal", "role_knowledge")
CHANCE_EVIL = {"good": 2 / (N_PLAYERS - 1), "evil": 1 / (N_PLAYERS - 1)}  # P(a judged player is evil) by guessing


def _team(role: str) -> str:
    return "evil" if role in EVIL_NAMES else "good"


class _Counts:
    """k-of-n counters keyed by name."""

    def __init__(self) -> None:
        self.k: Counter = Counter()
        self.n: Counter = Counter()

    def add(self, key, hit: bool) -> None:
        self.n[key] += 1
        self.k[key] += bool(hit)

    def rates(self) -> dict:
        return {str(key): _rate(self.k[key], self.n[key]) for key in sorted(self.n, key=str)}


def message_stats(records: list[dict]) -> dict:
    """Statistics of the discussion messages and the evil talk in `records` (games with communication)."""
    records = [r for r in records if r.get("discussions") is not None]
    speaks, judges, basis_use = _Counts(), _Counts(), Counter()
    n_judgments: Counter = Counter()
    claims, teams, intents = _Counts(), _Counts(), _Counts()
    claim_values: dict[str, Counter] = defaultdict(Counter)
    accuse, endorse, role_hit, by_basis, by_quest = _Counts(), _Counts(), _Counts(), _Counts(), _Counts()
    include, approve = _Counts(), _Counts()
    n_messages = 0
    for rec in records:
        roles = rec["roles"]
        first = {p["quest"]: p for p in rec["proposals"] if p["attempt"] == 0}
        for disc in rec["discussions"]:
            accused: set[int] = set()
            for msg in disc["messages"]:
                n_messages += 1
                role = roles[msg["speaker"]]
                team = _team(role)
                judged = {int(seat): j for seat, j in msg.get("judgments", {}).items()}
                for key in (team, role):
                    speaks.add(key, bool(judged))
                    claims.add(key, "claim" in msg)
                    teams.add(key, "team" in msg)
                    intents.add(key, "intents" in msg)
                    n_judgments[key] += len(judged)
                if "claim" in msg:
                    claim_values[role][msg["claim"]] += 1
                for seat, j in judged.items():
                    target_evil = roles[seat] in EVIL_NAMES
                    basis = j.get("basis", "none")
                    basis_use[(team, basis)] += 1
                    judges.add(team, basis != "none")
                    if j["judgment"] in ACCUSATIONS:
                        accused.add(seat)
                        for key in (team, role):
                            accuse.add(key, target_evil)
                        by_basis.add(f"{team}/{basis}", target_evil)
                        by_quest.add(f"{team}/quest{disc['quest'] + 1}", target_evil)
                    else:
                        for key in (team, role):
                            endorse.add(key, not target_evil)
                    if j["judgment"] in ROLE_JUDGMENTS:
                        role_hit.add(f"{team}/{j['judgment']}", roles[seat] == j["judgment"].upper())
            prop = first.get(disc["quest"])
            if prop is None:  # the game ended during this discussion
                continue
            leader, team_seats = prop["leader"], set(prop["team"])
            for seat in range(N_PLAYERS):
                if seat != leader:
                    include.add(f"leader_{_team(roles[leader])}/{'accused' if seat in accused else 'not_accused'}",
                                seat in team_seats)
            for voter in range(N_PLAYERS):
                with_accused = bool((team_seats - {voter}) & accused)
                approve.add(f"voter_{_team(roles[voter])}/{'with_accused' if with_accused else 'without_accused'}",
                            voter in prop["approvals"])

    usage = {}
    for key in sorted(speaks.n):
        usage[key] = {
            "messages": speaks.n[key],
            "judges_someone": _rate(speaks.k[key], speaks.n[key]),
            "mean_judgments": n_judgments[key] / speaks.n[key],
            "claims": _rate(claims.k[key], claims.n[key]),
            "suggests_team": _rate(teams.k[key], teams.n[key]),
            "states_intent": _rate(intents.k[key], intents.n[key]),
        }
    basis = {team: {b: basis_use[(team, b)] / max(1, sum(basis_use[(team, x)] for x in BASES)) for b in BASES}
             for team in ("good", "evil")}
    return {
        "games": len(records),
        "messages": n_messages,
        "usage": usage,
        "claim_values_by_role": {role: dict(c) for role, c in sorted(claim_values.items())},
        "judgments_with_basis": judges.rates(),
        "basis_share": basis,
        "accusation_precision": accuse.rates(),
        "endorsement_precision": endorse.rates(),
        "chance_evil": CHANCE_EVIL,
        "role_judgment_precision": role_hit.rates(),
        "accusation_precision_by_basis": by_basis.rates(),
        "accusation_precision_by_quest": by_quest.rates(),
        "leader_includes": include.rates(),
        "first_proposal_approval": approve.rates(),
        "evil_talk": _evil_talk(records),
    }


def _evil_talk(records: list[dict]) -> dict:
    """Before the assassination: whom the evil players call Merlin, and whether the Assassin follows."""
    named, correct, followed, hit_followed, hit_other = _Counts(), _Counts(), _Counts(), _Counts(), _Counts()
    for rec in records:
        talk = rec.get("evil_talk") or []
        if not talk or rec.get("assassin_target") is None:
            continue
        roles, target = rec["roles"], rec["assassin_target"]
        merlin = roles.index("MERLIN")
        picks = []
        for msg in talk:
            role = roles[msg["speaker"]]
            seats = [int(s) for s, j in msg.get("judgments", {}).items() if j["judgment"] == "merlin"]
            named.add(role, bool(seats))
            if seats:
                correct.add(role, seats[0] == merlin)
                picks.append(seats[0])
        if picks:
            agree = target in picks
            followed.add("assassin_targets_a_named_merlin", agree)
            (hit_followed if agree else hit_other).add("hit", target == merlin)
    return {
        "names_a_merlin": named.rates(),
        "named_merlin_is_merlin": correct.rates(),
        "assassin_follows": followed.rates(),
        "hit_when_following": hit_followed.rates(),
        "hit_otherwise": hit_other.rates(),
    }
