"""Game outcomes and the RQ1 summary statistics.

P(good wins) = P(three successes) * (1 - P(assassin hits Merlin | three successes)); the
conditional rate uses only games that reached the assassination, and per-quest success rates use
only quests that were actually played.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ..env.game import AvalonGame
from ..env.rules import N_QUESTS, WinReason


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def game_outcome(game: AvalonGame) -> dict:
    played = game.quest_result >= 0
    return {
        "good_won": bool(game.good_won),
        "reason": game.win_reason.name,
        "quest_results": [int(r) for r in game.quest_result[played]],
        "n_proposals": int((game.prop_leader >= 0).sum()),
    }


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson_interval(k, n)
    return {"rate": k / n if n else float("nan"), "k": k, "n": n, "ci95": [lo, hi]}


def summarize(outcomes: list[dict]) -> dict:
    n = len(outcomes)
    reasons = Counter(o["reason"] for o in outcomes)
    good_wins = sum(o["good_won"] for o in outcomes)
    reached = reasons[WinReason.ASSASSIN_HIT.name] + reasons[WinReason.ASSASSIN_MISSED.name]
    quest_success = []
    for q in range(N_QUESTS):
        results = [o["quest_results"][q] for o in outcomes if len(o["quest_results"]) > q]
        quest_success.append(_rate(sum(r == 0 for r in results), len(results)))
    return {
        "games": n,
        "good_win": _rate(good_wins, n),
        "three_successes": _rate(reached, n),
        "assassin_hit_given_three_successes": _rate(reasons[WinReason.ASSASSIN_HIT.name], reached),
        "evil_win_three_fails": _rate(reasons[WinReason.THREE_FAILS.name], n),
        "evil_win_five_rejections": _rate(reasons[WinReason.FIVE_REJECTIONS.name], n),
        "quest_success_by_index": quest_success,
        "mean_proposals": float(np.mean([o["n_proposals"] for o in outcomes])) if n else float("nan"),
    }
