"""Flat action space shared by all phases.

Every action is expressed in the acting player's *relative* frame: seat 0 is the player itself,
seat r is the player r places to its left. This lets one network serve every seat.

    [0, 20)   propose team: index into TEAMS_BY_SIZE[k] (padded to the largest k)
    20, 21    vote approve / reject
    22, 23    quest card success / fail
    [24, 30)  assassinate the player at relative seat r
"""
from __future__ import annotations

from itertools import combinations

from .rules import N_PLAYERS, QUEST_SIZES

TEAMS_BY_SIZE: dict[int, tuple[tuple[int, ...], ...]] = {
    k: tuple(combinations(range(N_PLAYERS), k)) for k in sorted(set(QUEST_SIZES))
}
TEAM_INDEX: dict[int, dict[tuple[int, ...], int]] = {
    k: {team: i for i, team in enumerate(teams)} for k, teams in TEAMS_BY_SIZE.items()
}
MAX_TEAMS = max(len(t) for t in TEAMS_BY_SIZE.values())  # C(6, 3) = 20

TEAM_OFFSET = 0
VOTE_OFFSET = TEAM_OFFSET + MAX_TEAMS
QUEST_OFFSET = VOTE_OFFSET + 2
ASSASSIN_OFFSET = QUEST_OFFSET + 2
N_ACTIONS = ASSASSIN_OFFSET + N_PLAYERS

APPROVE = VOTE_OFFSET
REJECT = VOTE_OFFSET + 1
SUCCESS = QUEST_OFFSET
FAIL = QUEST_OFFSET + 1


def to_absolute(player: int, rel_seat: int) -> int:
    return (player + rel_seat) % N_PLAYERS


def to_relative(player: int, abs_seat: int) -> int:
    return (abs_seat - player) % N_PLAYERS


def team_action(player: int, abs_team) -> int:
    """Action index for proposing the given absolute seats."""
    rel = tuple(sorted(to_relative(player, s) for s in abs_team))
    return TEAM_OFFSET + TEAM_INDEX[len(rel)][rel]


def decode_team(player: int, action: int, size: int) -> tuple[int, ...]:
    """Absolute seats of the team chosen by `action`."""
    rel = TEAMS_BY_SIZE[size][action - TEAM_OFFSET]
    return tuple(sorted(to_absolute(player, r) for r in rel))


def assassin_action(player: int, abs_target: int) -> int:
    return ASSASSIN_OFFSET + to_relative(player, abs_target)
