"""Fixed rules of the six-player game studied in this project.

Roles: two Loyal Servants, Merlin and Percival (good); Morgana and the Assassin (evil).
No Lady of the Lake, no Mordred/Oberon, no two-fail rule (it only applies to 7+ players).
"""
from __future__ import annotations

from enum import IntEnum

N_PLAYERS = 6
QUEST_SIZES = (2, 3, 4, 3, 4)
N_QUESTS = len(QUEST_SIZES)
MAX_PROPOSALS = 5  # the fifth consecutive rejection within one quest gives evil the win
APPROVALS_NEEDED = 4  # strict majority of six players; a 3-3 tie rejects
WINS_NEEDED = 3


class Role(IntEnum):
    SERVANT = 0
    MERLIN = 1
    PERCIVAL = 2
    MORGANA = 3
    ASSASSIN = 4


N_ROLES = len(Role)
ROLE_DECK = (Role.SERVANT, Role.SERVANT, Role.MERLIN, Role.PERCIVAL, Role.MORGANA, Role.ASSASSIN)
EVIL_ROLES = frozenset({Role.MORGANA, Role.ASSASSIN})


def is_evil(role: int) -> bool:
    return role in EVIL_ROLES


class Phase(IntEnum):
    PROPOSE = 0
    VOTE = 1
    QUEST = 2
    ASSASSINATE = 3
    DONE = 4


N_DECISION_PHASES = 4  # PROPOSE, VOTE, QUEST, ASSASSINATE


class WinReason(IntEnum):
    ASSASSIN_MISSED = 0  # good wins
    ASSASSIN_HIT = 1  # evil wins
    THREE_FAILS = 2  # evil wins
    FIVE_REJECTIONS = 3  # evil wins
