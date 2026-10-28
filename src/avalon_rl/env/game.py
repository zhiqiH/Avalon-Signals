"""Six-player Avalon state machine.

The game stores public history in absolute seats; `observation.encode` rotates it into each
player's relative frame. Hidden roles live in `roles` and must only be read by the referee,
by `private_knowledge`, and by offline analysis -- never by agents directly.

With an enabled `CommConfig`, every proposal is followed by a DISCUSS phase (one message per player,
leader first) before the vote, and three successes lead to EVIL_DISCUSS (Morgana, then the Assassin)
before the assassination; see `comm.protocol`. In those phases `step` takes messages instead of actions.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from ..comm import protocol as P  # module import: comm.protocol itself imports from this package
from . import actions as A
from .rules import (
    APPROVALS_NEEDED,
    DISCUSSION_PHASES,
    EVIL_ROLES,
    MAX_PROPOSALS,
    N_PLAYERS,
    N_QUESTS,
    QUEST_SIZES,
    ROLE_DECK,
    WINS_NEEDED,
    Phase,
    Role,
    WinReason,
)


class AvalonGame:
    def __init__(self, rng: np.random.Generator | None = None, comm: P.CommConfig | None = None):
        self.comm = comm if comm is not None else P.CommConfig()
        self.reset(rng if rng is not None else np.random.default_rng())

    def reset(self, rng: np.random.Generator) -> "AvalonGame":
        self.roles = np.array(rng.permutation(np.array(ROLE_DECK)), dtype=np.int8)
        self.leader = int(rng.integers(N_PLAYERS))
        self.quest = 0
        self.attempt = 0
        self.phase = Phase.PROPOSE
        self.team = np.zeros(N_PLAYERS, dtype=bool)  # team under vote or on the current quest
        self.successes = 0
        self.fails = 0
        self.good_won: bool | None = None
        self.win_reason: WinReason | None = None
        self.assassin_target = -1

        # Public history, indexed [quest, attempt] and absolute seats.
        self.prop_leader = np.full((N_QUESTS, MAX_PROPOSALS), -1, dtype=np.int8)
        self.prop_team = np.zeros((N_QUESTS, MAX_PROPOSALS, N_PLAYERS), dtype=bool)
        self.prop_voted = np.zeros((N_QUESTS, MAX_PROPOSALS), dtype=bool)
        self.prop_votes = np.zeros((N_QUESTS, MAX_PROPOSALS, N_PLAYERS), dtype=bool)  # True = approve
        self.prop_approved = np.zeros((N_QUESTS, MAX_PROPOSALS), dtype=bool)
        self.quest_result = np.full(N_QUESTS, -1, dtype=np.int8)  # -1 not played, 0 success, 1 fail
        self.quest_fails = np.zeros(N_QUESTS, dtype=np.int8)
        self.quest_team = np.zeros((N_QUESTS, N_PLAYERS), dtype=bool)

        # Private memory: each player's own quest cards, -1 none, 0 success, 1 fail.
        self.quest_cards = np.full((N_PLAYERS, N_QUESTS), -1, dtype=np.int8)

        # Discussion: speaking order of the current discussion and the public message log.
        self.speakers: list[int] = []
        self.speak_idx = 0
        if self.comm.enabled:
            self.msg_log = np.full((N_QUESTS, MAX_PROPOSALS, N_PLAYERS, P.N_MSG_HEADS), -1, dtype=np.int8)
            self.msg_feat = np.zeros((N_QUESTS, MAX_PROPOSALS, N_PLAYERS, P.MSG_FEAT_DIM), dtype=np.int8)
            self.evil_log = np.full((N_PLAYERS, P.N_MSG_HEADS), -1, dtype=np.int8)
            self.evil_feat = np.zeros((N_PLAYERS, N_PLAYERS, 2), dtype=np.int8)  # speaker x target x (Merlin, not)
        return self

    # ------------------------------------------------------------------ queries
    @property
    def done(self) -> bool:
        return self.phase == Phase.DONE

    @property
    def team_size(self) -> int:
        return QUEST_SIZES[self.quest]

    @property
    def in_discussion(self) -> bool:
        return self.phase in DISCUSSION_PHASES

    def is_evil_seat(self, seat: int) -> bool:
        return int(self.roles[seat]) in EVIL_ROLES

    def seat_of(self, role: Role) -> int:
        return int(np.flatnonzero(self.roles == role)[0])

    def acting_players(self) -> list[int]:
        if self.phase == Phase.PROPOSE:
            return [self.leader]
        if self.phase == Phase.VOTE:
            return list(range(N_PLAYERS))
        if self.phase == Phase.QUEST:
            return [int(s) for s in np.flatnonzero(self.team)]
        if self.phase == Phase.ASSASSINATE:
            return [self.seat_of(Role.ASSASSIN)]
        if self.in_discussion:
            return [self.speakers[self.speak_idx]]
        return []

    def legal_mask(self, player: int) -> np.ndarray:
        """Legal game actions (none during discussions, where players send messages instead)."""
        mask = np.zeros(A.N_ACTIONS, dtype=bool)
        if self.phase == Phase.PROPOSE and player == self.leader:
            mask[A.TEAM_OFFSET:A.TEAM_OFFSET + len(A.TEAMS_BY_SIZE[self.team_size])] = True
        elif self.phase == Phase.VOTE:
            mask[A.APPROVE] = mask[A.REJECT] = True
        elif self.phase == Phase.QUEST and self.team[player]:
            mask[A.SUCCESS] = True
            mask[A.FAIL] = self.is_evil_seat(player)  # good players can only succeed
        elif self.phase == Phase.ASSASSINATE and self.roles[player] == Role.ASSASSIN:
            for seat in range(N_PLAYERS):
                if not self.is_evil_seat(seat):
                    mask[A.assassin_action(player, seat)] = True
        return mask

    def private_knowledge(self, player: int) -> tuple[np.ndarray, np.ndarray]:
        """(known_evil, merlin_candidates) in absolute seats, as allowed by the player's role.

        Cached per player for the current `roles` array (callers must not modify the returned arrays).
        """
        if getattr(self, "_knowledge_roles", None) is not self.roles:
            self._knowledge_roles, self._knowledge = self.roles, {}
        if player not in self._knowledge:
            self._knowledge[player] = self._private_knowledge(player)
        return self._knowledge[player]

    def _private_knowledge(self, player: int) -> tuple[np.ndarray, np.ndarray]:
        known_evil = np.zeros(N_PLAYERS, dtype=bool)
        candidates = np.zeros(N_PLAYERS, dtype=bool)
        role = int(self.roles[player])
        evil = np.isin(self.roles, list(EVIL_ROLES))
        if role == Role.MERLIN or role in EVIL_ROLES:
            known_evil = evil
        elif role == Role.PERCIVAL:
            # An unordered pair: the encoding never reveals which candidate is Merlin.
            candidates = (self.roles == Role.MERLIN) | (self.roles == Role.MORGANA)
        return known_evil, candidates

    # ------------------------------------------------------------------ transitions
    def step(self, actions: Mapping[int, int | np.ndarray], message_masks: tuple | None = None) -> None:
        """Apply the simultaneous actions of all `acting_players()` (in a discussion: the speaker's message).

        `message_masks` may pass the speaker's `comm.protocol.message_mask` if the caller already computed it.
        """
        acting = self.acting_players()
        if sorted(actions) != acting:
            raise ValueError(f"expected actions for {acting}, got {sorted(actions)}")
        if self.in_discussion:
            speaker, msg = next(iter(actions.items()))
            P.check_message(self, speaker, msg, message_masks)
            self._speak(speaker, np.asarray(msg, dtype=np.int8))
            return
        for p, a in actions.items():
            if not self.legal_mask(p)[a]:
                raise ValueError(f"illegal action {a} for player {p} in phase {self.phase.name}")

        if self.phase == Phase.PROPOSE:
            self._propose(actions[self.leader])
        elif self.phase == Phase.VOTE:
            self._vote(actions)
        elif self.phase == Phase.QUEST:
            self._run_quest(actions)
        elif self.phase == Phase.ASSASSINATE:
            self._assassinate(next(iter(actions.items())))

    def _propose(self, action: int) -> None:
        q, a = self.quest, self.attempt
        self.team[:] = False
        self.team[list(A.decode_team(self.leader, action, self.team_size))] = True
        self.prop_leader[q, a] = self.leader
        self.prop_team[q, a] = self.team
        if self.comm.enabled:
            self._start_discussion(Phase.DISCUSS, [(self.leader + k) % N_PLAYERS for k in range(N_PLAYERS)])
        else:
            self.phase = Phase.VOTE

    def _start_discussion(self, phase: Phase, speakers: list[int]) -> None:
        self.phase = phase
        self.speakers = speakers
        self.speak_idx = 0

    def _speak(self, speaker: int, msg: np.ndarray) -> None:
        if self.phase == Phase.DISCUSS:
            q, a = self.quest, self.attempt
            self.msg_log[q, a, speaker] = msg
            self.msg_feat[q, a, speaker] = P.message_features(self, speaker, msg)
        else:
            self.evil_log[speaker] = msg
            for r in range(1, N_PLAYERS):
                v = msg[P.H_ALIGN + r - 1]
                if v > 0:
                    self.evil_feat[speaker, (speaker + r) % N_PLAYERS, v - 1] = 1
        self.speak_idx += 1
        if self.speak_idx == len(self.speakers):
            self.phase = Phase.VOTE if self.phase == Phase.DISCUSS else Phase.ASSASSINATE
            self.speakers, self.speak_idx = [], 0

    def _vote(self, actions: Mapping[int, int]) -> None:
        q, a = self.quest, self.attempt
        approve = np.array([actions[p] == A.APPROVE for p in range(N_PLAYERS)])
        self.prop_voted[q, a] = True
        self.prop_votes[q, a] = approve
        approved = int(approve.sum()) >= APPROVALS_NEEDED
        self.prop_approved[q, a] = approved
        self.leader = (self.leader + 1) % N_PLAYERS  # leadership passes after every vote
        if approved:
            self.quest_team[q] = self.team
            self.phase = Phase.QUEST
        else:
            self.attempt += 1
            self.team[:] = False
            if self.attempt == MAX_PROPOSALS:
                self._finish(good_won=False, reason=WinReason.FIVE_REJECTIONS)
            else:
                self.phase = Phase.PROPOSE

    def _run_quest(self, actions: Mapping[int, int]) -> None:
        q = self.quest
        n_fail = 0
        for p, act in actions.items():
            failed = act == A.FAIL
            self.quest_cards[p, q] = int(failed)
            n_fail += failed
        self.quest_fails[q] = n_fail
        self.quest_result[q] = int(n_fail > 0)
        if n_fail > 0:
            self.fails += 1
        else:
            self.successes += 1
        self.quest += 1
        self.attempt = 0
        self.team[:] = False
        if self.fails == WINS_NEEDED:
            self._finish(good_won=False, reason=WinReason.THREE_FAILS)
        elif self.successes == WINS_NEEDED:
            if self.comm.is_open("alignment"):
                self._start_discussion(Phase.EVIL_DISCUSS, [self.seat_of(Role.MORGANA), self.seat_of(Role.ASSASSIN)])
            else:
                self.phase = Phase.ASSASSINATE
        else:
            self.phase = Phase.PROPOSE

    def _assassinate(self, player_action: tuple[int, int]) -> None:
        player, act = player_action
        target = A.to_absolute(player, act - A.ASSASSIN_OFFSET)
        self.assassin_target = target
        if self.roles[target] == Role.MERLIN:
            self._finish(good_won=False, reason=WinReason.ASSASSIN_HIT)
        else:
            self._finish(good_won=True, reason=WinReason.ASSASSIN_MISSED)

    def _finish(self, good_won: bool, reason: WinReason) -> None:
        self.good_won = good_won
        self.win_reason = reason
        self.phase = Phase.DONE

    # ------------------------------------------------------------------ logging
    def to_record(self) -> dict:
        """Full game record for offline analysis (includes hidden roles)."""
        n_quests_played = int((self.quest_result >= 0).sum())
        proposals = []
        for q in range(N_QUESTS):
            for a in range(MAX_PROPOSALS):
                if self.prop_leader[q, a] < 0:
                    continue
                proposals.append({
                    "quest": q,
                    "attempt": a,
                    "leader": int(self.prop_leader[q, a]),
                    "team": [int(s) for s in np.flatnonzero(self.prop_team[q, a])],
                    "approvals": [int(s) for s in np.flatnonzero(self.prop_votes[q, a])],
                    "approved": bool(self.prop_approved[q, a]),
                })
        record = {
            "roles": [Role(int(r)).name for r in self.roles],
            "good_won": self.good_won,
            "win_reason": self.win_reason.name if self.win_reason is not None else None,
            "quest_results": [int(r) for r in self.quest_result[:n_quests_played]],
            "quest_fails": [int(f) for f in self.quest_fails[:n_quests_played]],
            "quest_cards": self.quest_cards.tolist(),
            "proposals": proposals,
            "assassin_target": self.assassin_target,
        }
        if self.comm.enabled:
            record["comm"] = {s: self.comm.is_open(s) for s in ("claim", "alignment", "rationale", "team", "intent")}
            for p in proposals:
                q, a = p["quest"], p["attempt"]
                p["messages"] = [P.decode_message(self.msg_log[q, a, s], s, QUEST_SIZES[q])
                                 for s in [(p["leader"] + k) % N_PLAYERS for k in range(N_PLAYERS)]
                                 if self.msg_log[q, a, s, 0] >= 0]
            record["evil_talk"] = [P.decode_message(self.evil_log[s], s, QUEST_SIZES[0], evil_talk=True)
                                   for s in range(N_PLAYERS) if self.evil_log[s, 0] >= 0]
        return record
