"""Hand-written heuristic players.

They read only what the acting player may legally know: the public history stored on the game
and `game.private_knowledge(player)` (plus the player's own role). They exist to check that
learning improves on simple play and to serve as fixed opponents; they are not meant to be strong.

Good players: suspect whoever sat on failed quests or approved them, and whoever rejected a team
that went on to succeed; propose themselves plus
the least suspicious players; approve a team unless it contains one of the two prime suspects,
and always approve the fifth proposal of a round. Merlin uses his exact knowledge only with
probability `merlin_knowledge_prob` per decision and otherwise plays like a servant, since
always acting on it would reveal him to the Assassin.
Evil players: propose themselves plus innocent-looking good players; approve teams with an evil
member; reject the fifth proposal unless it contains evil; fail quests with probability
`fail_prob`, only one of them failing when both are on the team.
Assassin: targets the good player whose votes best tracked which teams contained evil.
In discussions, rule players say nothing and ignore messages.
"""
from __future__ import annotations

import numpy as np

from ..comm.protocol import sample_messages, stack_contexts
from ..env import actions as A
from ..env.game import AvalonGame
from ..env.rules import MAX_PROPOSALS, N_PLAYERS, N_QUESTS, Phase, Role
from .base import ActResult, Request


def public_suspicion(game: AvalonGame) -> np.ndarray:
    """Suspicion per absolute seat computed from public quest and vote records only."""
    s = np.zeros(N_PLAYERS)
    for q in range(N_QUESTS):
        if game.quest_result[q] < 0:
            continue
        team = game.quest_team[q]
        approved_attempt = int(np.flatnonzero(game.prop_approved[q])[0])
        approvers = game.prop_votes[q, approved_attempt]
        if game.quest_result[q] == 1:
            s[team] += 2.0 * game.quest_fails[q] / team.sum()
            s[approvers & ~team] += 0.3
        else:
            s[team] -= 0.5
            s[~approvers] += 0.5  # rejecting a team that turned out clean
    return s


class RuleController:
    learn = False

    def __init__(self, fail_prob: float = 0.9, merlin_knowledge_prob: float = 0.5):
        self.fail_prob = fail_prob
        self.merlin_knowledge_prob = merlin_knowledge_prob

    def act(self, requests: list[Request], rng: np.random.Generator) -> ActResult:
        return ActResult(np.array([self._act(r.game, r.player, rng) for r in requests], dtype=np.int64))

    def speak(self, requests: list[Request], rng: np.random.Generator) -> ActResult:
        return ActResult(messages=sample_messages(stack_contexts([r.ctx for r in requests]), rng, first=True))

    # ------------------------------------------------------------------ dispatch
    def _act(self, game: AvalonGame, player: int, rng: np.random.Generator) -> int:
        if game.phase == Phase.PROPOSE:
            return self._propose(game, player, rng)
        if game.phase == Phase.VOTE:
            return self._vote(game, player, rng)
        if game.phase == Phase.QUEST:
            return self._quest(game, player, rng)
        if game.phase == Phase.ASSASSINATE:
            return self._assassinate(game, player, rng)
        raise ValueError(f"no decision in phase {game.phase}")

    # ------------------------------------------------------------------ beliefs
    def _merlin_uses_knowledge(self, game: AvalonGame, player: int, rng: np.random.Generator) -> bool:
        # A player always knows its own role.
        return game.roles[player] == Role.MERLIN and rng.random() < self.merlin_knowledge_prob

    def _suspicion(self, game: AvalonGame, player: int, rng: np.random.Generator, use_knowledge: bool) -> np.ndarray:
        role = int(game.roles[player])
        known_evil, candidates = game.private_knowledge(player)
        if use_knowledge:
            s = np.where(known_evil, 10.0, -1.0)
        else:
            s = public_suspicion(game)
            if role == Role.PERCIVAL:
                s = s - 0.5 * candidates
        s = s + 1e-3 * rng.random(N_PLAYERS)  # random tie-breaking
        s[player] = -np.inf
        return s

    @staticmethod
    def _is_evil(game: AvalonGame, player: int) -> bool:
        return game.private_knowledge(player)[0][player]

    # ------------------------------------------------------------------ decisions
    def _propose(self, game: AvalonGame, player: int, rng: np.random.Generator) -> int:
        k = game.team_size
        if self._is_evil(game, player):
            known_evil, _ = game.private_knowledge(player)
            s = public_suspicion(game) + 1e-3 * rng.random(N_PLAYERS)
            s[known_evil] = np.inf  # keep the teammate off to avoid a double fail
            s[player] = -np.inf
        else:
            s = self._suspicion(game, player, rng, self._merlin_uses_knowledge(game, player, rng))
        team = np.argsort(s)[:k]
        return A.team_action(player, team)

    def _vote(self, game: AvalonGame, player: int, rng: np.random.Generator) -> int:
        team = game.team
        last_chance = game.attempt == MAX_PROPOSALS - 1
        known_evil, _ = game.private_knowledge(player)
        if self._is_evil(game, player):
            return A.APPROVE if (team & known_evil).any() else A.REJECT
        if last_chance or game.leader == player:
            return A.APPROVE
        if self._merlin_uses_knowledge(game, player, rng):
            return A.REJECT if (team & known_evil).any() else A.APPROVE
        s = self._suspicion(game, player, rng, use_knowledge=False)
        suspects = [seat for seat in np.argsort(-s)[:2] if s[seat] > 0]
        return A.REJECT if team[suspects].any() else A.APPROVE

    def _quest(self, game: AvalonGame, player: int, rng: np.random.Generator) -> int:
        if not self._is_evil(game, player):
            return A.SUCCESS
        known_evil, _ = game.private_knowledge(player)
        evil_on_team = np.flatnonzero(game.team & known_evil)
        if len(evil_on_team) > 1 and player != evil_on_team.min():
            return A.SUCCESS  # let the teammate fail so the quest shows a single fail
        return A.FAIL if rng.random() < self.fail_prob else A.SUCCESS

    def _assassinate(self, game: AvalonGame, player: int, rng: np.random.Generator) -> int:
        known_evil, _ = game.private_knowledge(player)
        score = 1e-3 * rng.random(N_PLAYERS)
        voted = np.argwhere(game.prop_voted)
        for q, a in voted:
            has_evil = (game.prop_team[q, a] & known_evil).any()
            approve = game.prop_votes[q, a]
            if has_evil:
                score += np.where(approve, -1.0, 1.0)
            else:
                score += np.where(approve, 0.25, 0.0)
        score[known_evil] = -np.inf
        return A.assassin_action(player, int(np.argmax(score)))
