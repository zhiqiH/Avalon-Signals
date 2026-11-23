"""Rule-based teacher for the message slots (stage 1 of the communication pilot).

The teacher gives the message heads a meaningful starting point before reinforcement learning; it is
distilled into the network (scripts/distill_speaker.py) and is never an opponent. It is built to be equally
competent for both teams, so that a head start from the teacher does not favor one side:

- Every speaker forms a "good player's view" of the others: public suspicion (`rule_agent.public_suspicion`)
  plus what the deduction fields make certain once the speaker counts itself as good. Percival also leans
  away from his two candidates, as the rule player does.
- Merlin acts on his knowledge in a share of games (`merlin_knowledge_prob`, fixed per game, as for the rule
  player). Then he says what the good player's view would say, minus what he knows to be false: he never
  accuses a good player and never vouches for an evil one. So his talk is a part of what any good player
  could say and gives him away only by never being wrong. Otherwise he speaks from the good player's view
  without his knowledge.
- Evil players speak from the same good player's view, except that they never accuse their teammate; they
  vouch for the teammate only when that view would.
- Judgments: accuse players whose score is at least `accuse_at` (at most two, highest first), vouch for players
  at or below `vouch_at`, say nothing about the rest. The protocol allows a judgment only where a record points
  its way (so nobody judges before the first quest is resolved) and derives its rationale; a judgment the
  protocol does not allow is dropped.
- Claims, once, in the first round (a claim stands): everybody claims Servant, except that Percival claims
  Percival in a share of games (`claim_prob`, fixed per game) and Morgana claims Percival in the same share,
  so that the claim alone proves nothing. Nobody claims Merlin.
- Team suggestions (everybody but the leader; the protocol leaves only the cleanest teams by the record, with the speaker):
  among those, the team with the lowest scores, ties at random. Merlin acting on his knowledge avoids evil
  players and an evil player avoids its teammate (two fails would give both away) wherever the protocol leaves
  such a team.
- Intents, only where the good player's view is certain by deduction: veto a player who must be evil, require
  a player who must be good. An evil player does not veto its teammate.
- Evil talk before the assassination: both evil players name as Merlin the good player whose votes look most
  like Merlin's under the rule Assassin's vote reading.

Every message is built head by head under the protocol's consistency masks, so it is always legal.
"""
from __future__ import annotations

import numpy as np

from ..comm import protocol as P
from ..comm.protocol import good_view  # noqa: F401  (the view itself lives with the protocol's team rule)
from ..env.game import AvalonGame
from ..env.rules import N_PLAYERS, Phase, Role
from .base import ActResult, Request

# A player's words bind its own actions (comm.protocol), so the teacher speaks only on strong evidence: an
# accusation needs about two records against a player (both seats of a failed two-player quest score 1.0 each),
# a vouch two records for it. With the earlier thresholds (1.0, -0.5) good players bound to the teacher's words
# lost a third of their wins: 59% of their accusations were right, against 84% here (frozen baseline, seed 0).
ACCUSE_AT, VOUCH_AT = 2.0, -1.0


def merlin_score(game: AvalonGame, known_evil: np.ndarray) -> np.ndarray:
    """The rule Assassin's vote reading: higher means the seat's votes look more like Merlin's."""
    score = np.zeros(N_PLAYERS)
    for q, a in np.argwhere(game.prop_voted):
        approve = game.prop_votes[q, a]
        if (game.prop_team[q, a] & known_evil).any():
            score += np.where(approve, -1.0, 1.0)
        else:
            score += np.where(approve, 0.25, 0.0)
    score[known_evil] = -np.inf
    return score


class TeacherSpeaker:
    """Speaks by the rules above; acting is delegated to `actor` (any controller) when one is given."""

    learn = False

    def __init__(self, actor=None, merlin_knowledge_prob: float = 0.5, accuse_at: float = ACCUSE_AT,
                 vouch_at: float = VOUCH_AT, claim_prob: float = 0.5, seed: int = 0):
        self.actor = actor
        self.merlin_knowledge_prob, self.accuse_at, self.vouch_at = merlin_knowledge_prob, accuse_at, vouch_at
        self.claim_prob = claim_prob
        self.rng = np.random.default_rng(seed)

    def act(self, requests: list[Request], rng: np.random.Generator) -> ActResult:
        if self.actor is None:
            raise ValueError("the teacher only speaks; give it an actor to play")
        return self.actor.act(requests, rng)

    def speak(self, requests: list[Request], rng: np.random.Generator) -> ActResult:
        return ActResult(messages=np.stack([self.message(r.game, r.player, r.ctx) for r in requests]))

    # ------------------------------------------------------------------ beliefs
    def _draws(self, game: AvalonGame) -> dict[str, bool]:
        """The per-game coin flips of this teacher: drawn at its first message of the game."""
        draws = game.__dict__.setdefault("_teacher_draws", {})
        roles, flips = draws.get(id(self), (None, None))
        if roles is not game.roles:  # first message of this game, or the game was reset
            flips = {"merlin_knows": bool(self.rng.random() < self.merlin_knowledge_prob),
                     "percival_claims": bool(self.rng.random() < self.claim_prob),
                     "morgana_claims": bool(self.rng.random() < self.claim_prob)}
            draws[id(self)] = (game.roles, flips)
        return flips

    def merlin_knows(self, game: AvalonGame) -> bool:
        """Whether Merlin acts on his knowledge in this game."""
        return self._draws(game)["merlin_knows"]

    def claim(self, game: AvalonGame, player: int) -> int:
        role = int(game.roles[player])
        if role == Role.PERCIVAL and self._draws(game)["percival_claims"]:
            return P.C_PERCIVAL
        if role == Role.MORGANA and self._draws(game)["morgana_claims"]:
            return P.C_PERCIVAL
        return P.C_SERVANT

    def scores(self, game: AvalonGame, player: int) -> np.ndarray:
        role = int(game.roles[player])
        known_evil, candidates = game.private_knowledge(player)
        s = good_view(game, player)
        if role == Role.MERLIN and self.merlin_knows(game):
            s[known_evil] = np.maximum(s[known_evil], self.vouch_at + 1e-3)  # never vouch for an evil player
            s[~known_evil] = np.minimum(s[~known_evil], self.accuse_at - 1e-3)  # never accuse a good one
            return s
        if role == Role.PERCIVAL:
            s = s - 0.5 * candidates
        if game.is_evil_seat(player):
            mate = known_evil.copy()
            mate[player] = False
            s[mate] = np.minimum(s[mate], self.accuse_at - 1e-3)  # never accuse the teammate
        return s

    # ------------------------------------------------------------------ messages
    def message(self, game: AvalonGame, player: int, ctx: dict) -> np.ndarray:
        ctx1 = {k: v[None] for k, v in ctx.items()}
        msg = np.full((1, P.N_MSG_HEADS), -1, dtype=np.int64)
        others = P._others(player)
        want = np.zeros(P.N_OTHERS, dtype=np.int64)  # desired judgment per relative seat
        intent = np.zeros(P.N_OTHERS, dtype=np.int64)
        claim, table = P.C_NONE, game.phase != Phase.EVIL_DISCUSS
        if not table:
            score = merlin_score(game, game.private_knowledge(player)[0])
            want[int(np.argmax(score[others]))] = P.J_MERLIN
        else:
            if not (game.msg_log[:game.quest, player, P.H_CLAIM] > P.C_NONE).any():  # said once, it stands
                claim = self.claim(game, player)
            full = self.scores(game, player)
            s = full[others]
            for r in np.argsort(-s)[:2]:
                if s[r] >= self.accuse_at:
                    want[r] = P.J_EVIL
            want[(s <= self.vouch_at) & (want == 0)] = P.J_GOOD
            view = good_view(game, player)[others]  # +-10 where deduction is certain
            intent[view >= 10.0] = P.I_VETO
            intent[view <= -10.0] = P.I_REQUIRE
            if game.is_evil_seat(player):
                intent[game.private_knowledge(player)[0][others]] = P.I_NONE
        self._set(msg, P.H_CLAIM, claim, ctx1)
        for r in range(P.N_OTHERS):
            self._set(msg, P.H_JUDGE + r, want[r], ctx1)
        for h in P.STAGE_B:  # the rationale is derived: its only legal value
            if table and h == P.H_TEAM:
                msg[0, h] = self._team(game, player, full, P.head_mask(h, msg, ctx1)[0])
            else:
                self._set(msg, h, intent[h - P.H_INTENT] if h >= P.H_INTENT else 0, ctx1)
        return msg[0].astype(np.int8)

    def _team(self, game: AvalonGame, player: int, s: np.ndarray, ok: np.ndarray) -> int:
        """Team-head value of the suggestion: the legal team with the lowest scores (0 only if none is legal)."""
        frame = (player + np.arange(N_PLAYERS)) % N_PLAYERS
        cost = s.copy()
        cost[player] = 0.0
        cost = cost + 1e-3 * self.rng.random(N_PLAYERS)  # ties at random
        never = np.zeros(N_PLAYERS, dtype=bool)  # the teammate / a player known to be evil
        if game.is_evil_seat(player) or (game.roles[player] == Role.MERLIN and self.merlin_knows(game)):
            never = game.private_knowledge(player)[0] & (np.arange(N_PLAYERS) != player)
        members = P.TEAM_MEMBERS[game.team_size]  # rows: head values, columns: relative seats
        total = (members * np.clip(cost, -100.0, 100.0)[frame]).sum(axis=1) + 1000.0 * (members & never[frame]).sum(axis=1)
        total[~ok] = np.inf
        return int(np.argmin(total))  # value 0 is legal only when no team is

    @staticmethod
    def _set(msg: np.ndarray, h: int, value: int, ctx1: dict) -> None:
        ok = P.head_mask(h, msg, ctx1)[0]
        if not ok.any():
            msg[0, h] = -1
        elif 0 <= value < len(ok) and ok[value]:
            msg[0, h] = value
        else:
            msg[0, h] = int(np.flatnonzero(ok)[0])  # the "nothing" value where it exists
