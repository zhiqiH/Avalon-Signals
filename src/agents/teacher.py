"""Rule-based teacher for the message slots (stage 1 of the communication pilot).

The teacher gives the message heads a meaningful starting point before reinforcement learning; it is
distilled into the network (scripts/distill_speaker.py) and is never an opponent. It is built to be equally
competent for both teams, so that a head start from the teacher does not favor one side:

- Every speaker forms a "good player's view" of the others: public suspicion (`rule_agent.public_suspicion`)
  plus what the deduction fields make certain once the speaker counts itself as good. Percival also leans
  away from his two candidates, as the rule player does.
- With `CommConfig.teacher_hides_roles` the two role habits of this list are off: Percival has no leaning and
  Merlin never speaks from his knowledge, so both speak as a Servant in their seat would. A learned Assassin
  reads the habits (Percival ends up vouching for Merlin, whose record is the better of his two candidates;
  Merlin never names or endorses a team with an evil player), and the seats the teacher speaks for in training
  cannot learn to hide them.
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
- Where suggestions are a choice (`team_optional`), a speaker names a team only with a reason and only if it adds
  something: the public record must tell the teams with the speaker apart (their totals differ by more than the
  protocol's `TEAM_MARGIN`; never so before the first quest is resolved), and the speaker has endorsed nobody's
  team in this message (it then stands behind that team instead). Both conditions are the same for every role.
- Replies (where the slot is on), to every earlier speaker of the round: endorse its team where that team
  costs at most `ENDORSE_MARGIN` more than the best team with the speaker under the speaker's scores and holds
  nobody the speaker would never put on a team (as for suggestions); the leader endorses (adopts) only the
  cheapest such team. Otherwise dispute wherever the protocol finds a basis (an accusation of the speaker, a
  judgment the other way, an accused player on the team), and say nothing where it finds none.
- Promises (where the slot is on): "fail-evil" for players under suspicion but not yet accused (score at
  least `PROMISE_ACCUSE_AT`, below `accuse_at`), "success-good" for players leaning good but not yet vouched for
  (score above `vouch_at`, at most `PROMISE_VOUCH_AT`): what the teacher would then say anyway. An evil player
  never promises to accuse its teammate. Percival promises by the record alone, without the leaning his sight
  gives him towards his two candidates: promises for both would show the evil team who Merlin is. Merlin
  acting on his knowledge has no rule of his own here: by his scores he promises where another good player
  would accuse a good player or vouch for an evil one, so (where the protocol lets him,
  `merlin_may_accuse_good`) his promises do not set him apart.
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
ENDORSE_MARGIN = 0.5  # an endorsed team may cost this much more than the speaker's best team
PROMISE_ACCUSE_AT, PROMISE_VOUCH_AT = 1.0, -0.5


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
        """Whether Merlin speaks from his knowledge in this game."""
        knows = self._draws(game)["merlin_knows"]  # drawn in every configuration: the same random stream
        return knows and not game.comm.teacher_hides_roles

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
        if role == Role.PERCIVAL and not game.comm.teacher_hides_roles:
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
        if table:
            totals = self._team_totals(game, player, full)
            reply = self._replies(totals, ctx)
            promise = self._promises(game, player, full)
        for h in P.STAGE_B:  # the rationale and the reply basis are derived: their only legal value
            if table and h == P.H_TEAM:
                ok = P.head_mask(h, msg, ctx1)[0]
                if ok[0] and ok[1:].any() and not self._names_a_team(msg[0], ctx):  # a choice: nothing to add
                    msg[0, h] = 0
                    continue
                msg[0, h] = int(np.argmin(np.where(ok, totals, np.inf)))  # 0 only when no team is legal
            elif table and P.H_REPLY <= h < P.H_RBASIS:
                self._set(msg, h, reply[h - P.H_REPLY], ctx1)
            elif table and h >= P.H_PROMISE:
                self._set(msg, h, promise[h - P.H_PROMISE], ctx1)
            else:
                self._set(msg, h, intent[h - P.H_INTENT] if P.H_INTENT <= h < P.H_REPLY else 0, ctx1)
        return msg[0].astype(np.int8)

    def _team_totals(self, game: AvalonGame, player: int, s: np.ndarray) -> np.ndarray:
        """(1 + MAX_TEAMS,) cost of every team-head value's team under the speaker's scores (lower is better; inf
        for value 0 and the padding of smaller quests): a team it would never back costs 1000 more (Merlin acting
        on his knowledge: per evil player on it; an evil player: its own team with its teammate)."""
        frame = (player + np.arange(N_PLAYERS)) % N_PLAYERS
        cost = s.copy()
        cost[player] = 0.0
        cost = cost + 1e-3 * self.rng.random(N_PLAYERS)  # ties at random
        never = np.zeros(N_PLAYERS, dtype=bool)
        if game.is_evil_seat(player) or (game.roles[player] == Role.MERLIN and self.merlin_knows(game)):
            never = game.private_knowledge(player)[0] & (np.arange(N_PLAYERS) != player)
        members = P.TEAM_MEMBERS[game.team_size]  # rows: head values, columns: relative seats
        unwanted = (members & never[frame]).sum(axis=1)
        if game.is_evil_seat(player):  # only a team with both evil players (somebody else's team with just the
            unwanted = unwanted * members[:, 0]  # teammate is as good as any to it)
        total = (members * np.clip(cost, -100.0, 100.0)[frame]).sum(axis=1) + 1000.0 * unwanted
        total[~members.any(axis=1)] = np.inf
        return total

    @staticmethod
    def _names_a_team(msg: np.ndarray, ctx: dict) -> bool:
        """Whether a speaker that may leave the suggestion out makes one (see the module docstring)."""
        cost = ctx["team_cost"][np.isfinite(ctx["team_cost"])]
        told_apart = len(cost) > 0 and cost.max() - cost.min() > P.TEAM_MARGIN
        return bool(told_apart and not (msg[P.H_REPLY:P.H_REPLY + P.N_OTHERS] == P.Y_ENDORSE).any())

    def _replies(self, totals: np.ndarray, ctx: dict) -> np.ndarray:
        """Desired reply per relative seat: endorse a team close to the speaker's best, else dispute (where the
        protocol finds no basis the dispute falls back to saying nothing)."""
        members = ctx["team_members"]  # rows: head values, columns: relative seats
        best = np.min(np.where(members[:, 0], totals, np.inf))  # the best team with the speaker
        values = ctx["their_team"]
        good = ctx["spoke"] & (values > 0) & (totals[values] <= best + ENDORSE_MARGIN) & (totals[values] < 1000.0)
        if ctx["is_leader"] and good.any():  # the leader adopts only the cheapest
            cheapest = np.flatnonzero(good)[np.argmin(totals[values[good]])]
            good[:] = False
            good[cheapest] = True
        return np.where(good, P.Y_ENDORSE, np.where(ctx["spoke"], P.Y_DISPUTE, P.Y_NONE))

    def _promises(self, game: AvalonGame, player: int, s: np.ndarray) -> np.ndarray:
        """Desired promise per relative seat (see the module docstring)."""
        others = P._others(player)
        score = s[others]
        if game.roles[player] == Role.PERCIVAL and not game.comm.teacher_hides_roles:  # by the record alone: his leaning would promise for both
            score = score + 0.5 * game.private_knowledge(player)[1][others]  # candidates, and so point at Merlin
        out = np.full(P.N_OTHERS, P.K_NONE, dtype=np.int64)
        out[(score >= PROMISE_ACCUSE_AT) & (score < self.accuse_at)] = P.K_FAIL_EVIL
        out[(score > self.vouch_at) & (score <= PROMISE_VOUCH_AT)] = P.K_SUCCESS_GOOD
        known_evil = game.private_knowledge(player)[0][others]
        if game.is_evil_seat(player):
            out[known_evil & (out == P.K_FAIL_EVIL)] = P.K_NONE  # never against the teammate
        return out

    @staticmethod
    def _set(msg: np.ndarray, h: int, value: int, ctx1: dict) -> None:
        ok = P.head_mask(h, msg, ctx1)[0]
        if not ok.any():
            msg[0, h] = -1
        elif 0 <= value < len(ok) and ok[value]:
            msg[0, h] = value
        else:
            msg[0, h] = int(np.flatnonzero(ok)[0])  # the "nothing" value where it exists
