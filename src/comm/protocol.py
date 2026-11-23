"""Five-slot structured message protocol.

Timing. Every quest opens with one round of public discussion before its first proposal: starting with
the player to the leader's left and going round the table, each player speaks once and the leader speaks
last; then the leader proposes and everyone votes. After a rejected proposal leadership passes on and the
next leader proposes at once, without discussion. After the third successful quest the two evil players
(Morgana, then the Assassin) each speak once to judge together who is Merlin; then the Assassin strikes.
There is no dialogue: each player speaks once, in order, and cannot address or question anyone.

A message has five slots. Every slot may say nothing, may be false and may be partial, and a message only
concerns the current round (players may change their story in later rounds), but a single message cannot
contradict itself. A speaker chooses its stance (claim, judgments, team suggestion, intents); the rationale of
each judgment is derived from that stance by a fixed rule, so how a judgment is justified cannot become a
private code. Heads, in the speaker's relative frame (r = 1..5 are the other players):

    head        values                                                                  slot
    claim       none, Servant, Percival (Merlin: closed, nobody at a table claims it)      (1) role claim
    judge[r]    none, good, evil (role names only in the evil talk)                        (2) judgment
    rtype[r]    quest record, vote record, proposal record, role knowledge                 (3) rationale (derived)
    rround[r]   quest 1..5 of the cited record                                          (3) rationale (derived)
    team        none, or a team of the current quest size (same index as proposals)      (4) team suggestion
    intent[r]   none, veto, require                                                     (5) action intent

Action intents announce how the speaker will vote in the current round:
    veto      "I will reject any team with player r in this round"
    require   "I will reject any team without player r in this round"

A player's words bind its own actions (`AvalonGame.word_binding`, a public rule for both teams, enforced by the
action masks; `word_bounds`, `word_vote`):
    accusation   whoever it accuses (the evil judgments of its latest message that accused anybody: a new list
                 of suspects replaces the old one, silence keeps it, calling a suspect good clears it) it will
                 not propose, and it rejects any team holding one of them;
    intent       it proposes and approves only teams that meet the intents it announced in this round;
    suggestion   it approves exactly the team it suggested in this round;
    vouching     it approves a team whose other members it all judged good in this round, unless an intent of
                 its own says otherwise;
    proposal     a leader approves its own proposal, and proposes one of the cleanest teams by the record
                 that its words leave it (the rule on team suggestions below, applied to the deed);
    naming       the Assassin strikes the player it named as Merlin in the evil talk.
Voters are free on the forced fifth proposal of a round, and the good players' vote guard (`AvalonGame.good_guard`)
comes first. With the binding switched off, the same deeds against one's own words are counted in public instead
(`AUDIT`, kept by the game and shown to every player); vouching for a player who then sits on a failed quest is
always counted:
    judge     backed (proposed or approved) a team holding a player it judged evil in that round
    intent    backed a team that breaks an intent it announced in that round
    team      rejected exactly the team it suggested in that round
    vouch     its latest judgment of a member of a failed quest's team was good (counted per member)

Consistency rules (enforced by masks, sampled in the order above):
- claim and judgments respect the role deck: at most one Merlin, one Percival, one Morgana, one Assassin,
  two Servants; at most two players judged evil (evil, Morgana, Assassin) and four judged good (good, Servant,
  Merlin, Percival), where a claim counts the speaker as good;
- a claim stands: once a player has claimed a role, it can only repeat that claim or say nothing about itself;
- no judgment without a basis: a player can be judged one way (good or evil) only if some record points that
  way. Before the first quest is resolved there is no record, so nobody can judge anybody: the first round is
  for claims, team suggestions and intents. The evil talk is exempt;
- Merlin cannot call evil a player he knows to be good (he may keep silent about anyone);
- the rationale is the reasoning behind a judgment: it exists only for a judged player and points the same
  way. It is derived: role knowledge if the claim covers the judgment (see below), else the strongest record
  that points the judgment's way: a quest record before a vote record before a proposal record, and the most
  recent of that kind. Records cite a resolved quest q:
      quest record     the player was on quest q, which failed (evil) / succeeded (good);
      vote record      the player approved quest q's team that then failed, or rejected one that then
                       succeeded (evil); approved a success or rejected a failure (good). A forced fifth
                       proposal leaves no vote record: good players had to approve it;
      proposal record  the player led quest q's team, which failed (evil) / succeeded (good);
      role knowledge   the claimed role's sight (unused while the Merlin claim and role names are closed);
- everybody but the leader suggests a team (the leader's proposal follows its message). The team holds the
  speaker, nobody the speaker accuses, and is not a team the good players' vote guard would make the speaker
  reject. Among the teams this leaves, it must be one of the cleanest by the public record as a good player in
  the speaker's seat reads it (`good_view`: the public suspicion score, and what the quest results make certain
  for a player who is good): its total may exceed the lowest total by at most `TEAM_MARGIN`. So nobody can
  suggest a team with a player the record points at while a cleaner team is open to it, and what a speaker adds
  by its choice lies among teams the record cannot tell apart. The same holds for the team a leader proposes
  (in every configuration with the team slot, also while the slot is silenced at test time);
- intents cannot contradict the judgments: no veto on a player judged good, no require with an accused
  player; the vote conditions, with every accusation counted as a veto, must be jointly satisfiable by some
  team of the current size, and by the speaker's own suggestion if it makes one;
- no unfounded change of mind: if the speaker's most recent earlier judgment of a player (in an earlier round's
  discussion) pointed the other way (good vs evil), a judgment in the new direction needs a record rationale that
  cites a quest resolved since that judgment and points the new way; without such a record the speaker can only
  keep the direction or say nothing about that player. The evil talk is exempt.
Before the assassination only judgments of good players as Servant, Merlin or Percival (with their rationale,
without role knowledge) are open. The Assassin, who speaks last, names exactly one player as Merlin.

A switched-off slot (`CommConfig`) or a slot silenced at test time can only say nothing. The observation tells
which slots exist in the trained configuration (`slot_open`) but not which are silenced, so a silenced slot looks
exactly like a player choosing to say nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..env import actions as A
from ..env.rules import EVIL_ROLES, MAX_PROPOSALS, N_PLAYERS, N_QUESTS, QUEST_SIZES, Phase, Role

SLOTS = ("claim", "alignment", "rationale", "team", "intent")


@dataclass
class CommConfig:
    enabled: bool = False  # False: the silent game (no discussion phases at all)
    claim: bool = True
    alignment: bool = True  # (2) judgment
    rationale: bool = True
    team: bool = True
    intent: bool = True  # (5) action intent
    silence: list[str] = field(default_factory=list)  # test time: slots forced to say nothing (F -> F-k, F -> S)

    def __post_init__(self) -> None:
        unknown = set(self.silence) - set(SLOTS)
        if unknown:
            raise ValueError(f"unknown slots to silence: {sorted(unknown)}")
        if self.enabled and self.rationale and not self.alignment:
            raise ValueError("the rationale slot needs the judgment slot")

    def is_open(self, slot: str) -> bool:
        """Whether the slot exists in this configuration (what the policy was trained with)."""
        return self.enabled and getattr(self, slot)

    def can_speak(self, slot: str) -> bool:
        return self.is_open(slot) and slot not in self.silence


# ---------------------------------------------------------------------- values and heads
N_OTHERS = N_PLAYERS - 1
CLAIM_VALUES = ("none", "servant", "percival", "merlin")
C_NONE, C_SERVANT, C_PERCIVAL, C_MERLIN = range(len(CLAIM_VALUES))
JUDGE_VALUES = ("none", "good", "evil", "servant", "merlin", "percival", "morgana", "assassin")
J_NONE, J_GOOD, J_EVIL, J_SERVANT, J_MERLIN, J_PERCIVAL, J_MORGANA, J_ASSASSIN = range(len(JUDGE_VALUES))
GOOD_JUDGMENTS = (J_GOOD, J_SERVANT, J_MERLIN, J_PERCIVAL)
EVIL_JUDGMENTS = (J_EVIL, J_MORGANA, J_ASSASSIN)
JUDGE_DIR = np.array([-1, 0, 1, 0, 0, 0, 1, 1])  # direction of a judgment: 0 good, 1 evil
EVIL_TALK_JUDGMENTS = (J_SERVANT, J_MERLIN, J_PERCIVAL)  # before the assassination: which good role
RATIONALE_TYPES = ("none", "quest", "vote", "proposal", "role_knowledge")
R_NONE, R_QUEST, R_VOTE, R_PROPOSAL, R_ROLE = range(len(RATIONALE_TYPES))
RECORD_TYPES = (R_QUEST, R_VOTE, R_PROPOSAL)
INTENT_VALUES = ("none", "veto", "require")
I_NONE, I_VETO, I_REQUIRE = range(len(INTENT_VALUES))

H_CLAIM = 0
H_JUDGE = 1  # judge[r] is head H_JUDGE + r - 1
H_RTYPE = H_JUDGE + N_OTHERS
H_RROUND = H_RTYPE + N_OTHERS
H_TEAM = H_RROUND + N_OTHERS
H_INTENT = H_TEAM + 1
N_MSG_HEADS = H_INTENT + N_OTHERS

MSG_HEAD_SIZES = (
    (len(CLAIM_VALUES),)
    + (len(JUDGE_VALUES),) * N_OTHERS
    + (len(RATIONALE_TYPES),) * N_OTHERS
    + (N_QUESTS,) * N_OTHERS
    + (1 + A.MAX_TEAMS,)
    + (len(INTENT_VALUES),) * N_OTHERS
)
MSG_HEAD_OFFSETS = tuple(int(x) for x in np.cumsum((0,) + MSG_HEAD_SIZES[:-1]))
N_MSG_LOGITS = sum(MSG_HEAD_SIZES)
# Sampling order. Stage A (claim, judgments) is what the later heads are conditioned on.
STAGE_A = (H_CLAIM, *range(H_JUDGE, H_JUDGE + N_OTHERS))
STAGE_B = (*range(H_RTYPE, H_RTYPE + N_OTHERS), *range(H_RROUND, H_RROUND + N_OTHERS), H_TEAM,
           *range(H_INTENT, H_INTENT + N_OTHERS))
COND_DIM = len(CLAIM_VALUES) + N_OTHERS * len(JUDGE_VALUES)

# TEAM_MEMBERS[k][v, s]: team-head value v (team v-1 of size k, 0 = none) contains relative seat s.
TEAM_MEMBERS = {}
for _k, _teams in A.TEAMS_BY_SIZE.items():
    TEAM_MEMBERS[_k] = np.zeros((1 + A.MAX_TEAMS, N_PLAYERS), dtype=bool)
    for _i, _team in enumerate(_teams):
        TEAM_MEMBERS[_k][1 + _i, list(_team)] = True


def head_slice(mask: np.ndarray, h: int) -> np.ndarray:
    return mask[..., MSG_HEAD_OFFSETS[h]:MSG_HEAD_OFFSETS[h] + MSG_HEAD_SIZES[h]]


# ---------------------------------------------------------------------- legality
def _others(speaker: int) -> np.ndarray:
    """Absolute seats of the speaker's relative seats 1..5."""
    return (speaker + np.arange(1, N_PLAYERS)) % N_PLAYERS


# The public record as a score, for the rule on team suggestions (also the rule players' and the teacher's view).
BROKEN_WORD, FAILED_VOUCH = 1.0, 0.5  # suspicion per counted deed against one's words / per failed vouch
TEAM_MARGIN = 0.5  # how much more suspicious than the cleanest open team a named team may be


def public_suspicion(game) -> np.ndarray:
    """Suspicion per absolute seat computed from public quest and vote records and the public audit only."""
    audit = game.audit.astype(float)  # all zero in the silent game
    s = BROKEN_WORD * (audit.sum(axis=1) - audit[:, A_VOUCH]) + FAILED_VOUCH * audit[:, A_VOUCH]
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


def good_view(game, player: int) -> np.ndarray:
    """Suspicion of every seat as a good player without special knowledge would see it from `player`'s seat:
    the public score, +-10 where the quest results make a seat certainly evil / good (own seat -inf)."""
    from ..env.observation import _PAIR_MEMBERS, _certain, possible_evil_pairs  # observation imports this module

    s = public_suspicion(game).astype(float)
    possible = possible_evil_pairs(game) & ~_PAIR_MEMBERS[:, player]  # "I am good"
    if possible.any():
        evil, good = _certain(possible)
        s = np.where(evil, 10.0, np.where(good, -10.0, s))
    s[player] = -np.inf
    return s


def team_costs(game, speaker: int) -> np.ndarray:
    """(1 + MAX_TEAMS,) total `good_view` score of each team-head value's team; inf for the teams not open to the
    speaker whatever it says (without itself, or one the vote guard would make it reject) and for value 0."""
    n = len(A.TEAMS_BY_SIZE[game.team_size])
    members = TEAM_MEMBERS[game.team_size][1:1 + n]
    view = good_view(game, speaker)
    view[speaker] = 0.0
    total = (members * view[(speaker + np.arange(N_PLAYERS)) % N_PLAYERS]).sum(axis=1)
    total[~members[:, 0]] = np.inf
    if game.word_binding:
        total[game.guarded_teams(speaker)] = np.inf
    cost = np.full(1 + A.MAX_TEAMS, np.inf)
    cost[1:1 + n] = total
    return cost


def evidence(game, speaker: int) -> np.ndarray:
    """evidence[r-1, k, d, q]: record type RECORD_TYPES[k] about player r in resolved quest q points to d (0 good, 1 evil)."""
    ev = np.zeros((N_OTHERS, len(RECORD_TYPES), 2, N_QUESTS), dtype=bool)
    q = np.flatnonzero(game.quest_result >= 0)
    if len(q) == 0:
        return ev
    others = _others(speaker)
    failed = game.quest_result[q] == 1
    attempt = game.prop_approved[q].argmax(axis=1)  # the proposal that went on the quest
    on_team = game.quest_team[q][:, others].T
    approved = game.prop_votes[q, attempt][:, others].T
    led = game.prop_leader[q, attempt][None, :] == others[:, None]
    wrong_vote = approved == failed[None, :]  # approved a failure or rejected a success
    free_vote = np.broadcast_to(attempt < MAX_PROPOSALS - 1, on_team.shape)  # the fifth proposal is forced
    for k, (hit, evil) in enumerate([(on_team, np.broadcast_to(failed, on_team.shape)),
                                     (free_vote, wrong_vote),
                                     (led, np.broadcast_to(failed, led.shape))]):
        ev[:, k, 1, q] = hit & evil
        ev[:, k, 0, q] = hit & ~evil
    return ev


def message_context(game, speaker: int) -> dict[str, np.ndarray]:
    """Everything the masks need about the game, for one speaker (stack several with `stack_contexts`)."""
    cfg = game.comm
    evil_talk = game.phase == Phase.EVIL_DISCUSS
    claim_ok = np.zeros(len(CLAIM_VALUES), dtype=bool)
    judge_ok = np.zeros((N_OTHERS, len(JUDGE_VALUES)), dtype=bool)
    team_ok = np.zeros(1 + A.MAX_TEAMS, dtype=bool)
    team_cost = np.full(1 + A.MAX_TEAMS, np.inf)
    claim_ok[0] = judge_ok[:, 0] = team_ok[0] = True
    size = QUEST_SIZES[min(game.quest, N_QUESTS - 1)]
    if evil_talk:
        if cfg.can_speak("alignment"):
            good = ~np.isin(game.roles[_others(speaker)], list(EVIL_ROLES))
            judge_ok[np.ix_(good, EVIL_TALK_JUDGMENTS)] = True
    else:
        claim_ok[[C_SERVANT, C_PERCIVAL]] = cfg.can_speak("claim")
        for q in range(min(game.quest, N_QUESTS)):  # a claim stands
            if game.msg_log[q, speaker, H_CLAIM] > C_NONE:
                claim_ok[1:] = False
                claim_ok[game.msg_log[q, speaker, H_CLAIM]] = cfg.can_speak("claim")
        judge_ok[:, [J_GOOD, J_EVIL]] = cfg.can_speak("alignment")
        if game.roles[speaker] == Role.MERLIN:  # Merlin does not accuse a player he knows to be good
            judge_ok[~game.private_knowledge(speaker)[0][_others(speaker)], J_EVIL] = False
        if cfg.can_speak("team") and speaker != game.leader:  # the teams open to it; `head_mask` keeps the cleanest
            team_cost = team_costs(game, speaker)
            team_ok[1:] = np.isfinite(team_cost[1:])
    rationale = cfg.can_speak("rationale")
    records = evidence(game, speaker)
    basis = np.ones((N_OTHERS, 2), dtype=bool) if evil_talk else records.any(axis=(1, 3))  # (player, direction)
    ev = records if rationale else np.zeros_like(records)
    for v in range(1, len(JUDGE_VALUES)):  # no basis whatever the claim: closed here, so an empty message is forced
        by_sight = claim_ok[C_MERLIN] if v in (J_GOOD, J_EVIL) else claim_ok[C_PERCIVAL] if v in (J_MERLIN, J_MORGANA) else False
        judge_ok[:, v] &= basis[:, JUDGE_DIR[v]] | by_sight
    # Change of mind: the direction of the speaker's latest earlier judgment of each player, and the records
    # resolved since then (a discussion of round q precedes quest q, so quests q, q+1, ... are new to it).
    prev_dir = np.full(N_OTHERS, -1, dtype=np.int8)
    prev_accused = np.zeros(N_OTHERS, dtype=bool)  # whom the speaker's earlier messages still accuse
    fresh = ev.copy()
    if not evil_talk:
        for q in range(min(game.quest, N_QUESTS)):
            judged = game.msg_log[q, speaker, H_JUDGE:H_JUDGE + N_OTHERS].astype(np.int64)
            said = judged > J_NONE
            prev_dir[said] = JUDGE_DIR[judged[said]]
            fresh[said] = ev[said] & (np.arange(N_QUESTS) >= q)
            prev_accused = _accused_after(prev_accused, judged)
    return {
        "claim_ok": claim_ok, "judge_ok": judge_ok, "team_ok": team_ok, "team_members": TEAM_MEMBERS[size],
        "team_cost": team_cost,
        "must_name": np.array(evil_talk and game.roles[speaker] == Role.ASSASSIN and cfg.can_speak("alignment")),
        "evidence": ev, "prev_dir": prev_dir, "prev_accused": prev_accused, "fresh": fresh, "basis": basis,
        "sight": np.array(not evil_talk),
        "rationale": np.array(rationale), "role_knowledge": np.array(rationale and not evil_talk),
        "intent": np.array(cfg.can_speak("intent") and not evil_talk),
    }


def _accused_after(accused: np.ndarray, judged: np.ndarray) -> np.ndarray:
    """Whom a speaker accuses (..., relative seats 1..5) after a message with the judgments `judged`: the
    message's own evil judgments if it has any, else the earlier list without those it now calls good."""
    evil = np.isin(judged, EVIL_JUDGMENTS)
    return np.where(evil.any(axis=-1, keepdims=True), evil, accused & ~np.isin(judged, GOOD_JUDGMENTS))


def _flipped(judge: np.ndarray, prev_dir: np.ndarray) -> np.ndarray:
    """Whether judgments (batch,) turn against the speaker's latest earlier judgment of the same player."""
    return (judge > J_NONE) & (prev_dir >= 0) & (JUDGE_DIR[judge.clip(0)] != prev_dir)


def _latest_record(records: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For records (batch, record type, quest): type index and quest of the one to cite (the first type that has
    a record, its most recent quest), and whether any exists."""
    k, q = np.arange(records.shape[1])[None, :, None], np.arange(records.shape[2])[None, None, :]
    rank = np.where(records, q - k * records.shape[2], -records.shape[1] * records.shape[2] - 1)
    best = rank.reshape(len(records), -1).argmax(axis=1)
    return best // records.shape[2], best % records.shape[2], records.any(axis=(1, 2))


def stack_contexts(contexts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.stack([c[k] for c in contexts]) for k in contexts[0]}


def head_mask(h: int, msg: np.ndarray, ctx: dict[str, np.ndarray]) -> np.ndarray:
    """Legal values of head h (batch, size) given the heads already chosen in `msg` (batch, N_MSG_HEADS)."""
    n = len(msg)
    rows = np.arange(n)
    claim = msg[:, H_CLAIM]
    if h == H_CLAIM:
        return ctx["claim_ok"].copy()
    if h < H_RTYPE:  # judgment of relative seat r = h - H_JUDGE + 1, limited by the role deck
        prev = msg[:, H_JUDGE:h]
        count = lambda v: (prev == v).sum(axis=1)  # noqa: E731
        n_good = np.isin(prev, GOOD_JUDGMENTS).sum(axis=1) + (claim > C_NONE)
        n_evil = np.isin(prev, EVIL_JUDGMENTS).sum(axis=1)
        ok = ctx["judge_ok"][:, h - H_JUDGE].copy()
        ok[:, list(GOOD_JUDGMENTS)] &= (n_good < 4)[:, None]
        ok[:, list(EVIL_JUDGMENTS)] &= (n_evil < 2)[:, None]
        ok[:, J_SERVANT] &= count(J_SERVANT) + (claim == C_SERVANT) < 2
        ok[:, J_MERLIN] &= count(J_MERLIN) + (claim == C_MERLIN) < 1
        ok[:, J_PERCIVAL] &= count(J_PERCIVAL) + (claim == C_PERCIVAL) < 1
        ok[:, J_MORGANA] &= count(J_MORGANA) < 1
        ok[:, J_ASSASSIN] &= count(J_ASSASSIN) < 1
        r = h - H_JUDGE
        unnamed = ctx["must_name"] & (count(J_MERLIN) == 0)  # the Assassin still has to name its Merlin
        # A judgment needs a basis: a record pointing its way, or the claimed role's sight.
        merlin_sight = ctx["sight"] & (claim == C_MERLIN)
        percival_sight = ctx["sight"] & (claim == C_PERCIVAL)
        # Turning against one's earlier judgment needs a new record pointing the new way.
        prev = ctx["prev_dir"][:, r]
        new_record = ctx["fresh"][:, r].any(axis=(1, 3))  # (batch, direction)
        for v in range(1, len(JUDGE_VALUES)):
            d = JUDGE_DIR[v]
            sight = (merlin_sight if v in (J_GOOD, J_EVIL) else percival_sight if v in (J_MERLIN, J_MORGANA)
                     else np.zeros(n, dtype=bool))
            ok[:, v] &= ctx["basis"][:, r, d] | sight
            ok[:, v] &= (prev < 0) | (prev == d) | new_record[:, d]
        last_chance = unnamed & ok[:, J_MERLIN] & ~ctx["judge_ok"][:, r + 1:, J_MERLIN].any(axis=1)
        ok[last_chance] = False
        ok[last_chance, J_MERLIN] = True
        return ok
    if h < H_RROUND:  # rationale type: only for a judged player, pointing the judgment's way
        r = h - H_RTYPE
        judge = msg[:, H_JUDGE + r]
        judged = (judge > J_NONE) & ctx["rationale"]
        direction = JUDGE_DIR[judge.clip(0)].clip(0)
        flipped = _flipped(judge, ctx["prev_dir"][:, r])
        records = np.where(flipped[:, None, None], ctx["fresh"][rows, r, :, direction, :],
                           ctx["evidence"][rows, r, :, direction, :])
        # The rationale is derived, not chosen: the claimed role's sight if the claim covers the judgment, else
        # the strongest record pointing the judgment's way (quest, vote, proposal; the most recent of its kind).
        # "No basis" is left only where the judgment mask asks for none (the evil talk).
        k, _, found = _latest_record(records)
        derived = np.where(found, np.array(RECORD_TYPES)[k], R_NONE)
        merlin_sight = np.isin(judge, (J_GOOD, J_EVIL)) & (claim == C_MERLIN)
        percival_sight = np.isin(judge, (J_MERLIN, J_MORGANA)) & (claim == C_PERCIVAL)
        derived[~flipped & ctx["role_knowledge"] & (merlin_sight | percival_sight)] = R_ROLE
        ok = np.zeros((n, len(RATIONALE_TYPES)), dtype=bool)
        ok[rows, derived] = judged
        return ok
    if h < H_TEAM:  # cited quest of a record rationale
        r = h - H_RROUND
        rtype = msg[:, H_RTYPE + r]
        cites = np.isin(rtype, RECORD_TYPES)
        judge = msg[:, H_JUDGE + r]
        direction = JUDGE_DIR[judge.clip(0)].clip(0)
        source = np.where(_flipped(judge, ctx["prev_dir"][:, r])[:, None, None, None, None], ctx["fresh"],
                          ctx["evidence"])
        cited = source[rows, r, (rtype - R_QUEST).clip(0, 2), direction, :] & cites[:, None]
        latest = N_QUESTS - 1 - cited[:, ::-1].argmax(axis=1)  # derived as well: the most recent such record
        ok = np.zeros_like(cited)
        ok[rows, latest] = cited.any(axis=1)
        return ok
    accused = _accused_after(ctx["prev_accused"], msg[:, H_JUDGE:H_JUDGE + N_OTHERS])  # relative seats 1..5
    if h == H_TEAM:  # one of the cleanest teams open to the speaker; nothing only if none is open
        open_ = ctx["team_ok"] & ~(ctx["team_members"][:, :, 1:] & accused[:, None, :]).any(axis=-1)
        open_[:, 0] = False
        cost = np.where(open_, ctx["team_cost"], np.inf)
        ok = open_ & (cost <= cost.min(axis=1, keepdims=True) + TEAM_MARGIN)
        ok[:, 0] = ~ok.any(axis=1)
        return ok
    r = h - H_INTENT  # intent about relative seat r + 1
    judge = msg[:, H_JUDGE + r]
    team = msg[:, H_TEAM]
    candidates = ctx["team_members"][:, 1:]
    real = candidates.any(axis=-1)  # padding rows of smaller quests are empty
    own = ctx["team_members"][rows, team.clip(0)][:, None]
    ok = np.zeros((n, len(INTENT_VALUES)), dtype=bool)
    ok[:, I_NONE] = True
    chosen = msg[:, H_INTENT:H_INTENT + N_OTHERS].clip(0)  # heads not chosen yet count as none
    for v in range(1, len(INTENT_VALUES)):
        intents = chosen.copy()
        intents[:, r] = v
        bound = np.where(accused & (intents == I_NONE), I_VETO, intents)  # an accusation is a veto
        feasible = (_satisfies(candidates, bound) & real).any(axis=1)
        ok[:, v] = ctx["intent"] & feasible & ((team <= 0) | _satisfies(own, intents)[:, 0])
    ok[:, I_VETO] &= ~np.isin(judge, GOOD_JUDGMENTS)
    ok[:, I_REQUIRE] &= ~accused[:, r]
    return ok


def _satisfies(teams: np.ndarray, intents: np.ndarray) -> np.ndarray:
    """Whether each team (batch, k, 6 relative seats) meets the vote conditions in `intents` (batch, 5)."""
    others = teams[..., 1:]
    it = intents[:, None, :]
    broken = ((it == I_VETO) & others) | ((it == I_REQUIRE) & ~others)
    return ~broken.any(axis=-1)


# ---------------------------------------------------------------------- words bind deeds
def accused_by(game, player: int) -> np.ndarray:
    """(6,) bool over absolute seats: whom `player` accuses now, i.e. the evil judgments of its latest message,
    up to this round, that accused anybody, without those it has called good since."""
    accused = np.zeros(N_OTHERS, dtype=bool)
    for q in range(min(game.quest, N_QUESTS - 1) + 1):
        accused = _accused_after(accused, game.msg_log[q, player, H_JUDGE:H_JUDGE + N_OTHERS])
    out = np.zeros(N_PLAYERS, dtype=bool)
    out[_others(player)] = accused
    return out


def word_bounds(game, player: int) -> tuple[np.ndarray, np.ndarray]:
    """(banned, required) over absolute seats: the teams `player` may back in this round by its own words
    hold nobody banned (accused or vetoed) and everybody required."""
    intents = game.msg_log[game.quest, player, H_INTENT:H_INTENT + N_OTHERS]
    banned, required = accused_by(game, player), np.zeros(N_PLAYERS, dtype=bool)
    banned[_others(player)] |= intents == I_VETO
    required[_others(player)] = intents == I_REQUIRE
    return banned, required


def word_teams(game, player: int) -> np.ndarray:
    """(number of teams of the current size,) bool: the teams `player` may propose by its own words."""
    banned, required = word_bounds(game, player)
    frame = (player + np.arange(N_PLAYERS)) % N_PLAYERS
    members = TEAM_MEMBERS[game.team_size][1:1 + len(A.TEAMS_BY_SIZE[game.team_size])]
    return ~(members & banned[frame]).any(axis=1) & ~(~members & required[frame]).any(axis=1)


def clean_teams(game, player: int) -> np.ndarray:
    """(number of teams of the current size,) bool: the cleanest teams by the record that `player`'s words leave it."""
    n = len(A.TEAMS_BY_SIZE[game.team_size])
    cost = np.where(word_teams(game, player), team_costs(game, player)[1:1 + n], np.inf)
    return np.isfinite(cost) & (cost <= cost.min() + TEAM_MARGIN)


def named_merlin(game, player: int) -> int | None:
    """Absolute seat of the player `player` named as Merlin in the evil talk, if it named one."""
    r = np.flatnonzero(game.evil_log[player, H_JUDGE:H_JUDGE + N_OTHERS] == J_MERLIN)
    return int(_others(player)[r[0]]) if len(r) else None


def word_vote(game, player: int) -> bool | None:
    """The vote on the team under vote that `player`'s own words and deeds leave it: True approve, False reject,
    None free."""
    if player == game.leader:
        return True  # its own proposal
    if game.attempt == MAX_PROPOSALS - 1:
        return None  # the forced fifth proposal tests nobody's word
    banned, required = word_bounds(game, player)
    if (game.team & banned).any() or (required & ~game.team).any():
        return False
    if is_own_suggestion(game, player):
        return True
    judged = game.msg_log[game.quest, player, H_JUDGE:H_JUDGE + N_OTHERS]
    vouched = np.zeros(N_PLAYERS, dtype=bool)
    vouched[_others(player)] = np.isin(judged, GOOD_JUDGMENTS)
    mates = game.team.copy()
    mates[player] = False
    return True if vouched[mates].all() else None


# ---------------------------------------------------------------------- word against deed (public audit)
# With the binding switched off, what a player does against its own words is counted in public
# (`AvalonGame.audit`, shown to every player by `encode_comm`), so that others can hold it to account.
AUDIT = ("judge", "intent", "team", "vouch")
A_JUDGE, A_INTENT, A_TEAM, A_VOUCH = range(len(AUDIT))
AUDIT_LEVELS = 3  # each count is shown as "at least 1, 2, 3"


def backs_against_word(game, player: int) -> tuple[bool, bool]:
    """Backing (proposing or approving) the team under vote against one's own message of this round:
    (the team holds a player it judged evil, the team breaks an intent it announced)."""
    msg = game.msg_log[game.quest, player].astype(np.int64)
    on_team = game.team[_others(player)]
    intents = msg[H_INTENT:H_INTENT + N_OTHERS]
    judged_evil = np.isin(msg[H_JUDGE:H_JUDGE + N_OTHERS], EVIL_JUDGMENTS)
    return bool((judged_evil & on_team).any()), bool((((intents == I_VETO) & on_team) | ((intents == I_REQUIRE) & ~on_team)).any())


def is_own_suggestion(game, player: int) -> bool:
    """Whether the team under vote is exactly the team the player suggested in this round's discussion."""
    value = int(game.msg_log[game.quest, player, H_TEAM])
    if value <= 0:
        return False
    return bool((TEAM_MEMBERS[game.team_size][value] == game.team[(player + np.arange(N_PLAYERS)) % N_PLAYERS]).all())


def vouched_for(game, player: int) -> np.ndarray:
    """(6,) bool over absolute seats: players whose latest judgment by `player`, up to this round, is good."""
    direction = np.full(N_OTHERS, -1, dtype=np.int64)
    for q in range(min(game.quest, N_QUESTS - 1) + 1):
        judged = game.msg_log[q, player, H_JUDGE:H_JUDGE + N_OTHERS].astype(np.int64)
        said = judged > J_NONE
        direction[said] = JUDGE_DIR[judged[said]]
    out = np.zeros(N_PLAYERS, dtype=bool)
    out[_others(player)] = direction == 0
    return out


def sample_messages(ctx: dict[str, np.ndarray], rng: np.random.Generator, first: bool = False) -> np.ndarray:
    """Messages for a batch of contexts: each head uniformly among its legal values (or its first legal value)."""
    n = len(ctx["claim_ok"])
    msg = np.full((n, N_MSG_HEADS), -1, dtype=np.int64)
    for h in STAGE_A + STAGE_B:
        ok = head_mask(h, msg, ctx)
        score = np.where(ok, 1.0 if first else rng.random(ok.shape), -1.0)
        msg[:, h] = np.where(ok.any(axis=1), score.argmax(axis=1), -1)
    return msg.astype(np.int8)


def forced_message(ctx: dict[str, np.ndarray]) -> np.ndarray | None:
    """The only legal message of one speaker when no slot can say anything (all closed or silenced)."""
    if ctx["claim_ok"][1:].any() or ctx["judge_ok"][:, 1:].any() or ctx["team_ok"][1:].any() or ctx["intent"]:
        return None
    msg = np.full(N_MSG_HEADS, -1, dtype=np.int8)
    msg[[H_CLAIM, *range(H_JUDGE, H_JUDGE + N_OTHERS), H_TEAM, *range(H_INTENT, H_INTENT + N_OTHERS)]] = 0
    return msg


def check_messages(msgs: np.ndarray, ctx: dict[str, np.ndarray]) -> None:
    """Raise ValueError unless every message of the batch is legal for its context."""
    msgs = np.asarray(msgs, dtype=np.int64)
    if msgs.ndim != 2 or msgs.shape[1] != N_MSG_HEADS:
        raise ValueError(f"messages must have shape (batch, {N_MSG_HEADS}), got {msgs.shape}")
    rows = np.arange(len(msgs))
    for h in STAGE_A + STAGE_B:
        ok = head_mask(h, msgs, ctx)
        v = msgs[:, h]
        legal = np.where(ok.any(axis=1), (v >= 0) & (v < ok.shape[1]) & ok[rows, v.clip(0, ok.shape[1] - 1)], v == -1)
        if not legal.all():
            bad = int(np.flatnonzero(~legal)[0])
            raise ValueError(f"illegal value {v[bad]} for message head {h} (message {bad} of the batch)")


def check_message(game, speaker: int, msg: np.ndarray, free: bool = False) -> None:
    """Raise ValueError unless `msg` is a legal message of `speaker` now (`free`: of a seat the rules do not bind)."""
    msg = np.asarray(msg, dtype=np.int64)
    if not free:
        return check_messages(msg[None], stack_contexts([message_context(game, speaker)]))
    if msg.shape != (N_MSG_HEADS,):
        raise ValueError(f"a message must have shape ({N_MSG_HEADS},), got {msg.shape}")
    rebuilt = free_message(game, speaker, msg[H_CLAIM], msg[H_JUDGE:H_JUDGE + N_OTHERS], msg[H_TEAM],
                           msg[H_INTENT:H_INTENT + N_OTHERS])
    if not np.array_equal(rebuilt, msg):
        raise ValueError("the rationale heads of a free message must be the derived ones")


def free_message(game, speaker: int, claim: int = 0, judgments=(), team: int = 0, intents=()) -> np.ndarray:
    """The message of a seat the rules do not bind (`AvalonGame.free_seats`), from any stance the heads can hold:
    `judgments` and `intents` are per relative seat 1, 2, ... (none for the seats left out). Only the rationale is
    still derived, from the stance and the record: a judgment the record does not support has none."""
    ctx = stack_contexts([message_context(game, speaker)])
    msg = np.full((1, N_MSG_HEADS), -1, dtype=np.int64)
    msg[0, [H_CLAIM, H_TEAM]] = claim, team
    for h0, values in ((H_JUDGE, judgments), (H_INTENT, intents)):
        msg[0, h0:h0 + N_OTHERS] = 0
        msg[0, h0:h0 + len(values)] = values
    stance = [H_CLAIM, *range(H_JUDGE, H_JUDGE + N_OTHERS), H_TEAM, *range(H_INTENT, H_INTENT + N_OTHERS)]
    sizes = np.array(MSG_HEAD_SIZES)[stance]
    talk_only = game.phase == Phase.EVIL_DISCUSS and (msg[0, [H_CLAIM, H_TEAM]].any() or msg[0, H_INTENT:].any())
    if (msg[0, stance] < 0).any() or (msg[0, stance] >= sizes).any() or talk_only or (
            team > 0 and not ctx["team_members"][0, team].any()):
        raise ValueError("a free message holds a value its heads do not have")
    for h in range(H_RTYPE, H_TEAM):
        ok = head_mask(h, msg, ctx)[0]
        msg[0, h] = np.flatnonzero(ok)[0] if ok.any() else -1
    return msg[0].astype(np.int8)


# ---------------------------------------------------------------------- observation features
# Per message, in absolute target seats inside the game; rotated into the observer's frame by `encode_comm`.
_MSG_FIELDS = [
    ("spoke", 1),
    ("claim", len(CLAIM_VALUES) - 1),
    ("judge", N_PLAYERS * (len(JUDGE_VALUES) - 1)),
    ("rtype", N_PLAYERS * (len(RATIONALE_TYPES) - 1)),
    ("rround", N_PLAYERS * N_QUESTS),
    ("team_flag", 1),
    ("team", N_PLAYERS),
    ("intent", N_PLAYERS * (len(INTENT_VALUES) - 1)),
]
MSG_LAYOUT: dict[str, slice] = {}
_o = 0
for _name, _size in _MSG_FIELDS:
    MSG_LAYOUT[_name] = slice(_o, _o + _size)
    _o += _size
MSG_FEAT_DIM = _o
_SEAT_FIELDS = {"judge": len(JUDGE_VALUES) - 1, "rtype": len(RATIONALE_TYPES) - 1, "rround": N_QUESTS, "team": 1,
                "intent": len(INTENT_VALUES) - 1}
N_MSG_SLOTS = N_QUESTS * N_PLAYERS + N_PLAYERS  # one discussion per quest x speaker, then the evil talk by speaker

_COMM_FIELDS = [
    ("slot_open", len(SLOTS)),
    ("discussion", 2),  # in DISCUSS, in EVIL_DISCUSS
    ("spoken", N_PLAYERS),  # who has already spoken in the current discussion
    ("audit", N_PLAYERS * len(AUDIT) * AUDIT_LEVELS),  # per player: word-against-deed counts, as thresholds
]
COMM_LAYOUT: dict[str, slice] = {}
_o = 0
for _name, _size in _COMM_FIELDS:
    COMM_LAYOUT[_name] = slice(_o, _o + _size)
    _o += _size
COMM_GLOBAL_DIM = _o
COMM_LAYOUT["messages"] = slice(COMM_GLOBAL_DIM, COMM_GLOBAL_DIM + N_MSG_SLOTS * MSG_FEAT_DIM)
COMM_DIM = COMM_LAYOUT["messages"].stop


def _feature_perm(observer: int) -> np.ndarray:
    """idx[j] = absolute-frame feature behind the observer-relative feature j."""
    idx: list[int] = []
    for name, size in _MSG_FIELDS:
        start, per_seat = MSG_LAYOUT[name].start, _SEAT_FIELDS.get(name)
        if per_seat is None:
            idx += range(start, start + size)
            continue
        for rel in range(N_PLAYERS):
            seat = (observer + rel) % N_PLAYERS
            idx += range(start + seat * per_seat, start + (seat + 1) * per_seat)
    return np.array(idx)


_FEATURE_PERM = [_feature_perm(o) for o in range(N_PLAYERS)]


def message_features(speaker: int, msg: np.ndarray, team_size: int) -> np.ndarray:
    """Absolute-frame features of one message."""
    f = np.zeros(MSG_FEAT_DIM, dtype=np.int8)
    f[MSG_LAYOUT["spoke"]] = 1
    if msg[H_CLAIM] > 0:
        f[MSG_LAYOUT["claim"].start + msg[H_CLAIM] - 1] = 1
    for r, seat in enumerate(_others(speaker)):
        if msg[H_JUDGE + r] > 0:
            f[MSG_LAYOUT["judge"].start + _SEAT_FIELDS["judge"] * seat + msg[H_JUDGE + r] - 1] = 1
        if msg[H_RTYPE + r] > 0:
            f[MSG_LAYOUT["rtype"].start + _SEAT_FIELDS["rtype"] * seat + msg[H_RTYPE + r] - 1] = 1
        if msg[H_RROUND + r] >= 0:
            f[MSG_LAYOUT["rround"].start + N_QUESTS * seat + msg[H_RROUND + r]] = 1
        if msg[H_INTENT + r] > 0:
            f[MSG_LAYOUT["intent"].start + _SEAT_FIELDS["intent"] * seat + msg[H_INTENT + r] - 1] = 1
    if msg[H_TEAM] > 0:
        f[MSG_LAYOUT["team_flag"]] = 1
        f[MSG_LAYOUT["team"].start + np.array(A.decode_team(speaker, msg[H_TEAM] - 1, team_size))] = 1
    return f


def encode_comm(game, player: int) -> np.ndarray:
    """Communication part of the observation (int8, length COMM_DIM), in the player's relative frame."""
    out = np.zeros(COMM_DIM, dtype=np.int8)
    cfg = game.comm
    out[COMM_LAYOUT["slot_open"]] = [cfg.is_open(s) for s in SLOTS]
    if not cfg.enabled:
        return out
    perm = (player + np.arange(N_PLAYERS)) % N_PLAYERS
    if game.in_discussion:
        out[COMM_LAYOUT["discussion"].start + (game.phase == Phase.EVIL_DISCUSS)] = 1
        for seat in game.speakers[:game.speak_idx]:
            out[COMM_LAYOUT["spoken"].start + (seat - player) % N_PLAYERS] = 1
    levels = game.audit[perm][:, :, None] >= np.arange(1, AUDIT_LEVELS + 1)
    out[COMM_LAYOUT["audit"]] = levels.reshape(-1)
    slots = np.concatenate([game.msg_feat[:, perm].reshape(-1, MSG_FEAT_DIM), game.evil_feat[perm]])
    out[COMM_LAYOUT["messages"]] = slots[:, _FEATURE_PERM[player]].reshape(-1)
    return out


def decode_message(msg: np.ndarray, speaker: int, team_size: int) -> dict:
    """Readable form of a message in absolute seats (for game records)."""
    out: dict = {"speaker": speaker}
    if msg[H_CLAIM] > 0:
        out["claim"] = CLAIM_VALUES[msg[H_CLAIM]]
    judged = {}
    for r, seat in enumerate(_others(speaker)):
        if msg[H_JUDGE + r] > 0:
            entry = {"judgment": JUDGE_VALUES[msg[H_JUDGE + r]]}
            if msg[H_RTYPE + r] > 0:
                entry["basis"] = RATIONALE_TYPES[msg[H_RTYPE + r]]
                if msg[H_RROUND + r] >= 0:
                    entry["quest"] = int(msg[H_RROUND + r]) + 1
            judged[int(seat)] = entry
    if judged:
        out["judgments"] = judged
    if msg[H_TEAM] > 0:
        out["team"] = [int(s) for s in A.decode_team(speaker, msg[H_TEAM] - 1, team_size)]
    intents = {int(seat): INTENT_VALUES[msg[H_INTENT + r]]
               for r, seat in enumerate(_others(speaker)) if msg[H_INTENT + r] > 0}
    if intents:
        out["intents"] = intents
    return out
