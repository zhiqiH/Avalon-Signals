import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.agents import ActorCritic, PolicyController, RandomController, Request, RuleController
from src.agents.policy import load_silent_weights
from src.comm import protocol as P
from src.config import load_config
from src.env import AvalonGame, Phase, Role, encode
from src.env import actions as A
from src.runner import play
from src.training import TrajectoryRecorder
from src.training.checkpoint import load_checkpoint, load_policies
from src.training.trainer import train
from helpers import ROLES, make_game, quest, vote

PILOT = Path(__file__).resolve().parents[1] / "configs" / "pilot_silent.yaml"
FIVE = P.CommConfig(enabled=True)
M = P.CommConfig(enabled=True, claim=False, team=False, intent=False)
T = P.CommConfig(enabled=True, team=False)  # no teams named: scripts in which the leaders propose freely
SEVEN = P.CommConfig(enabled=True, reply=True, promise=True)  # with replies and promises
OPT = P.CommConfig(enabled=True, reply=True, promise=True, team_optional=True)  # and team suggestions as a choice (configs/comm.yaml)


def head_index(name: str) -> int:
    if name in ("claim", "team"):
        return getattr(P, f"H_{name.upper()}")
    return getattr(P, f"H_{name[:-1].upper()}") + int(name[-1]) - 1  # judge3 -> judgment of relative seat 3


def build(game, **heads) -> np.ndarray:
    """The current speaker's message: the given heads (relative seats), the first legal value elsewhere."""
    ctx = P.stack_contexts([P.message_context(game, game.acting_players()[0])])
    given = {head_index(k): v for k, v in heads.items()}
    msg = np.full((1, P.N_MSG_HEADS), -1, dtype=np.int64)
    for h in P.STAGE_A + P.STAGE_B:
        ok = P.head_mask(h, msg, ctx)[0]
        msg[0, h] = given[h] if h in given else (np.flatnonzero(ok)[0] if ok.any() else -1)
    return msg[0].astype(np.int8)


def say(game, **heads) -> None:
    game.step({game.acting_players()[0]: build(game, **heads)})


def mask_of(game, h: int, **heads) -> np.ndarray:
    """Legal values of head h for the current speaker after the given earlier heads."""
    msg = build(game, **heads).astype(np.int64)[None]
    msg[0, [x for x in range(P.N_MSG_HEADS) if x >= h]] = -1
    for k, v in heads.items():
        if head_index(k) < h:
            msg[0, head_index(k)] = v
    return P.head_mask(h, msg, P.stack_contexts([P.message_context(game, game.acting_players()[0])]))[0]


def discuss(game) -> None:
    while game.in_discussion:
        say(game)


def propose(game, seats) -> None:
    game.step({game.leader: A.team_action(game.leader, seats)})


def play_quest(game, seats, approvers=range(6), failers=()) -> None:
    discuss(game)
    propose(game, seats)
    vote(game, approvers)
    quest(game, failers)


def test_each_quest_opens_with_one_discussion_leader_last():
    game = make_game(leader=2, comm=FIVE)
    assert game.phase == Phase.DISCUSS
    order = []
    while game.in_discussion:
        order.append(game.acting_players()[0])
        say(game)
    assert order == [3, 4, 5, 0, 1, 2] and game.phase == Phase.PROPOSE and game.acting_players() == [2]
    propose(game, [2, 3])
    vote(game, [2])  # rejected (a leader approves its own team): the next leader proposes at once, without discussion
    assert game.phase == Phase.PROPOSE and game.acting_players() == [3]
    propose(game, [3, 4])
    vote(game, range(6))
    quest(game)
    assert game.phase == Phase.DISCUSS and game.speakers == [5, 0, 1, 2, 3, 4]


def test_closed_slots_can_only_say_nothing():
    game = make_game(comm=M)
    play_quest(game, [0, 1])  # a success everyone approved: a record for every good judgment
    ctx = P.message_context(game, 2)
    assert ctx["claim_ok"].tolist() == [True, False, False, False]
    assert ctx["team_ok"].sum() == 1 and not ctx["intent"]
    assert ctx["judge_ok"][:, [P.J_NONE, P.J_GOOD]].all() and not ctx["judge_ok"][:, P.J_EVIL:].any()


def test_a_judgment_needs_a_record_and_the_first_round_is_for_claims_suggestions_and_intents():
    game = make_game(leader=0, comm=FIVE)  # seat 1 speaks first; nothing is on record yet
    assert mask_of(game, P.H_CLAIM).tolist() == [True, True, True, False]  # nobody claims Merlin
    for claim in (P.C_NONE, P.C_SERVANT, P.C_PERCIVAL):
        assert np.flatnonzero(mask_of(game, P.H_JUDGE, claim=claim)).tolist() == [P.J_NONE]
    assert (mask_of(game, P.H_TEAM) == P.TEAM_MEMBERS[2][:, 0]).all()  # a team is named: any team with the speaker
    assert mask_of(game, P.H_INTENT)[P.I_REQUIRE] and mask_of(game, P.H_INTENT + 1)[P.I_VETO]  # in line with that team
    game = make_game(leader=0, comm=M)  # without those slots the first discussion is empty: nobody is asked
    assert P.forced_message(P.message_context(game, 1)) is not None
    play_quest(game, [0, 1], approvers=[0, 1, 2, 4, 5])  # seat 3 rejects a success; seat 2 approves it
    speaker = game.acting_players()[0]
    assert speaker == 2 and P.forced_message(P.message_context(game, 2)) is None
    assert np.flatnonzero(mask_of(game, P.H_JUDGE)).tolist() == [P.J_NONE, P.J_EVIL]  # about seat 3
    # Seat 0 was on the successful quest, led it and approved it.
    assert np.flatnonzero(mask_of(game, P.H_JUDGE + 3)).tolist() == [P.J_NONE, P.J_GOOD]


def test_judgments_respect_the_role_deck():
    game = make_game(leader=0, comm=T)
    play_quest(game, [0, 1])  # two quests everyone approved, a success and a failure:
    play_quest(game, [0, 1, 4], failers=[4])  # every judgment of every player now has a record
    assert game.acting_players() == [3]
    ok = mask_of(game, P.H_JUDGE + 2, judge1=P.J_EVIL, judge2=P.J_EVIL)
    assert np.flatnonzero(ok).tolist() == [P.J_NONE, P.J_GOOD]  # two evil players, no third
    three_good = dict(judge1=P.J_GOOD, judge2=P.J_GOOD, judge3=P.J_GOOD)
    assert mask_of(game, P.H_JUDGE + 3, **three_good)[P.J_GOOD]
    ok = mask_of(game, P.H_JUDGE + 3, claim=P.C_SERVANT, **three_good)
    assert np.flatnonzero(ok).tolist() == [P.J_NONE, P.J_EVIL]  # four good players, the speaker's claim included


def test_merlin_cannot_accuse_a_player_he_knows_to_be_good():
    game = make_game(leader=0, comm=T)
    play_quest(game, [0, 1])
    play_quest(game, [0, 1, 4], failers=[4])  # a record against everybody
    merlin = P.message_context(game, 0)["judge_ok"]  # about seats 1-5; seats 4 and 5 are evil
    assert merlin[:, P.J_EVIL].tolist() == [False, False, False, True, True] and merlin[:, P.J_GOOD].all()
    assert P.message_context(game, 2)["judge_ok"][:, P.J_EVIL].all()  # a Servant can be wrong about anybody


def test_merlin_may_accuse_a_good_player_where_configured():
    bluff = P.CommConfig(enabled=True, team=False, merlin_may_accuse_good=True)
    game = make_game(leader=0, comm=bluff)
    play_quest(game, [0, 1])
    play_quest(game, [0, 1, 4], failers=[4])  # a record against everybody
    assert P.message_context(game, 0)["judge_ok"][:, P.J_EVIL].all()  # Merlin judges like a Servant
    for flag in (False, True):  # ... and promises like one: in round 1 about anybody, also seats he knows are good
        cfg = P.CommConfig(enabled=True, reply=True, promise=True, team_optional=True, merlin_may_accuse_good=flag)
        ok = P.message_context(make_game(leader=1, comm=cfg), 0)["promise_ok"][:, P.K_FAIL_EVIL]
        assert ok.tolist() == ([True] * 5 if flag else [False, False, False, True, True])


def test_a_forced_fifth_proposal_leaves_no_vote_record():
    game = make_game(leader=0, comm=T)
    discuss(game)
    for _ in range(4):
        propose(game, [2, 3])
        vote(game, [game.leader])
    propose(game, [0, 4])  # the fifth proposal: good players must approve it
    vote(game, range(6))
    quest(game, failers=[4])
    ev = P.evidence(game, 2)  # about seats 3, 4, 5, 0, 1
    assert not ev[:, 1].any()  # no vote record
    assert ev[:, 0, 1, 0].tolist() == [False, True, False, True, False]  # the quest record stands


def scripted_history(game) -> None:
    """Quest 1 (leader 0): seats 0, 1 succeed, all approve. Quest 2 (leader 1): seats 2, 3, 4; seat 4 fails;
    approvers 0, 2, 4, 5. Quest 3 opens with leader 2, so seat 3 speaks first."""
    play_quest(game, [0, 1])
    play_quest(game, [2, 3, 4], approvers=[0, 2, 4, 5], failers=[4])


def test_rationale_points_the_same_way_as_the_judgment():
    game = make_game(leader=0, comm=T, binding=False)  # the leader of quest 2 rejects its own team
    scripted_history(game)
    assert game.acting_players() == [3]
    r = {seat: (seat - 3) % 6 for seat in range(6) if seat != 3}  # relative seats of speaker 3
    ev = P.evidence(game, 3)
    quest_k, vote_k, prop_k = 0, 1, 2
    assert ev[r[4] - 1, quest_k, 1].tolist() == [False, True, False, False, False]  # on the failed quest 2
    assert ev[r[0] - 1, quest_k, 0].tolist() == [True, False, False, False, False]  # on the successful quest 1
    assert ev[r[0] - 1, vote_k, 1, 1] and ev[r[1] - 1, vote_k, 0, 1]  # 0 approved the failure, 1 rejected it
    assert ev[r[0] - 1, prop_k, 0, 0] and ev[r[1] - 1, prop_k, 1, 1]  # 0 led the success, 1 led the failure

    seat4 = r[4]
    evil = mask_of(game, P.H_RTYPE + seat4 - 1, **{f"judge{seat4}": P.J_EVIL})
    assert np.flatnonzero(evil).tolist() == [P.R_QUEST]  # derived: the quest record before any vote record
    good = mask_of(game, P.H_RTYPE + seat4 - 1, **{f"judge{seat4}": P.J_GOOD})
    assert np.flatnonzero(good).tolist() == [P.R_VOTE]  # only its approval of the successful quest 1 speaks for it
    seat5 = r[5]  # approved the failed quest 2 after the successful quest 1: the vote that points this way
    assert np.flatnonzero(mask_of(game, P.H_RTYPE + seat5 - 1, **{f"judge{seat5}": P.J_EVIL})).tolist() == [P.R_VOTE]
    rounds = mask_of(game, P.H_RROUND + seat5 - 1, **{f"judge{seat5}": P.J_EVIL, f"rtype{seat5}": P.R_VOTE})
    assert rounds.tolist() == [False, True, False, False, False]
    seat1 = r[1]  # rejected the failed quest 2: nothing speaks against it but leading that quest
    assert np.flatnonzero(mask_of(game, P.H_RTYPE + seat1 - 1, **{f"judge{seat1}": P.J_EVIL})).tolist() == [P.R_PROPOSAL]
    rounds = mask_of(game, P.H_RROUND + seat4 - 1, **{f"judge{seat4}": P.J_GOOD, f"rtype{seat4}": P.R_VOTE})
    assert rounds.tolist() == [True, False, False, False, False]
    assert not mask_of(game, P.H_RTYPE + r[1] - 1).any()  # no judgment, no rationale

    bad = build(game, **{f"judge{seat4}": P.J_GOOD, f"rtype{seat4}": P.R_QUEST, f"rround{seat4}": 1})
    with pytest.raises(ValueError):
        P.check_message(game, 3, bad)


def named_teams(game, **heads) -> list[list[int]]:
    """The teams the current speaker may name after the given heads, in absolute seats."""
    speaker, ok = game.acting_players()[0], mask_of(game, P.H_TEAM, **heads)
    assert not ok[0]  # a team is named whenever one is open
    return sorted(list(A.decode_team(speaker, int(v) - 1, game.team_size)) for v in np.flatnonzero(ok))


def test_suggested_and_proposed_teams_are_among_the_cleanest_by_the_record():
    game = make_game(leader=0, comm=FIVE)
    assert named_teams(game) == [[0, 1], [1, 2], [1, 3], [1, 4], [1, 5]]  # seat 1, nothing on record: any team with itself
    while game.acting_players() != [0]:
        say(game)
    assert np.flatnonzero(mask_of(game, P.H_TEAM)).tolist() == [0]  # the leader suggests nothing: its proposal follows
    say(game)
    with pytest.raises(ValueError):
        propose(game, [1, 4])  # a leader is on its own team
    propose(game, [0, 4])
    assert bound_vote(game, 0) == "approve"
    vote(game, range(6))
    quest(game, failers=[4])
    # Round 2 (three seats), leader 1; seat 2 speaks first. The record points at seats 0 and 4 (the failed quest).
    assert game.acting_players() == [2]
    assert named_teams(game) == [[1, 2, 3], [1, 2, 5], [2, 3, 5]]  # no team with seat 0 or 4 while cleaner ones are open
    assert named_teams(game, judge5=P.J_EVIL) == [[2, 3, 5]]  # nor with seat 1, once it accuses it (its vote for the failure)
    say(game, judge5=P.J_EVIL)
    say(game, team=1 + A.TEAM_INDEX[3][(0, 2, 5)])  # seat 3 names seats 3, 5, 2
    while game.acting_players() != [0]:
        say(game)
    # Seat 0 knows itself good, so seat 4 failed the quest: it is out whatever else is on record.
    assert all(4 not in team and 0 in team for team in named_teams(game)) and len(named_teams(game)) == 6
    discuss(game)
    legal = lambda p: sorted(list(A.decode_team(p, a, 3)) for a in np.flatnonzero(game.legal_mask(p)))  # noqa: E731
    assert legal(1) == [[1, 2, 3], [1, 2, 5], [1, 3, 5]]  # the leader's proposal follows the same rule
    with pytest.raises(ValueError):
        propose(game, [0, 1, 2])
    propose(game, [1, 2, 3])
    assert bound_vote(game, 2) == "reject"  # seat 2 accused the leader
    vote(game, [1])
    assert legal(2) == [[2, 3, 5]]  # leader 2: the cleanest team without the player it accuses
    propose(game, [2, 3, 5])  # the very team seats 2 and 3 suggested
    assert bound_vote(game, 2) == "approve" and bound_vote(game, 3) == "approve"


def test_intents_cannot_contradict_judgments_or_ones_own_suggestion():
    game = make_game(leader=0, comm=FIVE)
    team = 1 + A.TEAM_INDEX[2][(0, 1)]  # seat 1 suggests itself and seat 2
    assert not mask_of(game, P.H_INTENT, judge1=P.J_GOOD)[P.I_VETO]
    assert not mask_of(game, P.H_INTENT, judge1=P.J_EVIL)[P.I_REQUIRE]
    member = mask_of(game, P.H_INTENT, team=team)  # seat 2, in the suggestion
    assert member[P.I_REQUIRE] and not member[P.I_VETO]
    outsider = mask_of(game, P.H_INTENT + 2, team=team)  # seat 4
    assert outsider[P.I_VETO] and not outsider[P.I_REQUIRE]
    assert P.INTENT_VALUES == ("none", "veto", "require")


def test_vote_intents_must_be_jointly_satisfiable():
    game = make_game(leader=0, comm=T)  # quest 1 sends two players
    third = mask_of(game, P.H_INTENT + 2, intent1=P.I_REQUIRE, intent2=P.I_REQUIRE)
    assert not third[P.I_REQUIRE] and third[P.I_VETO]  # seats 2 and 3 already fill the team
    last = mask_of(game, P.H_INTENT + 4, intent1=P.I_REQUIRE, intent2=P.I_VETO, intent3=P.I_VETO, intent4=P.I_VETO)
    assert last[[P.I_REQUIRE, P.I_VETO]].all()  # seat 2 with seat 0 (require) or with the speaker (veto)
    with pytest.raises(ValueError):
        say(game, intent1=P.I_REQUIRE, intent2=P.I_REQUIRE, intent3=P.I_REQUIRE)


def speak_as(game, seat: int, **heads) -> None:
    """Let the others say nothing until `seat` speaks, then say the given heads."""
    while game.acting_players() != [seat]:
        say(game)
    say(game, **heads)


def audit_seen_by(game, observer: int) -> np.ndarray:
    """(relative seat, audit kind): the public count as the observation shows it (capped at AUDIT_LEVELS)."""
    comm = P.encode_comm(game, observer)
    old = comm[P.COMM_LAYOUT["audit"]].reshape(6, P.LEGACY_AUDIT, P.AUDIT_LEVELS)
    new = comm[P.COMM_LAYOUT["audit_x"]].reshape(6, len(P.AUDIT) - P.LEGACY_AUDIT, P.AUDIT_LEVELS)
    return np.concatenate([old, new], axis=1).sum(axis=-1)


def bound_vote(game, player: int) -> str:
    mask = game.legal_mask(player)
    return "approve" if not mask[A.REJECT] else "reject" if not mask[A.APPROVE] else "free"


def test_words_bind_the_speakers_own_actions():
    game = make_game(leader=0, comm=T)
    play_quest(game, [0, 1], approvers=[0, 1, 2, 4, 5])  # seat 3 rejects a success: the record against it
    # Round 2 (three seats), leader 1.
    speak_as(game, 2, judge1=P.J_EVIL)  # seat 2 accuses seat 3
    speak_as(game, 5, intent1=P.I_REQUIRE, intent4=P.I_VETO)  # seat 5: only teams with seat 0 and without seat 3
    speak_as(game, 0, judge1=P.J_GOOD, judge2=P.J_GOOD)  # seat 0 vouches for seats 1 and 2
    speak_as(game, 1, judge2=P.J_EVIL)  # the leader accuses seat 3 ...
    members = P.TEAM_MEMBERS[3][1:21]  # the twenty teams of three, in relative seats
    teams = game.legal_mask(1)[:20]
    assert not teams[members[:, 2]].any() and teams[~members[:, 2]].all()  # ... and cannot propose it
    with pytest.raises(ValueError):
        propose(game, [1, 2, 3])
    propose(game, [0, 1, 2])  # the leader backs its own team; seat 0 vouched for both others on it
    assert [bound_vote(game, p) for p in range(6)] == ["approve", "approve", "free", "free", "free", "free"]
    vote(game, [0, 1])
    propose(game, [4, 5, 0])  # leader 2; seat 5's conditions are met
    assert [bound_vote(game, p) for p in range(6)] == ["free", "free", "approve", "free", "free", "free"]
    vote(game, [2, 4])
    propose(game, [3, 4, 5])  # leader 3: seats 1 and 2 accused it, seat 5 vetoed it and asked for seat 0
    assert [bound_vote(game, p) for p in range(6)] == ["free", "reject", "reject", "approve", "free", "reject"]
    vote(game, [3])
    propose(game, [1, 2, 4])  # leader 4: without seat 0; seat 0 did not vouch for seat 4
    assert [bound_vote(game, p) for p in range(6)] == ["free", "free", "free", "free", "approve", "reject"]
    vote(game, [4])
    teams = game.legal_mask(5)[:20]  # the fifth proposal: its leader is still bound by its words ...
    assert (teams == (members[:, 1] & ~members[:, 4])).all()
    propose(game, [5, 0, 1])  # ... the voters are not (good players must approve it anyway)
    assert [bound_vote(game, p) for p in range(6)] == ["approve"] * 4 + ["free", "approve"]
    vote(game, range(6))
    assert not game.audit.any()  # nothing to count: nobody could act against its words
    quest(game, failers=[5])
    # Round 3 (four seats), leader 0: silence keeps an accusation, a new list of suspects replaces it.
    speak_as(game, 2, judge3=P.J_EVIL)  # seat 2 now accuses seat 5 (on the failed quest) and no longer seat 3
    discuss(game)
    assert np.flatnonzero(P.accused_by(game, 1)).tolist() == [3] and np.flatnonzero(P.accused_by(game, 2)).tolist() == [5]
    propose(game, [0, 1, 2, 3])
    assert bound_vote(game, 1) == "reject" and bound_vote(game, 2) == "free"


def test_without_the_binding_deeds_against_ones_words_are_counted_in_public():
    game = make_game(leader=0, comm=FIVE, binding=False)
    play_quest(game, [0, 1], approvers=[0, 1, 2, 4, 5])  # seat 3 rejects a success: the record against it
    speak_as(game, 2, judge1=P.J_EVIL, team=1 + A.TEAM_INDEX[3][(0, 4, 5)])  # seat 2 accuses seat 3; names 2, 0, 1
    speak_as(game, 4, team=1 + A.TEAM_INDEX[3][(0, 1, 2)])  # seat 4 names seats 4, 5, 0
    speak_as(game, 0, team=1 + A.TEAM_INDEX[3][(0, 1, 2)], intent3=P.I_VETO)  # seat 0: 0, 1, 2; "no team with seat 3"
    speak_as(game, 1, judge2=P.J_EVIL)  # the leader accuses seat 3, then proposes it
    propose(game, [1, 2, 3])
    assert all(game.legal_mask(p)[[A.APPROVE, A.REJECT]].all() for p in range(6))
    vote(game, [0, 2])
    expected = np.zeros((6, len(P.AUDIT)), dtype=int)
    expected[1, P.A_JUDGE] = expected[2, P.A_JUDGE] = expected[0, P.A_INTENT] = 1
    assert game.audit.tolist() == expected.tolist()
    propose(game, [4, 5, 0])
    vote(game, [0, 5])  # seat 4 rejects the very team it named
    expected[4, P.A_TEAM] = 1
    assert game.audit.tolist() == expected.tolist()
    for _ in range(2):
        propose(game, [3, 4, 5])  # nobody's team
        vote(game, [])
    propose(game, [1, 2, 3])  # the fifth proposal: approving it is not held against anyone
    vote(game, range(6))
    assert game.audit.tolist() == expected.tolist()
    quest(game)
    seen = audit_seen_by(game, 1)  # relative seats of observer 1: seat 2 is 1, seat 4 is 3, seat 0 is 5
    assert seen[0, P.A_JUDGE] == 1 and seen[1, P.A_JUDGE] == 1 and seen[3, P.A_TEAM] == 1 and seen[5, P.A_INTENT] == 1
    assert seen.sum() == 4


def test_a_free_seat_is_not_bound_and_its_deeds_against_its_words_are_counted():
    game = make_game(leader=0, comm=FIVE)
    game.free_seats = frozenset({2})  # a human in seat 2
    say(game)
    loose = P.free_message(game, 2, judgments=[P.J_EVIL])  # accuses seat 3 with no record, names no team
    assert loose[P.H_RTYPE] == P.R_NONE and loose[P.H_TEAM] == 0
    with pytest.raises(ValueError):
        P.check_message(game, 2, loose)  # nobody else could say this
    with pytest.raises(ValueError):
        P.free_message(game, 2, team=1 + len(A.TEAMS_BY_SIZE[2]))  # no such team of two
    game.step({2: loose})
    discuss(game)
    propose(game, [0, 3])
    assert bound_vote(game, 2) == "free"
    game.free_seats = frozenset()
    assert bound_vote(game, 2) == "reject"  # the same words would bind any other seat
    game.free_seats = frozenset({2})
    vote(game, [0, 2])  # it approves a team with the player it accused
    assert game.audit[2].tolist() == [1, 0, 0, 0, 0, 0] and game.audit.sum() == 1
    assert P.public_suspicion(game)[2] == P.BROKEN_WORD
    assert game.legal_mask(1)[:15].sum() < 15  # the next leader, like every other seat, is still bound


def test_a_claim_stands():
    game = make_game(leader=0, comm=T)
    speak_as(game, 1, claim=P.C_PERCIVAL)
    play_quest(game, [0, 1])
    while game.acting_players() != [1]:
        say(game)
    assert mask_of(game, P.H_CLAIM).tolist() == [True, False, True, False]


def test_vouching_for_a_member_of_a_failed_team_is_counted():
    game = make_game(leader=0, comm=T)
    play_quest(game, [0, 1])  # a success everyone approved: a record for vouching for anybody
    speak_as(game, 2, judge2=P.J_GOOD, judge5=P.J_GOOD)  # seat 2 vouches for seats 4 and 1
    speak_as(game, 3, judge1=P.J_GOOD)  # seat 3 vouches for seat 4
    discuss(game)
    propose(game, [1, 4, 5])
    vote(game, range(6))
    quest(game, failers=[4])
    assert game.audit[:, P.A_VOUCH].tolist() == [0, 0, 2, 1, 0, 0]
    speak_as(game, 3, judge1=P.J_EVIL)  # seat 3 turns on seat 4 (the failed quest is its new record) ...
    discuss(game)
    propose(game, [1, 2, 4, 5])
    vote(game, [0, 1, 2, 4, 5])  # ... and so has to reject this team
    quest(game, failers=[4])
    assert game.audit[:, P.A_VOUCH].tolist() == [0, 0, 4, 1, 0, 0]  # seat 2 said nothing new: its word stands
    assert make_game().audit.shape == (6, len(P.AUDIT))  # the silent game keeps the counts at zero


def test_silent_games_have_no_message_rules():
    game = make_game(leader=0)
    assert game.legal_mask(0)[:15].all() and not game.legal_mask(0)[15:20].any()
    propose(game, [0, 3])
    assert all(game.legal_mask(p)[[A.APPROVE, A.REJECT]].all() for p in range(6))


def accuse_seat_3_in_round_2(game) -> None:
    """Quest 1 (seats 3, 4) fails; seat 3 rejected it: an evil quest record and a good vote record. In round 2
    seat 1, the leader, calls seat 3 evil."""
    discuss(game)
    propose(game, [3, 4])
    vote(game, [0, 1, 2, 4, 5])
    quest(game, failers=[4])
    speak_as(game, 1, judge2=P.J_EVIL)


def test_a_change_of_mind_needs_a_record_resolved_since_the_last_judgment():
    game = make_game(leader=0, comm=T)
    accuse_seat_3_in_round_2(game)
    propose(game, [0, 1, 2])
    vote(game, [0, 1, 2, 4, 5])  # seat 3 rejects the successful quest 2: an evil vote record, no good one
    quest(game)
    while game.acting_players() != [1]:
        say(game)
    ctx = P.message_context(game, 1)
    assert ctx["prev_dir"].tolist() == [-1, 1, -1, -1, -1]
    assert ctx["prev_accused"].tolist() == [False, True, False, False, False]
    judge = mask_of(game, P.H_JUDGE + 1)
    assert np.flatnonzero(judge).tolist() == [P.J_NONE, P.J_EVIL]  # the only good record predates the accusation
    with pytest.raises(ValueError):
        P.check_message(game, 1, build(game, judge2=P.J_GOOD, rtype2=P.R_VOTE, rround2=0))
    # Saying nothing about seat 3 keeps the accusation: no "require" for it.
    assert not mask_of(game, P.H_INTENT + 1)[P.I_REQUIRE]
    say(game, judge2=P.J_EVIL)


def test_a_change_of_mind_must_cite_the_new_record():
    game = make_game(leader=0, comm=T)
    accuse_seat_3_in_round_2(game)
    propose(game, [0, 1, 2])
    vote(game, range(6))  # seat 3 approves the successful quest 2: a new good record
    quest(game)
    while game.acting_players() != [1]:
        say(game)
    assert mask_of(game, P.H_JUDGE + 1)[[P.J_GOOD, P.J_EVIL]].all()
    assert np.flatnonzero(mask_of(game, P.H_RTYPE + 1, judge2=P.J_GOOD)).tolist() == [P.R_VOTE]
    rounds = mask_of(game, P.H_RROUND + 1, judge2=P.J_GOOD, rtype2=P.R_VOTE)
    assert rounds.tolist() == [False, True, False, False, False]  # quest 2, not the older vote on quest 1
    with pytest.raises(ValueError):
        P.check_message(game, 1, build(game, judge2=P.J_GOOD, rtype2=P.R_VOTE, rround2=0))
    say(game, judge2=P.J_GOOD)
    assert not P.accused_by(game, 1).any()  # calling the suspect good clears the accusation


def test_messages_are_encoded_in_the_observers_frame():
    game = make_game(leader=0, comm=FIVE)
    play_quest(game, [0, 1], approvers=[0, 1, 2, 3, 5])  # seat 4 rejects a success: the record against it
    team = 1 + A.TEAM_INDEX[3][(0, 4, 5)]  # seats 2, 0, 1: the cleanest team by the record
    say(game, claim=P.C_SERVANT, judge2=P.J_EVIL, team=team, intent2=P.I_VETO)  # seat 2 speaks first, about seat 4
    comm = P.encode_comm(game, 3)
    f = comm[P.COMM_LAYOUT["messages"]].reshape(P.N_MSG_SLOTS, P.MSG_FEAT_DIM)[6 + 5]  # quest 2, seat 2 = relative 5
    L, per = P.MSG_LAYOUT, P._SEAT_FIELDS
    assert f[L["spoke"]][0] == 1 and np.flatnonzero(f[L["claim"]]).tolist() == [P.C_SERVANT - 1]
    assert np.flatnonzero(f[L["judge"]]).tolist() == [per["judge"] * 1 + P.J_EVIL - 1]  # seat 4 = relative 1
    assert np.flatnonzero(f[L["rtype"]]).tolist() == [per["rtype"] * 1 + P.R_VOTE - 1]
    assert np.flatnonzero(f[L["rround"]]).tolist() == [P.N_QUESTS * 1 + 0]
    assert np.flatnonzero(f[L["team"]]).tolist() == [3, 4, 5]
    assert np.flatnonzero(f[L["intent"]]).tolist() == [per["intent"] * 1 + P.I_VETO - 1]
    assert np.flatnonzero(comm[P.COMM_LAYOUT["spoken"]]).tolist() == [5]
    assert game.to_record()["discussions"][1] == {"quest": 1, "leader": 1, "messages": [{
        "speaker": 2, "claim": "servant", "judgments": {4: {"judgment": "evil", "basis": "vote", "quest": 1}},
        "team": [0, 1, 2], "intents": {4: "veto"}}]}


def test_messages_do_not_reveal_hidden_roles():
    roles_a = [Role.SERVANT, Role.MERLIN, Role.PERCIVAL, Role.SERVANT, Role.MORGANA, Role.ASSASSIN]
    roles_b = [Role.SERVANT, Role.ASSASSIN, Role.MERLIN, Role.MORGANA, Role.PERCIVAL, Role.SERVANT]
    games = [make_game(roles_a, comm=FIVE), make_game(roles_b, comm=FIVE)]
    for g in games:
        say(g, claim=P.C_SERVANT, intent2=P.I_VETO)
        say(g, claim=P.C_PERCIVAL, intent1=P.I_REQUIRE, intent2=P.I_VETO)
    assert np.array_equal(P.encode_comm(games[0], 0), P.encode_comm(games[1], 0))
    assert np.array_equal(encode(games[0], 0), encode(games[1], 0))


def test_evil_players_judge_merlin_together_before_the_assassination():
    game = make_game(comm=T)
    for _ in range(3):
        play_quest(game, list(range(game.team_size)))
    assert game.phase == Phase.EVIL_DISCUSS and game.acting_players() == [4]  # Morgana, then the Assassin
    ctx = P.message_context(game, 4)
    assert ctx["judge_ok"][0].tolist() == [True] + [False] * 7  # relative 1 is the Assassin
    assert np.flatnonzero(ctx["judge_ok"][1]).tolist() == [P.J_NONE, *P.EVIL_TALK_JUDGMENTS]
    assert not ctx["claim_ok"][1:].any() and not ctx["team_ok"][1:].any() and not ctx["intent"]
    assert not mask_of(game, P.H_JUDGE + 2, judge2=P.J_MERLIN)[P.J_MERLIN]  # the role deck: one Merlin
    say(game, judge2=P.J_MERLIN)  # Morgana: seat 0 is Merlin
    assert game.acting_players() == [5]
    # The Assassin names exactly one Merlin: silent about seats 0-2, it has to name seat 3 (relative seat 4).
    assert np.flatnonzero(mask_of(game, P.H_JUDGE + 3)).tolist() == [P.J_MERLIN]
    assert mask_of(game, P.H_JUDGE + 3, judge1=P.J_MERLIN).tolist()[:2] == [True, False]
    say(game, judge1=P.J_MERLIN)  # it agrees with Morgana ...
    assert game.phase == Phase.ASSASSINATE
    assert np.flatnonzero(game.legal_mask(5)).tolist() == [A.assassin_action(5, 0)]  # ... and strikes whom it named
    with pytest.raises(ValueError):
        game.step({5: A.assassin_action(5, 1)})
    slots = P.encode_comm(game, 5)[P.COMM_LAYOUT["messages"]].reshape(P.N_MSG_SLOTS, P.MSG_FEAT_DIM)
    morgana = slots[P.N_QUESTS * 6 + 5]  # evil talk, Morgana is relative seat 5 of the Assassin
    assert np.flatnonzero(morgana[P.MSG_LAYOUT["judge"]]).tolist() == [P._SEAT_FIELDS["judge"] * 1 + P.J_MERLIN - 1]
    game.step({5: A.assassin_action(5, 0)})
    assert game.done and not game.good_won
    assert game.to_record()["evil_talk"] == [
        {"speaker": 4, "judgments": {0: {"judgment": "merlin", "basis": "quest", "quest": 3}}},
        {"speaker": 5, "judgments": {0: {"judgment": "merlin", "basis": "quest", "quest": 3}}}]


def test_silenced_slots_look_like_saying_nothing():
    mute = make_game(comm=P.CommConfig(enabled=True, silence=list(P.SLOTS)))
    assert P.forced_message(P.message_context(mute, 1)) is not None  # nothing to choose: nobody is asked
    # A team is always named where the slot is open, so only the other slots can look like a choice of silence.
    silenced = make_game(comm=P.CommConfig(enabled=True, silence=[s for s in P.SLOTS if s != "team"]))
    reference = make_game(comm=FIVE)
    for game in (silenced, reference):
        for _ in range(3):
            assert P.forced_message(P.message_context(game, game.acting_players()[0])) is None
            say(game)
    assert P.encode_comm(silenced, 4)[P.COMM_LAYOUT["slot_open"]].all()
    assert np.array_equal(P.encode_comm(silenced, 4), P.encode_comm(reference, 4))


@pytest.mark.parametrize("cfg", [FIVE, M, SEVEN, OPT])
def test_scripted_players_finish_communication_games(cfg):
    rng = np.random.default_rng(0)
    rule, rand = RuleController(), RandomController()
    for good, evil in [(rule, rand), (rand, rand)]:
        games = [AvalonGame(rng, comm=cfg) for _ in range(50)]
        play(games, [good] * 50, [evil] * 50, rng)  # step() rejects illegal messages
        assert all(g.done for g in games)
    record = games[0].to_record()
    assert record["comm"]["alignment"] and record["discussions"][0]["quest"] == 0


def test_policy_messages_are_legal_and_logps_are_reproducible():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = ActorCritic(comm=True)
    with torch.no_grad():
        net.actor[-1].weight.mul_(100)  # make message heads far from uniform
        net.actor_cond.weight.normal_()  # exercise the conditioning of stage-B heads
    learner = PolicyController(net, learn=True)
    games = [AvalonGame(rng, comm=FIVE) for _ in range(16)]
    recorder = TrajectoryRecorder()
    play(games, [learner] * 16, [RandomController()] * 16, rng, on_decision=recorder)
    batch = recorder.build(games, gamma=1.0, lam=0.95)["good"]
    assert (batch.actions[:, 0] < 0).any() and (batch.actions[:, 0] >= 0).any()
    t = torch.as_tensor
    logp, _, value, _ = net.evaluate(t(batch.obs), t(batch.comm_obs), t(batch.masks), t(batch.actions))
    assert torch.allclose(logp, t(batch.logps), atol=1e-4)
    assert torch.allclose(value, t(batch.values), atol=1e-5)


def play_one_tick(games, rng) -> None:
    for g in games:
        if g.done:
            continue
        chosen = {}
        for p in g.acting_players():
            if g.in_discussion:
                chosen[p] = P.sample_messages(P.stack_contexts([P.message_context(g, p)]), rng)[0]
            else:
                chosen[p] = int(rng.choice(np.flatnonzero(g.legal_mask(p))))
        g.step(chosen)


def test_branch_from_silent_weights_plays_like_the_silent_network():
    torch.manual_seed(0)
    silent, comm = ActorCritic(), ActorCritic(comm=True)
    load_silent_weights(comm, silent.state_dict())
    rng = np.random.default_rng(1)
    games = [AvalonGame(rng, comm=FIVE) for _ in range(8)]
    for _ in range(40):  # advance through several discussions with random messages
        play_one_tick(games, rng)
        if sum(not g.done for g in games) <= 2:
            break
    reqs = [Request(i, g, p, g.legal_mask(p)) for i, g in enumerate(games) if not g.done for p in g.acting_players()]
    obs = torch.as_tensor(np.stack([encode(r.game, r.player) for r in reqs]))
    comm_obs = torch.as_tensor(np.stack([P.encode_comm(r.game, r.player) for r in reqs]))
    assert comm_obs[:, P.COMM_LAYOUT["messages"]].any()
    cond = torch.ones(len(reqs), P.COND_DIM)
    with torch.no_grad():
        logits = comm.logits(obs, comm_obs, cond)
        assert torch.allclose(logits[:, :A.N_ACTIONS], silent.logits(obs), atol=1e-6)
        assert torch.all(logits[:, A.N_ACTIONS:] == 0)  # every message head starts uniform
        assert torch.allclose(comm.value(obs, comm_obs), silent.value(obs), atol=1e-6)


def test_short_communication_run_branches_from_a_silent_checkpoint(tmp_path):
    base = {"out_dir": str(tmp_path), "device": "cpu", "games_per_iter": 32, "eval.every": 1, "eval.games": 16,
            "ppo.minibatch_size": 256, "checkpoint_every": 1}
    train(load_config(PILOT, {**base, "name": "silent", "total_games": 32}))
    comm_over = {**base, "name": "comm", "total_games": 64, "comm.enabled": True,
                 "init_from": str(tmp_path / "silent" / "seed{seed}" / "latest.pt")}
    run_dir = train(load_config(PILOT, comm_over))
    lines = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {"good/says_judgment", "good/says_claim", "good/says_intent", "evil/says_judgment",
            "good/msg_entropy"} <= set(lines[1])
    assert load_checkpoint(run_dir / "latest.pt")["config"]["comm"]["enabled"]
    assert load_policies(run_dir / "latest.pt")["good"].comm
    train(load_config(PILOT, {**comm_over, "total_games": 96, "resume": True}))
    lines = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[-1]["iteration"] == 3


# ---------------------------------------------------------------------- replies and promises
def team_value(speaker: int, seats) -> int:
    """Team-head value, in `speaker`'s frame, of the absolute seats."""
    return 1 + A.TEAM_INDEX[len(seats)][tuple(sorted((s - speaker) % 6 for s in seats))]


def test_replies_answer_earlier_speakers_and_bind_votes():
    game = make_game(leader=0, comm=SEVEN)
    play_quest(game, [0, 4], approvers=[0, 1, 4, 5], failers=[4])  # records against 0, 4 (quest) and 1, 5 (vote)
    # Round 2 (three seats), leader 1: seats 2, 3, 4, 5, 0 speak, then the leader.
    assert not mask_of(game, P.H_REPLY + 4).any()  # the first speaker has nobody to answer
    speak_as(game, 2, team=team_value(2, [1, 2, 5]))
    endorse_ok = mask_of(game, P.H_REPLY + 4, judge2=P.J_EVIL)  # seat 3 answers seat 2 after accusing seat 5
    assert endorse_ok.tolist() == [True, False, True]  # it cannot endorse a team with a player it accuses
    say(game, judge2=P.J_EVIL, reply5=P.Y_DISPUTE)
    assert game.msg_log[1, 3, P.H_RBASIS + 4] == P.B_TEAM  # derived: seat 2's team holds the accused seat 5
    assert mask_of(game, P.H_REPLY + 3).tolist() == [True, True, False]  # seat 4 has no basis to dispute seat 2
    say(game, reply4=P.Y_ENDORSE)
    say(game, team=team_value(5, [2, 3, 5]))  # seat 5 suggests another team
    say(game)  # seat 0
    say(game, reply1=P.Y_ENDORSE)  # the leader adopts seat 2's team ...
    teams = game.legal_mask(1)[:20]
    assert np.flatnonzero(teams).tolist() == [A.team_action(1, [1, 2, 5])]  # ... and proposes it
    propose(game, [1, 2, 5])
    assert [bound_vote(game, p) for p in range(6)] == ["free", "approve", "approve", "reject", "approve", "free"]
    record = game.to_record()["discussions"][1]["messages"]
    assert record[1]["replies"] == {2: {"reply": "dispute", "basis": "team"}} and record[2]["replies"] == {2: {"reply": "endorse"}}
    vote(game, [1, 2, 4, 0])
    assert not game.audit[:, [P.A_REPLY, P.A_PROMISE]].any()


def test_a_leader_adopts_only_a_clean_team_with_itself():
    game = make_game(leader=0, comm=SEVEN)
    speak_as(game, 1, team=team_value(1, [1, 0]))  # round 1: no record, every team is equally clean
    speak_as(game, 2, team=team_value(2, [2, 3]))
    assert mask_of(game, P.H_REPLY + 4).tolist() == [True, True, False]  # nobody has a basis to dispute yet
    while game.acting_players() != [0]:
        say(game)
    assert mask_of(game, P.H_REPLY + 1).tolist() == [True, False, False]  # seat 2's team is without the leader
    assert mask_of(game, P.H_REPLY + 0)[P.Y_ENDORSE]
    say(game, reply1=P.Y_ENDORSE)
    assert np.flatnonzero(game.legal_mask(0)[:15]).tolist() == [A.team_action(0, [0, 1])]


def test_a_dispute_cites_the_first_basis_that_holds():
    game = make_game(leader=0, comm=SEVEN)
    play_quest(game, [0, 4], approvers=[0, 2, 4, 5], failers=[4])  # against 0, 4 (quest), 2, 5 (vote)
    play_quest(game, [1, 3, 5], approvers=[1, 3, 5, 0])  # for 1, 3, 5 (quest)
    # Round 3, leader 2: seats 3, 4, 5, 0, 1, then 2.
    speak_as(game, 3, judge2=P.J_GOOD, judge1=P.J_EVIL)  # seat 3 vouches for seat 5 and accuses seat 4
    say(game, judge1=P.J_EVIL, reply5=P.Y_DISPUTE)  # seat 4 calls seat 5 evil and disputes seat 3 ...
    assert game.msg_log[2, 4, P.H_RBASIS + 4] == P.B_DEFENSE  # ... on the accusation before the judgment
    speak_as(game, 1, judge4=P.J_EVIL, reply2=P.Y_DISPUTE)  # seat 1 calls seat 5 evil and disputes seat 3
    assert game.msg_log[2, 1, P.H_RBASIS + 1] == P.B_JUDGMENT


def test_promises_force_the_next_rounds_judgment():
    game = make_game(leader=0, comm=SEVEN)
    game.free_seats = frozenset({3})
    # Round 1, leader 0: seats 1-5, then 0.
    speak_as(game, 2, promise2=P.K_FAIL_EVIL, promise5=P.K_SUCCESS_GOOD)  # about seats 4 and 1
    assert mask_of(game, P.H_PROMISE + 2, promise1=P.K_FAIL_EVIL, promise2=P.K_FAIL_EVIL).tolist() == [True, False, True]
    game.step({3: P.free_message(game, 3, promises=[P.K_FAIL_EVIL])})  # the free seat 3: "if 4 fails, 4 is evil"
    while game.acting_players() != [0]:
        say(game)
    merlin = mask_of(game, P.H_PROMISE + 0)  # Merlin cannot promise to accuse seat 1, whom he knows to be good
    assert not merlin[P.K_FAIL_EVIL] and mask_of(game, P.H_PROMISE + 3)[P.K_FAIL_EVIL]  # seat 4 he can
    say(game)
    propose(game, [0, 4])
    vote(game, range(6))
    quest(game, failers=[4])
    # Round 2, leader 1: seat 2 must now call seat 4 evil; seat 1 was not on the quest, so that promise lapses.
    assert mask_of(game, P.H_JUDGE + 1).tolist() == [False, False, True] + [False] * 5
    assert mask_of(game, P.H_JUDGE + 4)[[P.J_NONE, P.J_GOOD, P.J_EVIL]].tolist() == [True, False, True]
    say(game)
    assert game.msg_log[1, 2, P.H_JUDGE + 1] == P.J_EVIL
    game.step({3: P.free_message(game, 3)})  # the free seat breaks its promise: counted in public
    assert game.audit[3, P.A_PROMISE] == 1 and game.audit[:, P.A_PROMISE].sum() == 1


def test_promises_close_where_no_discussion_follows():
    game = make_game(leader=0, comm=SEVEN, binding=False)  # leaders propose freely
    play_quest(game, [0, 4], failers=[4])
    play_quest(game, [1, 2, 5], failers=[5])  # two fails: a third ends the game
    ok = P.message_context(game, game.acting_players()[0])["promise_ok"]
    assert ok[:, P.K_FAIL_EVIL].sum() == 0 and ok[:, P.K_SUCCESS_GOOD].all()


def test_communication_networks_before_replies_and_promises_load_unchanged():
    from src.agents.policy import is_legacy_state

    torch.manual_seed(0)
    net = ActorCritic(comm=True)
    obs, g_old, g_new = net.obs_dim, P.LEGACY["comm_global_dim"], P.COMM_GLOBAL_DIM
    n_old = A.N_ACTIONS + P.LEGACY["n_msg_logits"]
    with torch.no_grad():  # what an old network looks like in today's layout: nothing reads or writes the new parts
        for layer in (net.actor[0], net.critic[0]):
            layer.weight[:, obs + g_old:obs + g_new] = 0
        net.actor[-1].weight[n_old:] = 0
        net.actor[-1].bias[n_old:] = 0
        for enc in (net.actor_msg, net.critic_msg):
            enc.weight[:, P.LEGACY["msg_feat_dim"]:] = 0
    legacy = {}
    for key, value in net.state_dict().items():
        if key in ("actor.0.weight", "critic.0.weight"):
            value = torch.cat([value[:, :obs + g_old], value[:, obs + g_new:]], dim=1)
        elif key in ("actor.4.weight", "actor.4.bias"):
            value = value[:n_old]
        elif key in ("actor_msg.weight", "critic_msg.weight"):
            value = value[:, :P.LEGACY["msg_feat_dim"]]
        legacy[key] = value
    assert is_legacy_state(legacy) and not is_legacy_state(net.state_dict())
    upgraded = ActorCritic(comm=True)
    upgraded.load_state_dict(legacy)
    for key, value in net.state_dict().items():
        assert torch.equal(upgraded.state_dict()[key], value), key


def test_a_team_suggestion_is_a_choice_only_where_the_configuration_says_so():
    forced, free = make_game(leader=0, comm=FIVE), make_game(leader=0, comm=OPT)
    assert not mask_of(forced, P.H_TEAM)[0] and mask_of(forced, P.H_TEAM)[1:].any()
    assert mask_of(free, P.H_TEAM)[0] and (mask_of(free, P.H_TEAM)[1:] == mask_of(forced, P.H_TEAM)[1:]).all()
    say(free, team=0)  # seat 1 names no team: nothing for the others to answer
    assert not mask_of(free, P.H_REPLY + 4)[[P.Y_ENDORSE, P.Y_DISPUTE]].any()
    discuss(free)
    propose(free, [0, 1])
    assert bound_vote(free, 1) == "free"  # and nothing binds its vote
