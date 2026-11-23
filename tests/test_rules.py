from collections import Counter

import numpy as np
import pytest

from src.env import AvalonGame, Phase, Role, WinReason
from src.env import actions as A
from src.env.rules import QUEST_SIZES
from helpers import good_team, make_game, play_quest, propose, quest, vote


def test_role_deck_is_dealt_each_game():
    rng = np.random.default_rng(1)
    for _ in range(50):
        counts = Counter(Role(int(r)) for r in AvalonGame(rng).roles)
        assert counts == {Role.SERVANT: 2, Role.MERLIN: 1, Role.PERCIVAL: 1, Role.MORGANA: 1, Role.ASSASSIN: 1}


def test_team_actions_round_trip():
    for player in range(6):
        for size, teams in A.TEAMS_BY_SIZE.items():
            for rel in teams:
                abs_team = tuple(sorted((player + r) % 6 for r in rel))
                assert A.decode_team(player, A.team_action(player, abs_team), size) == abs_team


def test_team_size_and_legal_proposals_follow_quest_index():
    game = make_game()
    for q in range(3):
        assert game.team_size == QUEST_SIZES[q]
        assert game.legal_mask(game.leader).sum() == len(A.TEAMS_BY_SIZE[QUEST_SIZES[q]])
        play_quest(game, good_team(game))
    assert game.phase == Phase.ASSASSINATE


def test_four_approvals_pass_and_three_reject():
    game = make_game()
    propose(game, [0, 1])
    vote(game, [0, 1, 2])
    assert game.phase == Phase.PROPOSE and game.attempt == 1
    propose(game, [0, 1])
    vote(game, [0, 1, 2, 3])
    assert game.phase == Phase.QUEST and game.prop_approved[0, 1]


def test_leader_passes_after_every_vote():
    game = make_game(leader=4)
    propose(game, [0, 1])
    vote(game, [])
    assert game.leader == 5
    propose(game, [0, 1])
    vote(game, range(6))
    assert game.leader == 0


def test_five_rejections_give_evil_the_win():
    game = make_game()
    game.good_guard = False  # with the guard the four good players approve the fifth proposal
    for _ in range(5):
        propose(game, [0, 1])
        vote(game, [])
    assert game.done and not game.good_won and game.win_reason == WinReason.FIVE_REJECTIONS


def test_single_fail_fails_a_quest_and_only_count_is_public():
    game = make_game()
    play_quest(game, [0, 4], failers=[4])
    assert game.quest_result[0] == 1 and game.quest_fails[0] == 1 and game.fails == 1
    assert game.attempt == 0 and game.quest == 1


def test_three_fails_give_evil_the_win():
    game = make_game()
    game.good_guard = False  # the scripted good players approve a team they know is dirty at two fails
    for _ in range(3):
        team = [4] + list(range(game.team_size - 1))
        play_quest(game, team, failers=[4])
    assert game.done and not game.good_won and game.win_reason == WinReason.THREE_FAILS


def test_good_players_cannot_fail():
    game = make_game()
    propose(game, [0, 4])
    vote(game, range(6))
    assert not game.legal_mask(0)[A.FAIL]
    assert game.legal_mask(4)[A.FAIL]


@pytest.mark.parametrize("target, good_won", [(0, False), (2, True)])
def test_assassination(target, good_won):
    game = make_game()
    for _ in range(3):
        play_quest(game, good_team(game))
    assassin = 5
    assert game.acting_players() == [assassin]
    mask = game.legal_mask(assassin)
    assert not mask[A.assassin_action(assassin, 4)] and not mask[A.assassin_action(assassin, 5)]
    game.step({assassin: A.assassin_action(assassin, target)})
    assert game.done and game.good_won == good_won


def test_illegal_or_missing_actions_are_rejected():
    game = make_game()
    with pytest.raises(ValueError):
        game.step({1: A.team_action(1, [0, 1])})  # not the leader
    propose(game, [0, 1])
    with pytest.raises(ValueError):
        game.step({0: A.APPROVE})  # every player must vote
    vote(game, range(6))
    with pytest.raises(ValueError):
        quest(game, failers=[0])  # a good player cannot fail


def test_good_players_must_approve_the_fifth_proposal():
    game = make_game()
    for _ in range(4):
        propose(game, [0, 1])
        vote(game, [])
    propose(game, [4, 5])
    for seat in range(4):
        mask = game.legal_mask(seat)
        assert mask[A.APPROVE] and not mask[A.REJECT]
    for seat in (4, 5):  # evil players stay free to reject
        assert game.legal_mask(seat)[[A.APPROVE, A.REJECT]].all()
    with pytest.raises(ValueError):
        vote(game, [])


def test_good_players_reject_a_certainly_dirty_team_at_two_fails():
    game = make_game()
    play_quest(game, [0, 4], failers=[4])  # evil pair holds 0 or 4
    play_quest(game, [1, 2, 5], failers=[5])  # ... and 1, 2 or 5
    assert game.fails == 2
    propose(game, [0, 1, 2, 4])
    # Merlin (0) knows 4 is evil; the Servant (2) knows itself good, so the pair holds 4 or 0: both see a dirty team.
    for seat in (0, 2):
        assert not game.legal_mask(seat)[A.APPROVE] and game.legal_mask(seat)[A.REJECT]
    assert game.legal_mask(4)[A.APPROVE]
    vote(game, [4])
    propose(game, [0, 1, 2, 3])  # clean in fact; no good player can rule that out
    assert all(game.legal_mask(seat)[A.APPROVE] for seat in range(4))
    vote(game, [])
    propose(game, [1, 2, 3, 5])
    assert not game.legal_mask(0)[A.APPROVE]  # Merlin knows 5 is evil
    assert not game.legal_mask(3)[A.APPROVE]  # every pair the Servant allows holds 1, 2 or 5


def test_guard_applies_only_at_two_fails_and_can_be_lifted():
    game = make_game()
    play_quest(game, [0, 4], failers=[4])
    propose(game, [0, 4, 5])
    assert game.legal_mask(0)[A.APPROVE]  # one fail: Merlin may still approve a dirty team
    game = make_game()
    game.good_guard = False
    for _ in range(4):
        propose(game, [0, 1])
        vote(game, [])
    propose(game, [0, 1])
    assert game.legal_mask(0)[A.REJECT]
    game = make_game()
    game.unguarded_seats = frozenset({1})  # a human in seat 1 may reject the fifth proposal; the others may not
    for _ in range(4):
        propose(game, [0, 1])
        vote(game, [])
    propose(game, [0, 1])
    assert game.legal_mask(1)[A.REJECT] and not game.legal_mask(0)[A.REJECT]
