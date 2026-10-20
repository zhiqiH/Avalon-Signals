import numpy as np

from avalon_rl.env import LAYOUT, OBS_DIM, Role, encode
from helpers import ROLES, make_game, play_quest, propose, vote


def test_layout_tiles_the_vector():
    spans = sorted((s.start, s.stop) for s in LAYOUT.values())
    assert spans[0][0] == 0 and spans[-1][1] == OBS_DIM
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))


def test_servant_has_no_private_knowledge():
    obs = encode(make_game(), 2)
    assert not obs[LAYOUT["known_evil"]].any() and not obs[LAYOUT["merlin_candidate"]].any()


def test_merlin_and_evil_see_exactly_the_evil_seats():
    game = make_game()
    # Merlin at seat 0: evil at relative seats 4, 5.
    assert np.flatnonzero(encode(game, 0)[LAYOUT["known_evil"]]).tolist() == [4, 5]
    # Morgana at seat 4: herself (0) and the Assassin (1).
    assert np.flatnonzero(encode(game, 4)[LAYOUT["known_evil"]]).tolist() == [0, 1]


def test_percival_cannot_tell_merlin_from_morgana():
    swapped = list(ROLES)
    swapped[0], swapped[4] = Role.MORGANA, Role.MERLIN
    a, b = encode(make_game(ROLES), 1), encode(make_game(swapped), 1)
    assert np.array_equal(a, b)
    # Relative seats of seats 0 and 4 from seat 1 are 5 and 3.
    assert np.flatnonzero(a[LAYOUT["merlin_candidate"]]).tolist() == [3, 5]


def test_hidden_roles_never_reach_a_servant():
    other = [Role.SERVANT, Role.ASSASSIN, Role.MERLIN, Role.MORGANA, Role.PERCIVAL]
    roles_a = [Role.SERVANT] + [ROLES[i] for i in (0, 1, 3, 4, 5)]
    roles_b = [Role.SERVANT] + other
    games = [make_game(roles_a), make_game(roles_b)]
    script = [lambda g: propose(g, [0, 1]), lambda g: vote(g, [0, 2]), lambda g: play_quest(g, [0, 3])]
    for step in script:
        assert np.array_equal(encode(games[0], 0), encode(games[1], 0))
        for g in games:
            step(g)
    assert np.array_equal(encode(games[0], 0), encode(games[1], 0))


def test_seat_fields_are_relative_to_the_observer():
    game = make_game(leader=2)
    propose(game, [2, 3])
    obs = encode(game, 3)
    assert np.flatnonzero(obs[LAYOUT["leader"]]).tolist() == [5]  # seat 2 is 5 places left of seat 3
    assert np.flatnonzero(obs[LAYOUT["current_team"]]).tolist() == [0, 5]


def test_votes_are_plus_minus_one_once_cast():
    game = make_game()
    propose(game, [0, 1])
    assert not encode(game, 0)[LAYOUT["proposal_votes"]].any()  # not voted yet
    vote(game, [0, 1, 2])
    votes = encode(game, 0)[LAYOUT["proposal_votes"]].reshape(25, 6)[0]
    assert votes.tolist() == [1, 1, 1, -1, -1, -1]
