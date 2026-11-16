import numpy as np

from src.env import LAYOUT, OBS_DIM, Role, encode
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


# ---------------------------------------------------------------------- deduction fields
from itertools import combinations  # noqa: E402

from src.env import AvalonGame  # noqa: E402
from src.env.observation import EVIL_PAIRS, encode_batch  # noqa: E402

EVIL = [4, 5]  # seats of Morgana and the Assassin in ROLES


def deduced(obs, view):
    pairs = [EVIL_PAIRS[i] for i in np.flatnonzero(obs[LAYOUT[f"pair_possible_{view}"]])]
    return (pairs, np.flatnonzero(obs[LAYOUT[f"certain_evil_{view}"]]).tolist(),
            np.flatnonzero(obs[LAYOUT[f"certain_good_{view}"]]).tolist())


def test_double_fail_on_a_two_player_team_exposes_both_evil_publicly():
    game = make_game()
    play_quest(game, EVIL, failers=EVIL)
    pairs, evil, good = deduced(encode(game, 2), "public")  # a servant; seats 4, 5 are relative 2, 3
    assert pairs == [(2, 3)] and evil == [2, 3] and good == [0, 1, 4, 5]


def test_double_fail_on_a_three_player_team_clears_everyone_off_it():
    game = make_game()
    play_quest(game, [0, 1], failers=())
    play_quest(game, [2, 4, 5], failers=EVIL)
    pairs, evil, good = deduced(encode(game, 0), "public")
    assert sorted(pairs) == [(2, 4), (2, 5), (4, 5)] and evil == [] and good == [0, 1, 3]
    # Seat 2 knows it is good, so for it the pair is certain.
    _, evil, _ = deduced(encode(game, 2), "private")
    assert evil == [2, 3]


def test_a_good_player_on_a_failed_pair_knows_its_partner():
    game = make_game()
    play_quest(game, [2, 4], failers=[4])
    _, evil, good = deduced(encode(game, 2), "public")
    assert evil == [] and good == []
    _, evil, good = deduced(encode(game, 2), "private")
    assert evil == [2] and 0 in good  # seat 4 is relative seat 2 of seat 2


def test_percival_keeps_only_pairs_with_one_candidate():
    pairs, _, good = deduced(encode(make_game(), 1), "private")  # candidates: seats 0 and 4 = relative 5 and 3
    assert pairs and all(len({3, 5} & set(p)) == 1 and 0 not in p for p in pairs)
    assert 0 in good


def test_merlin_and_evil_deduce_the_exact_pair():
    for player in (0, 4, 5):
        pairs, evil, _ = deduced(encode(make_game(), player), "private")
        rel = sorted((s - player) % 6 for s in EVIL)
        assert pairs == [tuple(rel)] and evil == rel


def test_a_servant_deduces_nothing_from_hidden_roles():
    roles_a = [Role.SERVANT, Role.MERLIN, Role.PERCIVAL, Role.MORGANA, Role.SERVANT, Role.ASSASSIN]
    roles_b = [Role.SERVANT, Role.PERCIVAL, Role.MORGANA, Role.ASSASSIN, Role.SERVANT, Role.MERLIN]
    games = [make_game(roles_a), make_game(roles_b)]
    for g in games:
        play_quest(g, [0, 3], failers=[3])  # seat 3 is evil in both assignments
    assert np.array_equal(encode(games[0], 0), encode(games[1], 0))


def test_deductions_are_sound_and_exact_in_random_games():
    rng = np.random.default_rng(1)
    for _ in range(150):
        game = AvalonGame(rng)
        evil_abs = set(np.flatnonzero([game.is_evil_seat(s) for s in range(6)]).tolist())
        while not game.done:
            played = np.flatnonzero(game.quest_result >= 0)
            public = {p for p in combinations(range(6), 2)
                      if all(game.quest_fails[q] <= len(set(p) & set(np.flatnonzero(game.quest_team[q]))) for q in played)}
            for player in range(6):
                obs = encode(game, player)
                rel = lambda pair: tuple(sorted((s - player) % 6 for s in pair))  # noqa: E731
                pub, ev_pub, good_pub = deduced(obs, "public")
                priv, ev_priv, good_priv = deduced(obs, "private")
                assert set(pub) == {rel(p) for p in public}
                true_pair = rel(evil_abs)
                assert true_pair in priv and set(priv) <= set(pub)
                true_evil = set(true_pair)
                assert set(ev_priv) <= true_evil and not true_evil & set(good_priv)
                assert set(ev_pub) <= set(ev_priv) and set(good_pub) <= set(good_priv)
            game.step({p: int(rng.choice(np.flatnonzero(game.legal_mask(p)))) for p in game.acting_players()})


def test_batch_encoding_equals_one_by_one_encoding():
    rng = np.random.default_rng(2)
    games = [AvalonGame(rng) for _ in range(40)]
    while not all(g.done for g in games):
        pairs = [(g, p) for g in games if not g.done for p in g.acting_players()]
        pairs += [(g, p) for g in games[:5] if not g.done for p in range(6)]  # every seat, including repeats
        order = rng.permutation(len(pairs))
        pairs = [pairs[i] for i in order]
        assert np.array_equal(encode_batch(pairs), np.stack([encode(g, p) for g, p in pairs]))
        for g in games:
            if not g.done:
                g.step({p: int(rng.choice(np.flatnonzero(g.legal_mask(p)))) for p in g.acting_players()})
