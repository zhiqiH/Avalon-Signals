import json
from pathlib import Path

import numpy as np
import pytest
import torch

from avalon_rl.agents import ActorCritic, PolicyController, RandomController, Request, RuleController
from avalon_rl.agents.policy import load_silent_weights
from avalon_rl.comm import protocol as P
from avalon_rl.config import load_config
from avalon_rl.env import AvalonGame, Phase, Role, encode
from avalon_rl.env import actions as A
from avalon_rl.runner import play
from avalon_rl.training import TrajectoryRecorder
from avalon_rl.training.checkpoint import load_checkpoint, load_policies
from avalon_rl.training.trainer import train
from helpers import ROLES, make_game, quest, vote

PILOT = Path(__file__).resolve().parents[1] / "configs" / "pilot_silent.yaml"
F = P.CommConfig(enabled=True)
M = P.CommConfig(enabled=True, claim=False, team=False, intent=False)


def say(game, **heads) -> None:
    """The current speaker says nothing except the given heads (the leader's team suggestion stays its proposal)."""
    speaker = game.acting_players()[0]
    mask, rtype_valid, rround_valid = P.message_mask(game, speaker)
    msg = P.sample_message(mask, rtype_valid, rround_valid, np.random.default_rng(0), first=True)
    for name, value in heads.items():
        msg[getattr(P, f"H_{name.upper()}") if not name[-1].isdigit() else
            getattr(P, f"H_{name[:-1].upper()}") + int(name[-1]) - 1] = value
    game.step({speaker: msg})


def discuss(game) -> None:
    while game.in_discussion:
        say(game)


def propose(game, seats) -> None:
    game.step({game.leader: A.team_action(game.leader, seats)})


def play_quest(game, seats, failers=()) -> None:
    propose(game, seats)
    discuss(game)
    vote(game, range(6))
    quest(game, failers)


def test_discussion_follows_each_proposal_leader_first():
    game = make_game(leader=2, comm=F)
    propose(game, [2, 3])
    assert game.phase == Phase.DISCUSS
    order = []
    while game.in_discussion:
        order.append(game.acting_players()[0])
        say(game)
    assert order == [2, 3, 4, 5, 0, 1] and game.phase == Phase.VOTE


def test_closed_slots_can_only_say_nothing():
    game = make_game(comm=M)
    propose(game, [0, 1])
    mask, rtype_valid, _ = P.message_mask(game, 0)
    for h in (P.H_CLAIM, P.H_TEAM, P.H_INTENT, P.H_NAME):
        assert np.flatnonzero(P.head_mask(mask, h)).tolist() == [0]
    assert P.head_mask(mask, P.H_ALIGN).all()
    assert rtype_valid[:, 0].all()  # rationale is open in M


def test_leader_suggestion_is_fixed_to_its_proposal():
    game = make_game(leader=1, comm=F)
    propose(game, [1, 4])
    leader_team = np.flatnonzero(P.head_mask(P.message_mask(game, 1)[0], P.H_TEAM))
    assert leader_team.tolist() == [1 + A.team_action(1, [1, 4])]
    other_team = P.head_mask(P.message_mask(game, 2)[0], P.H_TEAM)
    assert other_team[0] and other_team.sum() == 1 + len(A.TEAMS_BY_SIZE[2])


def test_naming_is_limited_to_players_who_have_not_spoken():
    game = make_game(leader=0, comm=F)
    propose(game, [0, 1])
    assert P.head_mask(P.message_mask(game, 0)[0], P.H_NAME).all()
    for _ in range(3):
        say(game)  # seats 0, 1, 2 speak
    name = P.head_mask(P.message_mask(game, 3)[0], P.H_NAME)
    assert np.flatnonzero(name).tolist() == [0, 1, 2]  # nobody, seat 4 (r=1), seat 5 (r=2)


def test_rationale_must_cite_real_events():
    game = make_game(leader=0, comm=F)
    play_quest(game, [0, 1])  # quest 1: seats 0, 1, success; leadership passes to seat 1
    propose(game, [1, 2, 3])  # quest 2, proposed by seat 1
    speaker = 1
    rtype_valid, rround_valid = P.rationale_validity(game, speaker)
    r = {seat: (seat - speaker) % 6 - 1 for seat in range(6) if seat != speaker}  # row of each target
    quest_t, vote_t, prop_t = 1, 2, 3
    assert rround_valid[r[0], quest_t].tolist() == [True, False, False, False, False]
    assert not rtype_valid[r[2], quest_t] and rtype_valid[r[2], prop_t] and rtype_valid[r[2], 4]
    assert rround_valid[r[4], vote_t].tolist() == [True, False, False, False, False]
    assert rround_valid[r[3], prop_t].tolist() == [False, True, False, False, False]  # on the current proposal
    assert rround_valid[r[0], prop_t].tolist() == [True, False, False, False, False]  # led and was on quest 1's team

    mask, _, _ = P.message_mask(game, speaker)
    msg = P.sample_message(mask, rtype_valid, rround_valid, np.random.default_rng(0), first=True)
    bad = msg.copy()
    bad[P.H_ALIGN + r[2]] = 2
    bad[P.H_RTYPE + r[2]] = quest_t  # seat 2 has never been on a quest
    bad[P.H_RROUND + r[2]] = 0
    with pytest.raises(ValueError):
        P.check_message(game, speaker, bad)
    unjudged = msg.copy()
    unjudged[P.H_RTYPE + r[3]] = 4  # a rationale without a judgment
    with pytest.raises(ValueError):
        P.check_message(game, speaker, unjudged)
    good = msg.copy()
    good[P.H_ALIGN + r[0]], good[P.H_RTYPE + r[0]], good[P.H_RROUND + r[0]] = 1, quest_t, 0
    P.check_message(game, speaker, good)


def test_messages_are_encoded_in_the_observers_frame():
    game = make_game(leader=0, comm=F)
    propose(game, [0, 1])
    # Seat 0 claims Merlin, calls seat 4 (r=4) evil citing role knowledge, rejects and names seat 2 (r=2).
    say(game, claim=3, align4=2, rtype4=4, intent=2, name=2)
    feats = P.encode_comm(game, 1)[P.COMM_LAYOUT["messages"]].reshape(5, 5, 6, P.MSG_FEAT_DIM)
    f = feats[0, 0, 5]  # seat 0 sits 5 places left of seat 1
    L = P.MSG_LAYOUT
    assert f[L["spoke"]][0] == 1 and np.flatnonzero(f[L["claim"]]).tolist() == [2]  # Merlin
    assert np.flatnonzero(f[L["align"]]).tolist() == [2 * 3 + 1]  # seat 4 is relative seat 3 of seat 1: evil
    assert np.flatnonzero(f[L["rtype"]]).tolist() == [4 * 3 + 3]  # role knowledge
    assert np.flatnonzero(f[L["name"]]).tolist() == [1] and np.flatnonzero(f[L["intent"]]).tolist() == [1]
    assert np.flatnonzero(f[L["team"]]).tolist() == [0, 5]  # the leader restates its proposal of seats 0, 1
    spoken = P.encode_comm(game, 1)[P.COMM_LAYOUT["spoken"]]
    assert np.flatnonzero(spoken).tolist() == [5]


def test_messages_do_not_reveal_hidden_roles():
    roles_b = [Role.SERVANT, Role.ASSASSIN, Role.MERLIN, Role.MORGANA, Role.PERCIVAL, Role.SERVANT]
    roles_a = [Role.SERVANT, Role.MERLIN, Role.PERCIVAL, Role.SERVANT, Role.MORGANA, Role.ASSASSIN]
    games = [make_game(roles_a, comm=F), make_game(roles_b, comm=F)]
    for g in games:
        propose(g, [0, 1])
        say(g, claim=1, align3=2, rtype3=4)  # seat 0 claims Servant and calls seat 3 evil (role knowledge)
        say(g, align5=1, rtype5=3, rround5=0)  # seat 1 calls seat 0 good, citing quest 1's proposal
    assert np.array_equal(P.encode_comm(games[0], 0), P.encode_comm(games[1], 0))
    assert np.array_equal(encode(games[0], 0), encode(games[1], 0))


def test_evil_players_talk_before_the_assassination():
    game = make_game(comm=F)
    for _ in range(3):
        play_quest(game, list(range(game.team_size)))
    assert game.phase == Phase.EVIL_DISCUSS
    assert game.acting_players() == [4]  # Morgana, then the Assassin
    mask, rtype_valid, _ = P.message_mask(game, 4)
    assert np.flatnonzero(P.head_mask(mask, P.H_ALIGN)).tolist() == [0]  # r=1 is the Assassin: no judgment
    assert P.head_mask(mask, P.H_ALIGN + 1).all()  # r=2 is seat 0
    assert np.flatnonzero(P.head_mask(mask, P.H_CLAIM)).tolist() == [0] and not rtype_valid.any()
    say(game, align2=1)  # Morgana: seat 0 is Merlin
    assert game.acting_players() == [5]
    say(game)
    assert game.phase == Phase.ASSASSINATE
    talk = P.encode_comm(game, 5)[P.COMM_LAYOUT["evil_talk"]].reshape(6, 6, 2)
    assert talk[5, 1, 0] == 1 and talk.sum() == 1  # Morgana (relative 5 of seat 5) named seat 0 (relative 1)
    game.step({5: A.assassin_action(5, 0)})
    assert game.done and not game.good_won
    assert game.to_record()["evil_talk"] == [{"speaker": 4, "alignment": {0: {"judgment": "merlin"}}}, {"speaker": 5}]


def test_silenced_slots_look_like_saying_nothing():
    silenced = P.CommConfig(enabled=True, silence=list(P.SLOTS))
    game = make_game(comm=silenced)
    propose(game, [0, 1])
    for speaker in (0, 1):
        assert P.forced_message(P.message_mask(game, speaker)[0]) is not None
        say(game)
    comm = P.encode_comm(game, 3)
    assert comm[P.COMM_LAYOUT["slot_open"]].all()  # the slots still exist for the policy
    reference = make_game(comm=F)
    propose(reference, [0, 1])
    say(reference)
    say(reference)
    assert np.array_equal(comm, P.encode_comm(reference, 3))


@pytest.mark.parametrize("cfg", [F, M])
def test_scripted_players_finish_communication_games(cfg):
    rng = np.random.default_rng(0)
    rule, rand = RuleController(), RandomController()
    for good, evil in [(rule, rand), (rand, rand)]:
        games = [AvalonGame(rng, comm=cfg) for _ in range(50)]
        play(games, [good] * 50, [evil] * 50, rng)  # step() rejects illegal messages
        assert all(g.done for g in games)
    record = games[0].to_record()
    assert record["comm"]["alignment"] and all("messages" in p for p in record["proposals"])


def test_policy_messages_are_legal_and_logps_are_reproducible():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = ActorCritic(comm=True)
    with torch.no_grad():
        net.actor[-1].weight.mul_(100)  # make message heads far from uniform
    learner = PolicyController(net, learn=True)
    games = [AvalonGame(rng, comm=F) for _ in range(16)]
    recorder = TrajectoryRecorder()
    play(games, [learner] * 16, [RandomController()] * 16, rng, on_decision=recorder)
    batch = recorder.build(games, gamma=1.0, lam=0.95)["good"]
    assert (batch.actions[:, 0] < 0).any() and (batch.actions[:, 0] >= 0).any()
    t = torch.as_tensor
    logp, _, value, _ = net.evaluate(t(batch.obs), t(batch.comm_obs), t(batch.masks), t(batch.actions))
    assert torch.allclose(logp, t(batch.logps), atol=1e-4)
    assert torch.allclose(value, t(batch.values), atol=1e-5)


def test_branch_from_silent_weights_plays_like_the_silent_network():
    torch.manual_seed(0)
    silent, comm = ActorCritic(), ActorCritic(comm=True)
    load_silent_weights(comm, silent.state_dict())
    rng = np.random.default_rng(1)
    games = [AvalonGame(rng, comm=F) for _ in range(8)]
    rand = RandomController()
    for _ in range(40):  # advance through several discussions with random messages
        play_one_tick(games, rand, rng)
    reqs = [Request(i, g, p, g.legal_mask(p)) for i, g in enumerate(games) if not g.done for p in g.acting_players()]
    obs = torch.as_tensor(np.stack([encode(r.game, r.player) for r in reqs]))
    comm_obs = torch.as_tensor(np.stack([P.encode_comm(r.game, r.player) for r in reqs]))
    assert comm_obs[:, P.COMM_LAYOUT["messages"]].any()
    with torch.no_grad():
        logits = comm.logits(obs, comm_obs)
        assert torch.allclose(logits[:, :A.N_ACTIONS], silent.logits(obs), atol=1e-6)
        assert torch.all(logits[:, A.N_ACTIONS:] == 0)  # every message head starts uniform
        assert torch.allclose(comm.value(obs, comm_obs), silent.value(obs), atol=1e-6)


def play_one_tick(games, ctrl, rng) -> None:
    for g in games:
        if g.done:
            continue
        chosen = {}
        for p in g.acting_players():
            if g.in_discussion:
                mask, tv, rv = P.message_mask(g, p)
                chosen[p] = P.sample_message(mask, tv, rv, rng)
            else:
                chosen[p] = int(rng.choice(np.flatnonzero(g.legal_mask(p))))
        g.step(chosen)


def test_short_communication_run_branches_from_a_silent_checkpoint(tmp_path):
    base = {"out_dir": str(tmp_path), "device": "cpu", "games_per_iter": 32, "eval.every": 1, "eval.games": 16,
            "ppo.minibatch_size": 256, "checkpoint_every": 1}
    silent_dir = train(load_config(PILOT, {**base, "name": "silent", "total_games": 32}))
    comm_over = {**base, "name": "comm", "total_games": 64, "comm.enabled": True,
                 "init_from": str(tmp_path / "silent" / "seed{seed}" / "latest.pt")}
    run_dir = train(load_config(PILOT, comm_over))
    lines = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {"good/says_alignment", "good/says_claim", "evil/says_alignment", "good/msg_entropy"} <= set(lines[1])
    assert load_checkpoint(run_dir / "latest.pt")["config"]["comm"]["enabled"]
    assert load_policies(run_dir / "latest.pt")["good"].comm
    train(load_config(PILOT, {**comm_over, "total_games": 96, "resume": True}))
    lines = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[-1]["iteration"] == 3
    assert silent_dir.exists()
