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
    game = make_game(leader=2, comm=F)
    assert game.phase == Phase.DISCUSS
    order = []
    while game.in_discussion:
        order.append(game.acting_players()[0])
        say(game)
    assert order == [3, 4, 5, 0, 1, 2] and game.phase == Phase.PROPOSE and game.acting_players() == [2]
    propose(game, [2, 3])
    vote(game, [])  # rejected: the next leader proposes at once, without discussion
    assert game.phase == Phase.PROPOSE and game.acting_players() == [3]
    propose(game, [3, 4])
    vote(game, range(6))
    quest(game)
    assert game.phase == Phase.DISCUSS and game.speakers == [5, 0, 1, 2, 3, 4]


def test_closed_slots_can_only_say_nothing():
    ctx = P.message_context(make_game(comm=M), 1)
    assert ctx["claim_ok"].tolist() == [True, False, False, False]
    assert ctx["team_ok"].sum() == 1 and not ctx["veto"] and ctx["judge_ok"].all()


def test_judgments_respect_the_role_deck():
    game = make_game(leader=0, comm=F)  # seat 1 speaks first
    assert not mask_of(game, P.H_JUDGE, claim=P.C_MERLIN)[P.J_MERLIN]  # claimed Merlin: nobody else is Merlin
    two_evil = dict(judge1=P.J_EVIL, judge2=P.J_MORGANA)
    ok = mask_of(game, P.H_JUDGE + 2, **two_evil)
    assert not ok[[P.J_EVIL, P.J_MORGANA, P.J_ASSASSIN]].any() and ok[P.J_GOOD]
    ok = mask_of(game, P.H_JUDGE + 1, claim=P.C_SERVANT, judge1=P.J_SERVANT)
    assert not ok[P.J_SERVANT] and ok[P.J_MERLIN]  # the claim and one judgment use both Servant cards
    four_good = dict(claim=P.C_PERCIVAL, judge1=P.J_GOOD, judge2=P.J_MERLIN, judge3=P.J_SERVANT)
    ok = mask_of(game, P.H_JUDGE + 3, **four_good)
    assert not ok[list(P.GOOD_JUDGMENTS)].any() and ok[P.J_ASSASSIN]


def scripted_history(game) -> None:
    """Quest 1 (leader 0): seats 0, 1 succeed, all approve. Quest 2 (leader 1): seats 2, 3, 4; seat 4 fails;
    approvers 0, 2, 4, 5. Quest 3 opens with leader 2, so seat 3 speaks first."""
    play_quest(game, [0, 1])
    play_quest(game, [2, 3, 4], approvers=[0, 2, 4, 5], failers=[4])


def test_rationale_points_the_same_way_as_the_judgment():
    game = make_game(leader=0, comm=F)
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
    assert evil[[P.R_NONE, P.R_QUEST, P.R_VOTE, P.R_ROLE]].all() and not evil[P.R_PROPOSAL]
    good = mask_of(game, P.H_RTYPE + seat4 - 1, **{f"judge{seat4}": P.J_GOOD})
    assert not good[P.R_QUEST] and good[P.R_VOTE]  # only its approval of the successful quest 1 speaks for it
    rounds = mask_of(game, P.H_RROUND + seat4 - 1, **{f"judge{seat4}": P.J_GOOD, f"rtype{seat4}": P.R_VOTE})
    assert rounds.tolist() == [True, False, False, False, False]
    assert not mask_of(game, P.H_RTYPE + r[1] - 1).any()  # no judgment, no rationale

    bad = build(game, **{f"judge{seat4}": P.J_GOOD, f"rtype{seat4}": P.R_QUEST, f"rround{seat4}": 1})
    with pytest.raises(ValueError):
        P.check_message(game, 3, bad)


@pytest.mark.parametrize("claim, judge, allowed", [
    (P.C_NONE, P.J_EVIL, True), (P.C_MERLIN, P.J_GOOD, True), (P.C_MERLIN, P.J_MORGANA, False),
    (P.C_PERCIVAL, P.J_MERLIN, True), (P.C_PERCIVAL, P.J_EVIL, False), (P.C_SERVANT, P.J_EVIL, False),
    (P.C_NONE, P.J_ASSASSIN, False),
])
def test_role_knowledge_matches_the_claimed_sight(claim, judge, allowed):
    game = make_game(comm=F)
    assert mask_of(game, P.H_RTYPE, claim=claim, judge1=judge)[P.R_ROLE] == allowed


def test_team_suggestions_exclude_the_accused_and_the_leader_suggests_nothing():
    game = make_game(leader=0, comm=F)
    teams = mask_of(game, P.H_TEAM, judge2=P.J_EVIL)  # seat 1 calls seat 3 evil
    members = P.TEAM_MEMBERS[2]
    assert teams[0] and teams[1:].any()
    assert not (teams & members[:, 2]).any()  # no legal team holds relative seat 2 (seat 3)
    for _ in range(5):
        say(game)
    assert game.acting_players() == [0]
    assert np.flatnonzero(mask_of(game, P.H_TEAM)).tolist() == [0]


def test_vetoes_cannot_target_trusted_players_or_ones_own_team():
    game = make_game(leader=0, comm=F)
    team = 1 + A.TEAM_INDEX[2][(0, 1)]  # seat 1 suggests itself and seat 2
    assert not mask_of(game, P.H_VETO, judge1=P.J_GOOD)[1]
    assert not mask_of(game, P.H_VETO, team=team)[1]
    assert mask_of(game, P.H_VETO + 2, team=team, judge3=P.J_EVIL)[1]


def test_messages_are_encoded_in_the_observers_frame():
    game = make_game(leader=0, comm=F)  # seat 1 speaks first
    team = 1 + A.TEAM_INDEX[2][(0, 1)]  # seats 1, 2
    say(game, claim=P.C_MERLIN, judge3=P.J_EVIL, rtype3=P.R_ROLE, team=team, veto3=1)  # about seat 4
    comm = P.encode_comm(game, 2)
    f = comm[P.COMM_LAYOUT["messages"]].reshape(P.N_MSG_SLOTS, P.MSG_FEAT_DIM)[5]  # quest 1, seat 1 = relative 5
    L, per = P.MSG_LAYOUT, P._SEAT_FIELDS
    assert f[L["spoke"]][0] == 1 and np.flatnonzero(f[L["claim"]]).tolist() == [P.C_MERLIN - 1]
    assert np.flatnonzero(f[L["judge"]]).tolist() == [per["judge"] * 2 + P.J_EVIL - 1]  # seat 4 = relative 2
    assert np.flatnonzero(f[L["rtype"]]).tolist() == [per["rtype"] * 2 + P.R_ROLE - 1]
    assert np.flatnonzero(f[L["team"]]).tolist() == [0, 5] and np.flatnonzero(f[L["veto"]]).tolist() == [2]
    assert np.flatnonzero(comm[P.COMM_LAYOUT["spoken"]]).tolist() == [5]
    assert game.to_record()["discussions"] == [{"quest": 0, "leader": 0, "messages": [{
        "speaker": 1, "claim": "merlin", "judgments": {4: {"judgment": "evil", "basis": "role_knowledge"}},
        "team": [1, 2], "veto": [4]}]}]


def test_messages_do_not_reveal_hidden_roles():
    roles_a = [Role.SERVANT, Role.MERLIN, Role.PERCIVAL, Role.SERVANT, Role.MORGANA, Role.ASSASSIN]
    roles_b = [Role.SERVANT, Role.ASSASSIN, Role.MERLIN, Role.MORGANA, Role.PERCIVAL, Role.SERVANT]
    games = [make_game(roles_a, comm=F), make_game(roles_b, comm=F)]
    for g in games:
        say(g, claim=P.C_SERVANT, judge2=P.J_EVIL, rtype2=P.R_NONE)
        say(g, judge4=P.J_PERCIVAL, veto1=1)
    assert np.array_equal(P.encode_comm(games[0], 0), P.encode_comm(games[1], 0))
    assert np.array_equal(encode(games[0], 0), encode(games[1], 0))


def test_evil_players_judge_merlin_together_before_the_assassination():
    game = make_game(comm=F)
    for _ in range(3):
        play_quest(game, list(range(game.team_size)))
    assert game.phase == Phase.EVIL_DISCUSS and game.acting_players() == [4]  # Morgana, then the Assassin
    ctx = P.message_context(game, 4)
    assert ctx["judge_ok"][0].tolist() == [True] + [False] * 7  # relative 1 is the Assassin
    assert np.flatnonzero(ctx["judge_ok"][1]).tolist() == [P.J_NONE, *P.EVIL_TALK_JUDGMENTS]
    assert not ctx["claim_ok"][1:].any() and not ctx["team_ok"][1:].any() and not ctx["veto"]
    say(game, judge2=P.J_MERLIN)  # Morgana: seat 0 is Merlin
    assert game.acting_players() == [5]
    say(game)
    assert game.phase == Phase.ASSASSINATE
    slots = P.encode_comm(game, 5)[P.COMM_LAYOUT["messages"]].reshape(P.N_MSG_SLOTS, P.MSG_FEAT_DIM)
    morgana = slots[P.N_QUESTS * 6 + 5]  # evil talk, Morgana is relative seat 5 of the Assassin
    assert np.flatnonzero(morgana[P.MSG_LAYOUT["judge"]]).tolist() == [P._SEAT_FIELDS["judge"] * 1 + P.J_MERLIN - 1]
    game.step({5: A.assassin_action(5, 0)})
    assert game.done and not game.good_won
    assert game.to_record()["evil_talk"] == [{"speaker": 4, "judgments": {0: {"judgment": "merlin"}}}, {"speaker": 5}]


def test_silenced_slots_look_like_saying_nothing():
    silenced = make_game(comm=P.CommConfig(enabled=True, silence=list(P.SLOTS)))
    reference = make_game(comm=F)
    for game in (silenced, reference):
        for _ in range(3):
            assert (P.forced_message(P.message_context(game, game.acting_players()[0])) is not None) == (game is silenced)
            say(game)
    assert P.encode_comm(silenced, 4)[P.COMM_LAYOUT["slot_open"]].all()
    assert np.array_equal(P.encode_comm(silenced, 4), P.encode_comm(reference, 4))


@pytest.mark.parametrize("cfg", [F, M])
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
    games = [AvalonGame(rng, comm=F) for _ in range(16)]
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
    games = [AvalonGame(rng, comm=F) for _ in range(8)]
    for _ in range(40):  # advance through several discussions with random messages
        play_one_tick(games, rng)
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
    assert {"good/says_judgment", "good/says_claim", "good/says_veto", "evil/says_judgment",
            "good/msg_entropy"} <= set(lines[1])
    assert load_checkpoint(run_dir / "latest.pt")["config"]["comm"]["enabled"]
    assert load_policies(run_dir / "latest.pt")["good"].comm
    train(load_config(PILOT, {**comm_over, "total_games": 96, "resume": True}))
    lines = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[-1]["iteration"] == 3
