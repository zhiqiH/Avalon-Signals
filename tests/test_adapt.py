"""LoRA adaptation, the listener stage's frozen speaker and the message KL anchor."""
import json
from pathlib import Path

import numpy as np
import torch

from src.agents import ActorCritic, PolicyController
from src.agents.policy import heads_kl, load_silent_weights
from src.comm.protocol import COMM_DIM
from src.config import load_config
from src.env import AvalonGame
from src.env import actions as A
from src.runner import play
from src.training import TrajectoryRecorder
from src.training.adapt import apply_lora, plain_state_dict, trainable_parameters
from src.training.checkpoint import load_checkpoint, load_policies
from src.training.trainer import train

PILOT = Path(__file__).resolve().parents[1] / "configs" / "pilot_silent.yaml"


def branched() -> ActorCritic:
    torch.manual_seed(0)
    net = ActorCritic(comm=True)
    load_silent_weights(net, ActorCritic().state_dict())
    return net


def rank(m: torch.Tensor) -> int:
    """Numerical rank, ignoring float32 round-off of the merged weights."""
    return int(torch.linalg.matrix_rank(m.double(), rtol=1e-4))


def test_lora_starts_unchanged_and_changes_the_baseline_only_in_low_rank():
    net = branched()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    apply_lora(net, rank=4)
    assert all(torch.equal(plain_state_dict(net)[k], before[k]) for k in before)
    opt = torch.optim.SGD(trainable_parameters(net), lr=1.0)
    obs = torch.randn(64, net.obs_dim)
    comm = torch.randint(0, 2, (64, COMM_DIM), dtype=torch.int8)
    for _ in range(3):
        opt.zero_grad()
        ((net.logits(obs, comm) - 3).pow(2).mean() + (net.value(obs, comm) - 1).pow(2).mean()).backward()
        opt.step()
    after = plain_state_dict(net)
    delta = after["actor.2.weight"] - before["actor.2.weight"]
    assert 0 < rank(delta) <= 4  # the hidden layer moves only in rank 4
    first = after["actor.0.weight"] - before["actor.0.weight"]
    assert rank(first[:, :net.obs_dim]) <= 4  # baseline inputs: low rank
    assert rank(first[:, net.obs_dim:]) > 4  # message inputs: free
    assert torch.equal(after["actor.0.bias"], before["actor.0.bias"])  # baseline biases frozen
    assert torch.equal(after["actor.4.bias"][:A.N_ACTIONS], before["actor.4.bias"][:A.N_ACTIONS])
    assert any(not torch.equal(after[k], before[k]) for k in after if k.startswith("critic"))  # critic trains fully


def test_a_frozen_speaker_speaks_and_its_messages_are_not_training_samples():
    learner_net, speaker_net = branched(), ActorCritic(comm=True)
    speaker = PolicyController(speaker_net, learn=False)
    learner = PolicyController(learner_net, learn=True, speaker=speaker)
    free = PolicyController(branched(), learn=True)
    for ctrl, expect_msgs in ((learner, False), (free, True)):
        rec = TrajectoryRecorder()
        games = [AvalonGame(np.random.default_rng(i), comm=load_config(PILOT, {"comm.enabled": True}).comm) for i in range(4)]
        play(games, [ctrl] * 4, [ctrl] * 4, np.random.default_rng(0), on_decision=rec)
        batches = rec.build(games, 1.0, 0.95)
        has_msgs = any((b.actions[:, 0] < 0).any() for b in batches.values())
        assert has_msgs == expect_msgs


def test_message_kl_is_zero_against_itself_and_positive_otherwise():
    net = branched()
    logits = torch.randn(8, sum(net.head_sizes))
    masks = torch.ones_like(logits, dtype=torch.bool)
    heads = torch.zeros(8, len(net.head_sizes), dtype=torch.long)
    heads[:, 0] = -1
    assert torch.allclose(heads_kl(logits, logits, masks, heads, net.head_sizes), torch.zeros(8), atol=1e-6)
    assert (heads_kl(logits, torch.zeros_like(logits), masks, heads, net.head_sizes) > 0).all()


def test_listener_stage_with_lora_and_anchor_runs_saves_merged_weights_and_resumes(tmp_path):
    base = {"out_dir": str(tmp_path), "device": "cpu", "games_per_iter": 32, "eval.every": 1, "eval.games": 16,
            "ppo.minibatch_size": 256, "checkpoint_every": 1, "opponents.snapshot_every": 1}
    train(load_config(PILOT, {**base, "name": "silent", "total_games": 32}))
    comm = {**base, "comm.enabled": True, "comm.claim": False, "comm.team": False, "comm.intent": False}
    train(load_config(PILOT, {**comm, "name": "teacher", "total_games": 32,
                              "init_from": str(tmp_path / "silent" / "seed{seed}" / "latest.pt")}))
    teacher = str(tmp_path / "teacher" / "seed{seed}" / "latest.pt")
    stage = {**comm, "name": "listen", "total_games": 64, "init_from": teacher, "adapt.lora_rank": 4,
             "adapt.speaker_from": teacher, "adapt.msg_kl_coef": 0.1, "adapt.msg_kl_anneal": 4, "adapt.anchor_from": teacher,
             "adapt.msg_kl_floor": 0.06}
    run_dir = train(load_config(PILOT, stage))
    lines = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[1]["msg_kl_coef"] == 0.1 and lines[2]["msg_kl_coef"] == 0.075
    assert "good/says_judgment" not in lines[1]  # the speaker's messages are not samples of the listener
    ck = load_checkpoint(run_dir / "latest.pt")
    assert ck["adapter"]["good"] is not None
    nets = load_policies(run_dir / "latest.pt")  # merged weights load into a plain network
    assert nets["good"].comm
    train(load_config(PILOT, {**stage, "total_games": 96, "resume": True}))
    lines = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[-1]["iteration"] == 3 and lines[-1]["msg_kl_coef"] == 0.06  # 0.05 by the anneal: held at the floor


def test_silent_lora_control_trains_only_low_rank_updates(tmp_path):
    base = {"out_dir": str(tmp_path), "device": "cpu", "games_per_iter": 32, "eval.every": 1, "eval.games": 16,
            "ppo.minibatch_size": 256, "checkpoint_every": 1}
    train(load_config(PILOT, {**base, "name": "silent", "total_games": 32}))
    start = load_policies(tmp_path / "silent" / "seed0" / "latest.pt")["good"].state_dict()
    run_dir = train(load_config(PILOT, {**base, "name": "silent_lora", "total_games": 64, "adapt.lora_rank": 2,
                                        "init_from": str(tmp_path / "silent" / "seed{seed}" / "latest.pt")}))
    after = load_policies(run_dir / "latest.pt")["good"].state_dict()
    assert 0 < rank(after["actor.2.weight"] - start["actor.2.weight"]) <= 2
    assert torch.equal(after["actor.4.bias"], start["actor.4.bias"])


def test_message_entropy_coefficient_is_separate_from_the_game_one():
    import dataclasses
    from src.training.ppo import PPOConfig, ppo_update
    ctrl = PolicyController(branched(), learn=True)
    rec = TrajectoryRecorder()
    games = [AvalonGame(np.random.default_rng(i), comm=load_config(PILOT, {"comm.enabled": True}).comm) for i in range(4)]
    play(games, [ctrl] * 4, [ctrl] * 4, np.random.default_rng(0), on_decision=rec)
    b = rec.build(games, 1.0, 0.95)["good"]
    msg = b.actions[:, 0] < 0
    b = dataclasses.replace(b, **{f.name: getattr(b, f.name)[msg] for f in dataclasses.fields(b)})
    b.advantages[:] = 0  # no policy gradient: only the entropy bonus can move the actor
    for coef, moves in ((None, True), (0.0, False)):
        net = branched()
        before = net.actor[-1].weight.detach().clone()
        cfg = PPOConfig(epochs=1, msg_ent_coef=coef)
        ppo_update(net, torch.optim.Adam(net.parameters(), lr=1e-3), b, cfg, torch.device("cpu"), np.random.default_rng(0))
        assert (not torch.equal(net.actor[-1].weight, before)) == moves
