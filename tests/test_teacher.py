"""Rule teacher for the message slots and its distillation into the message heads."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from src.agents import ActorCritic, PolicyController, RuleController
from src.agents.base import ActResult
from src.agents.policy import load_silent_weights
from src.agents.teacher import TeacherSpeaker, good_view
from src.comm import protocol as P
from src.env import AvalonGame, Phase, Role
from src.env import actions as A
from src.runner import play
from src.training.checkpoint import load_checkpoint, load_policies

M = P.CommConfig(enabled=True, claim=False, team=False, intent=False)
FULL = P.CommConfig(enabled=True)


class Spy:
    """Plays like the rule player, speaks as the teacher, and keeps every (game, speaker, message)."""

    learn = False

    def __init__(self, teacher: TeacherSpeaker):
        self.teacher, self.rule, self.log = teacher, RuleController(), []

    def act(self, requests, rng):
        return self.rule.act(requests, rng)

    def speak(self, requests, rng):
        msgs = self.teacher.speak(requests, rng).messages
        for r, m in zip(requests, msgs):
            P.check_message(r.game, r.player, m)  # legal in the state it was sent from
            teams = P.head_mask(P.H_TEAM, m[None].astype(np.int64), P.stack_contexts([P.message_context(r.game, r.player)]))[0]
            self.log.append((r.game.phase, int(r.game.roles[r.player]), r.player, r.game.private_knowledge(r.player)[0].copy(), m,
                             r.game, r.game.quest, r.player == r.game.leader, good_view(r.game, r.player), teams))
        return ActResult(messages=msgs)


def play_with_teacher(comm, n=60, knowledge=0.5, seed=0):
    spy = Spy(TeacherSpeaker(merlin_knowledge_prob=knowledge, seed=seed))
    games = [AvalonGame(np.random.default_rng([seed, i]), comm=comm) for i in range(n)]
    play(games, [spy] * n, [spy] * n, np.random.default_rng(seed))
    return games, spy.log


def test_teacher_messages_are_always_legal_in_m_and_f():
    for comm in (M, FULL):
        games, log = play_with_teacher(comm)
        assert all(g.done for g in games) and len(log) > 0


def test_evil_teacher_never_accuses_its_teammate():
    _, log = play_with_teacher(M, n=150)
    accusations = 0
    for phase, role, speaker, known_evil, msg, *_ in log:
        if role in (Role.MORGANA, Role.ASSASSIN) and phase == Phase.DISCUSS:
            for r, seat in enumerate(P._others(speaker)):
                if known_evil[seat]:
                    assert msg[P.H_JUDGE + r] not in P.EVIL_JUDGMENTS
                accusations += msg[P.H_JUDGE + r] in P.EVIL_JUDGMENTS
    assert accusations > 0  # evil players do accuse (good players)


def test_merlin_without_knowledge_speaks_from_the_good_players_view():
    teacher = TeacherSpeaker(merlin_knowledge_prob=0.0)
    for i in range(40):
        game = AvalonGame(np.random.default_rng(i), comm=M)
        play([game], [RuleController()], [RuleController()], np.random.default_rng(i))  # a finished history
        merlin = game.seat_of(Role.MERLIN)
        assert np.array_equal(teacher.scores(game, merlin), good_view(game, merlin))


def test_merlin_with_knowledge_says_what_a_good_player_could_minus_what_he_knows_is_false():
    games, log = play_with_teacher(M, n=150, knowledge=1.0)
    said = 0
    for phase, role, speaker, known_evil, msg, *_ in log:
        if role == Role.MERLIN and phase == Phase.DISCUSS:
            for r, seat in enumerate(P._others(speaker)):
                judged = int(msg[P.H_JUDGE + r])
                assert not (judged in P.EVIL_JUDGMENTS and not known_evil[seat])  # never accuses a good player
                assert not (judged in P.GOOD_JUDGMENTS and known_evil[seat])  # never vouches for an evil one
                said += judged > P.J_NONE
    assert said > 0
    teacher, game = TeacherSpeaker(merlin_knowledge_prob=1.0), games[0]
    merlin = game.seat_of(Role.MERLIN)
    view, s = good_view(game, merlin), teacher.scores(game, merlin)
    quiet = (view < teacher.accuse_at) & (view > teacher.vouch_at)
    assert np.array_equal(s[quiet], view[quiet])  # where the good player's view says nothing, neither does he


def test_nobody_judges_before_the_first_quest_is_resolved():
    games, log = play_with_teacher(M, n=40)
    assert all((g.msg_log[0, :, P.H_JUDGE:P.H_RTYPE] <= 0).all() for g in games)


def suggested(msg, speaker: int, size: int) -> list[int]:
    return list(A.decode_team(speaker, int(msg[P.H_TEAM]) - 1, size)) if msg[P.H_TEAM] > 0 else []


def test_in_the_first_round_everybody_claims_a_role_and_all_but_the_leader_suggest_a_team():
    games, log = play_with_teacher(FULL, n=200)
    claims = {role: [] for role in Role}
    for phase, role, speaker, known_evil, msg, game, quest, leads, view, teams in log:
        if phase != Phase.DISCUSS or quest > 0:
            continue
        claims[Role(role)].append(int(msg[P.H_CLAIM]))
        assert (msg[P.H_JUDGE:P.H_RTYPE] == P.J_NONE).all()  # nothing on record yet
        team = suggested(msg, speaker, 2)
        assert team == [] if leads else (len(team) == 2 and speaker in team)
    for role in (Role.SERVANT, Role.MERLIN, Role.ASSASSIN):
        assert set(claims[role]) == {P.C_SERVANT}  # Merlin hides among the Servants
    for role in (Role.PERCIVAL, Role.MORGANA):  # Percival says so in half the games; Morgana pretends as often
        assert set(claims[role]) == {P.C_SERVANT, P.C_PERCIVAL}
        assert 0.35 < np.mean(np.array(claims[role]) == P.C_PERCIVAL) < 0.65
    assert all((g.msg_log[1:, :, P.H_CLAIM] <= 0).all() for g in games)  # said once: a claim stands


def test_suggestions_and_intents_follow_what_the_speaker_knows():
    _, log = play_with_teacher(FULL, n=200, knowledge=1.0)
    vetoes = requires = clean = 0
    for phase, role, speaker, known_evil, msg, game, quest, leads, view, teams in log:
        if phase != Phase.DISCUSS:
            continue
        size = P.QUEST_SIZES[quest]
        team = suggested(msg, speaker, size)
        mates = [s for s in team if s != speaker]
        if role in (Role.MERLIN, Role.MORGANA, Role.ASSASSIN):
            # Merlin names no player he knows to be evil, an evil player not its teammate, wherever the
            # protocol's rule (the cleanest teams by the record) leaves such a team.
            others = known_evil.copy()
            others[speaker] = False
            dirty = (P.TEAM_MEMBERS[size] & others[(speaker + np.arange(6)) % 6]).any(axis=1)
            if (teams & ~dirty).any():
                clean += 1
                assert not known_evil[mates].any()
        for r, seat in enumerate(P._others(speaker)):
            intent = int(msg[P.H_INTENT + r])
            assert intent != P.I_VETO or view[seat] >= 10.0  # only where deduction is certain
            assert intent != P.I_REQUIRE or view[seat] <= -10.0
            assert not (intent == P.I_VETO and role in (Role.MORGANA, Role.ASSASSIN) and known_evil[seat])
            vetoes += intent == P.I_VETO
            requires += intent == P.I_REQUIRE
    assert clean > 0 and vetoes > 0 and requires > 0


def test_players_speaking_as_the_teacher_never_act_against_their_words():
    games, _ = play_with_teacher(FULL, n=100)  # the masks bind them; the audit would count any breach
    assert all(g.done for g in games)
    assert not any(g.audit[:, [P.A_JUDGE, P.A_INTENT, P.A_TEAM]].any() for g in games)


def test_distilled_network_plays_exactly_like_the_silent_one(tmp_path):
    torch.manual_seed(0)
    silent = {t: ActorCritic() for t in ("good", "evil")}
    base = tmp_path / "silent.pt"
    torch.save({"config": {"model": {"hidden_sizes": [256, 256], "deduction": True}},
                "nets": {t: n.state_dict() for t, n in silent.items()}}, base)
    out = tmp_path / "distill"
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "scripts/distill_speaker.py"), "--baseline", str(base),
                    "--config", str(root / "configs/comm_F.yaml"), "--games", "12", "--epochs", "2", "--out", str(out)],
                   check=True, cwd=root, capture_output=True)
    ck = load_checkpoint(out / "checkpoint_00000.pt")
    assert ck["config"]["comm"]["enabled"]
    nets = load_policies(out / "checkpoint_00000.pt")
    for t in ("good", "evil"):
        branched = ActorCritic(comm=True)
        load_silent_weights(branched, silent[t].state_dict())
        state, ref = nets[t].state_dict(), branched.state_dict()
        # The message encoders keep their random initialisation (they meet zero weights), so they are not compared.
        trunk_keys = [k for k in ref if not k.startswith(("actor.4", "actor_cond", "actor_msg", "critic_msg"))]
        assert all(torch.equal(state[k], ref[k]) for k in trunk_keys)  # trunk, critic, listener inputs untouched
        assert torch.equal(state["actor.4.weight"][:A.N_ACTIONS], ref["actor.4.weight"][:A.N_ACTIONS])
        assert not torch.equal(state["actor.4.weight"][A.N_ACTIONS:], ref["actor.4.weight"][A.N_ACTIONS:])


def test_rule_players_suspect_whoever_acted_against_its_words():
    from src.agents.rule_agent import BROKEN_WORD, FAILED_VOUCH, public_suspicion

    game = AvalonGame(np.random.default_rng(0), comm=P.CommConfig(enabled=True))
    assert not public_suspicion(game).any()
    game.audit[2, P.A_JUDGE] = game.audit[2, P.A_TEAM] = 1
    game.audit[3, P.A_VOUCH] = 2
    assert public_suspicion(game).tolist() == [0, 0, 2 * BROKEN_WORD, 2 * FAILED_VOUCH, 0, 0]


F3 = P.CommConfig(enabled=True, reply=True, promise=True)


def test_teacher_replies_and_promises_follow_what_the_speaker_knows():
    games, log = play_with_teacher(F3, n=120, knowledge=1.0)
    assert all(g.done for g in games)
    assert not any(g.audit[:, [P.A_JUDGE, P.A_INTENT, P.A_TEAM, P.A_REPLY, P.A_PROMISE]].any() for g in games)
    counts = {"endorse": 0, "dispute": 0, "promise": 0}
    for phase, role, speaker, known_evil, msg, game, quest, leader, *_ in log:
        if phase != Phase.DISCUSS:
            continue
        replies = msg[P.H_REPLY:P.H_REPLY + P.N_OTHERS]
        promises = msg[P.H_PROMISE:P.H_PROMISE + P.N_OTHERS]
        counts["endorse"] += int((replies == P.Y_ENDORSE).sum())
        counts["dispute"] += int((replies == P.Y_DISPUTE).sum())
        counts["promise"] += int((promises > 0).sum())
        assert not leader or (replies == P.Y_ENDORSE).sum() <= 1  # a leader adopts at most one team
        others = P._others(speaker)
        if role in (Role.MORGANA, Role.ASSASSIN):  # never promises to accuse its teammate
            assert not (known_evil[others] & (promises == P.K_FAIL_EVIL)).any()
        if role == Role.MERLIN:  # the protocol's rule here: no promise to accuse a player he knows to be good
            assert not (~known_evil[others] & (promises == P.K_FAIL_EVIL)).any()
    assert min(counts.values()) > 0, counts


def test_teacher_merlin_promises_like_a_good_player_where_he_may_accuse_good_players():
    bluff = P.CommConfig(enabled=True, reply=True, promise=True, team_optional=True, merlin_may_accuse_good=True)
    games, log = play_with_teacher(bluff, n=200, knowledge=1.0)
    assert all(g.done for g in games)
    assert not any(g.audit[:, [P.A_JUDGE, P.A_INTENT, P.A_TEAM, P.A_REPLY, P.A_PROMISE]].any() for g in games)
    about_good = about_evil = 0
    for phase, role, speaker, known_evil, msg, *_ in log:
        if phase == Phase.DISCUSS and role == Role.MERLIN:
            promises, evil = msg[P.H_PROMISE:P.H_PROMISE + P.N_OTHERS], known_evil[P._others(speaker)]
            about_good += int((~evil & (promises == P.K_FAIL_EVIL)).sum())
            about_evil += int((evil & (promises == P.K_SUCCESS_GOOD)).sum())
    assert about_good > 0 and about_evil > 0  # the promises only a player without his sight would make


def test_where_suggestions_are_a_choice_the_teacher_names_a_team_only_with_a_reason():
    opt = P.CommConfig(enabled=True, reply=True, promise=True, team_optional=True)
    games, log = play_with_teacher(opt, n=120)
    assert all(g.done for g in games)
    named = {True: 0, False: 0}
    for phase, role, speaker, known_evil, msg, game, quest, leader, *_ in log:
        if phase != Phase.DISCUSS or leader:
            continue
        suggests = msg[P.H_TEAM] > 0
        named[bool(suggests)] += 1
        assert not (suggests and quest == 0)  # no record yet: no reason
        assert not (suggests and (msg[P.H_REPLY:P.H_REPLY + P.N_OTHERS] == P.Y_ENDORSE).any())  # it backs that team
    assert min(named.values()) > 0, named
