import json

from avalon_rl.agents import RandomController
from avalon_rl.comm import CommConfig
from avalon_rl.evaluation import run_matchup
from avalon_rl.evaluation.messages import message_stats

RECORD = {  # as saved by scripts/evaluate.py: seat keys are strings
    "roles": ["SERVANT", "MERLIN", "PERCIVAL", "SERVANT", "MORGANA", "ASSASSIN"],
    "proposals": [{"quest": 0, "attempt": 0, "leader": 0, "team": [0, 1], "approvals": [0, 1, 2, 3], "approved": True}],
    "discussions": [{"quest": 0, "leader": 0, "messages": [
        {"speaker": 1, "judgments": {"4": {"judgment": "evil", "basis": "role_knowledge"}, "5": {"judgment": "evil"},
                                     "2": {"judgment": "good"}}},
        {"speaker": 4, "judgments": {"1": {"judgment": "evil"}}},
        {"speaker": 0},
    ]}],
    "evil_talk": [{"speaker": 4, "judgments": {"1": {"judgment": "merlin"}}}, {"speaker": 5}],
    "assassin_target": 1,
}


def rate(r):
    return r["k"], r["n"]


def test_message_statistics_of_a_hand_checked_game():
    s = message_stats([json.loads(json.dumps(RECORD))])
    assert s["messages"] == 3
    assert rate(s["accusation_precision"]["good"]) == (2, 2) and rate(s["accusation_precision"]["evil"]) == (0, 1)
    assert rate(s["endorsement_precision"]["MERLIN"]) == (1, 1)
    assert s["basis_share"]["good"]["role_knowledge"] == 1 / 3
    assert rate(s["leader_includes"]["leader_good/accused"]) == (1, 3)  # seats 1, 4, 5; only 1 is on the team
    assert rate(s["leader_includes"]["leader_good/not_accused"]) == (0, 2)
    approval = s["first_proposal_approval"]
    assert rate(approval["voter_good/with_accused"]) == (3, 3) and rate(approval["voter_good/without_accused"]) == (1, 1)
    assert rate(approval["voter_evil/with_accused"]) == (0, 2)
    talk = s["evil_talk"]
    assert rate(talk["assassin_follows"]["assassin_targets_a_named_merlin"]) == (1, 1)
    assert rate(talk["hit_when_following"]["hit"]) == (1, 1)


def test_random_speakers_accuse_at_chance():
    m = CommConfig(enabled=True, claim=False, team=False, intent=False)
    _, records = run_matchup(RandomController(), RandomController(), 300, 3, keep_records=True, comm=m)
    s = message_stats(records)
    assert abs(s["accusation_precision"]["good"]["rate"] - 0.4) < 0.05
    assert abs(s["accusation_precision"]["evil"]["rate"] - 0.2) < 0.05
