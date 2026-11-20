import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.play_web import CONSENT_VERSION, GameLog, HumanTable, SessionExpired, TableSessions, action_options, handler_for, new_game
from src.agents import RuleController
from src.env import actions as A
from src.env.rules import Phase, Role, WinReason
from helpers import make_game, play_quest, propose


@pytest.fixture
def table():
    table = HumanTable()
    rule = RuleController()
    table.models[0] = {"good": rule, "evil": rule}
    table.start({"role": "servant", "seat": 3, "share": False})
    return table


def request(table, **extra):
    return {"id": table.id, "revision": table.revision, **extra}


def test_only_human_information_is_sent_before_game_over(table):
    table.game = make_game()
    play_quest(table.game, [0, 4], failers=[4])
    propose(table.game, [0, 1, 2])
    before = table.view()
    table.game.roles = table.game.roles.copy()
    table.game.roles[0], table.game.roles[4] = table.game.roles[4], table.game.roles[0]
    table.game.quest_cards[0, 0], table.game.quest_cards[4, 0] = 1, 0
    table.game.prop_votes[1, 0] = True  # Not yet settled.
    assert before == table.view()
    assert "result" not in before and "roles" not in before
    assert "quest_cards" not in json.dumps(before)
    assert before["history"][-1]["votes"] is None
    with pytest.raises(ValueError, match="结束"):
        table.replay()
    # Percival must not be able to distinguish the two candidates, even via HTTP.
    table.human = 1
    before = table.view()
    table.game.roles = table.game.roles.copy()
    table.game.roles[0], table.game.roles[4] = table.game.roles[4], table.game.roles[0]
    assert table.view() == before


def test_invalid_or_duplicate_actions_do_not_advance_game(table):
    table.game = make_game(leader=table.human)
    initial = table.view()
    for data in (request(table, action=-1), request(table, action=True),
                 request(table, action=0, revision=-1)):
        with pytest.raises(ValueError):
            table.step(data)
        assert table.view() == initial
    with pytest.raises(ValueError):
        table.step(request(table), advance=True)
    valid = request(table, action=initial["options"][0]["action"])
    table.step(valid)
    assert table.game.phase == Phase.VOTE
    with pytest.raises(ValueError, match="已更新"):
        table.step(valid)
    assert table.game.phase == Phase.VOTE


@pytest.mark.parametrize("role", [r.name.lower() for r in Role])
def test_all_roles_can_complete_a_browser_game(table, role):
    table.start({"role": role, "seat": 5, "share": False})
    assert Role(int(table.game.roles[4])).name.lower() == role
    phases = set()
    for _ in range(160):
        view = table.view()
        json.dumps(view)  # Every state must cross the JSON boundary.
        if view["done"]:
            break
        phases.add(view["phase"])
        if view["human_turn"]:
            table.step(request(table, action=view["options"][0]["action"]))
        else:
            table.step(request(table), advance=True)
    assert table.game.done
    assert {"PROPOSE", "VOTE"} <= phases
    assert len(table.view()["result"]["roles"]) == 6
    assert table.replay()["game"]["good_won"] is not None


def test_stale_tab_cannot_act_on_replacement_game(table):
    old = request(table, action=0)
    table.start({"role": "merlin", "seat": 1, "share": False})
    with pytest.raises(ValueError, match="已更新"):
        table.step(old)


@pytest.fixture
def sessions():
    pool = TableSessions()
    rule = RuleController()
    pool.models[0] = {"good": rule, "evil": rule}
    return pool


def test_visitors_have_separate_identities_games_and_replays(sessions):
    a, sid_a = sessions.request(None, "/api/new", {"role": "merlin", "seat": 1, "share": False})
    b, sid_b = sessions.request(None, "/api/new", {"role": "assassin", "seat": 4, "share": False})
    assert sid_a != sid_b and a["id"] != b["id"]
    assert a["role"] == "梅林" and b["role"] == "刺客"
    # A leaked public game identifier alone cannot grant access to another visitor's table.
    with pytest.raises(ValueError, match="已更新"):
        sessions.request(sid_b, "/api/action", {"id": a["id"], "revision": a["revision"], "action": 0})
    assert sessions.request(sid_a, "/api/state")[0] == a
    sessions.request(sid_b, "/api/new", {"role": "servant", "share": False})
    assert sessions.request(sid_a, "/api/state")[0] == a
    table_a = sessions.tables[sid_a][0]
    table_b = sessions.tables[sid_b][0]
    assert table_a.models is table_b.models  # Frozen weights are shared, not reloaded per visitor.
    table_a.game._finish(True, reason=WinReason.ASSASSIN_MISSED)
    assert sessions.request(sid_a, "/api/replay")[0]["game"]["good_won"]
    with pytest.raises(ValueError, match="结束"):
        sessions.request(sid_b, "/api/replay")


def test_expiry_and_capacity_do_not_evict_active_visitors(sessions):
    now = [0.0]
    sessions.clock = lambda: now[0]
    sessions.ttl, sessions.capacity, sessions.idle_evict = 100, 2, 10
    a, sid_a = sessions.request(None, "/api/new", {"role": "merlin", "share": False})
    _, sid_b = sessions.request(None, "/api/new", {"role": "servant", "share": False})
    with pytest.raises(RuntimeError, match="已满"):
        sessions.request(None, "/api/new", {"role": "servant", "share": False})
    now[0] = 8
    sessions.request(sid_a, "/api/state")
    now[0] = 12  # b idle 12 s >= idle_evict, a only 4 s: the newcomer replaces b.
    _, sid_c = sessions.request(None, "/api/new", {"role": "servant", "share": False})
    assert set(sessions.tables) == {sid_a, sid_c}
    with pytest.raises(SessionExpired):
        sessions.request(sid_b, "/api/advance", {"id": a["id"], "revision": 0})
    now[0] = 200  # Past the TTL every table expires.
    with pytest.raises(SessionExpired):
        sessions.request(sid_a, "/api/advance", {"id": a["id"], "revision": 0})


def test_page_loads_and_bad_requests_create_no_tables(sessions):
    sessions.capacity = 1
    for _ in range(5):
        view, sid = sessions.request(None, "/api/state")
        assert sid is None and not view["active"]
    for bad in ({"role": "servant"}, {"role": "nobody", "share": True}, {"seat": 9, "share": True}):
        with pytest.raises(ValueError):
            sessions.request(None, "/api/new", bad)
    assert not sessions.tables
    sessions.request(None, "/api/new", {"role": "servant", "share": False})  # Capacity is still free.


def finish(sessions, sid, view):
    while not view["done"]:
        payload = {"id": view["id"], "revision": view["revision"]}
        if view["human_turn"]:
            view, _ = sessions.request(sid, "/api/action", {**payload, "action": view["options"][0]["action"]})
        else:
            view, _ = sessions.request(sid, "/api/advance", payload)
    return view


def test_only_consenting_visitors_are_logged_once_per_finished_game(sessions, tmp_path):
    path = tmp_path / "human.jsonl"
    sessions.log = GameLog(path)
    no, sid_no = sessions.request(None, "/api/new", {"role": "merlin", "seat": 2, "share": False})
    finish(sessions, sid_no, no)
    assert not path.exists()
    yes, sid_yes = sessions.request(None, "/api/new", {"role": "assassin", "seat": 5, "share": True})
    sessions.request(None, "/api/new", {"role": "servant", "share": True})  # Abandoned: never logged.
    end = finish(sessions, sid_yes, yes)
    with pytest.raises(ValueError):
        sessions.request(sid_yes, "/api/advance", {"id": end["id"], "revision": end["revision"]})
    again, _ = sessions.request(sid_yes, "/api/new", {"role": "random", "share": True})
    finish(sessions, sid_yes, again)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    first, second = lines
    assert first["player"] == second["player"] and (first["player_game"], second["player_game"]) == (1, 2)
    assert first["human_seat"] == 4 and first["human_role"] == "ASSASSIN"
    assert first["human_decisions"] and all("seconds" in d for d in first["human_decisions"])
    assert first["game"]["roles"][4] == "ASSASSIN" and first["consent_version"] == CONSENT_VERSION
    assert sid_yes not in json.dumps(lines)  # The session cookie never reaches the data file.
    assert sessions.request(sid_yes, "/api/replay")[0]["table_id"] == second["table_id"]


@pytest.mark.parametrize("seat", range(6))
def test_proposal_and_assassination_labels_use_absolute_seats(seat):
    game = make_game(leader=seat)
    for action, label in action_options(game, seat):
        team = A.decode_team(seat, action, 2)
        assert label == "组队 [" + "、".join(str(p + 1) for p in team) + "]"
    game.roles = np.roll(game.roles, seat - 5)
    game.phase = Phase.ASSASSINATE
    assert game.seat_of(Role.ASSASSIN) == seat
    options = action_options(game, seat)
    assert len(options) == 4
    for action, label in options:
        target = A.to_absolute(seat, action - A.ASSASSIN_OFFSET)
        assert not game.is_evil_seat(target)
        assert label == f"刺杀 {target + 1} 号"


@pytest.mark.parametrize("role", [r.name.lower() for r in Role])
def test_requested_role_is_dealt_to_the_human_seat(role):
    for seat in range(6):
        game = new_game(7, seat, role)
        assert Role(int(game.roles[seat])).name.lower() == role
        assert sorted(game.roles.tolist()) == sorted(new_game(7, seat, "random").roles.tolist())


def test_configured_entry_points_reject_unrelated_hosts_and_origins(sessions):
    handler = handler_for(sessions, extra_hosts=["192.168.1.8"], public_origin="https://avalon.example.com")

    def allowed(host, origin=None):
        headers = {"Host": host}
        if origin is not None:
            headers["Origin"] = origin
        return handler.same_origin(SimpleNamespace(headers=headers, server=SimpleNamespace(server_port=8765),
                                                   public_origin=handler.public_origin))

    assert allowed("127.0.0.1:8765", "http://127.0.0.1:8765")
    assert allowed("192.168.1.8:8765", "http://192.168.1.8:8765")
    assert allowed("avalon.example.com", "https://avalon.example.com")
    assert allowed("localhost:8765", "https://avalon.example.com")  # Tunnel rewrites Host.
    assert not allowed("unrelated.example.com")
    assert not allowed("127.0.0.1:8765", "https://unrelated.example.com")
    assert not allowed("avalon.example.com.evil.test", "https://avalon.example.com.evil.test")
