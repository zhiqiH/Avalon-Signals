import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

import numpy as np
import pytest

from scripts.assassin_advisor import Store, advise, handler_for, load_assassins, outcome, owner_of, replay
from src.env import actions as A
from src.env.game import AvalonGame
from src.env.observation import encode
from src.env.rules import Phase, Role


def random_game_to_assassination(seed):
    """Random legal play (evil fail at random) until good completes three quests."""
    rng = np.random.default_rng(seed)
    while True:
        game = AvalonGame(rng)
        while game.phase not in (Phase.ASSASSINATE, Phase.DONE):
            acts = {}
            for p in game.acting_players():
                legal = np.flatnonzero(game.legal_mask(p))
                if game.phase == Phase.VOTE:  # approve often enough that quests get played
                    acts[p] = A.APPROVE if rng.random() < 0.7 else A.REJECT
                else:
                    acts[p] = int(rng.choice(legal))
            game.step(acts)
        if game.phase == Phase.ASSASSINATE:
            return game


def to_record(game):
    """What a person would type into the page (UI seats 1-6), plus who played each ambiguous Fail."""
    rec = {"first_leader": int(game.prop_leader[0, 0]) + 1, "proposals": [], "quest_fails": [],
           "assassin": game.seat_of(Role.ASSASSIN) + 1, "morgana": game.seat_of(Role.MORGANA) + 1}
    for q in range(5):
        for a in range(5):
            if game.prop_leader[q, a] < 0:
                continue
            rec["proposals"].append({"team": [int(s) + 1 for s in np.flatnonzero(game.prop_team[q, a])],
                                     "approvals": [int(s) + 1 for s in np.flatnonzero(game.prop_votes[q, a])]})
        if game.quest_result[q] >= 0:
            rec["quest_fails"].append(int(game.quest_fails[q]))
    who = {q: "assassin" if game.quest_cards[game.seat_of(Role.ASSASSIN), q] == 1 else "morgana" for q in range(5)}
    return rec, who


def test_replay_rebuilds_the_assassins_exact_observation():
    ambiguous_seen = 0
    for seed in range(60):
        game = random_game_to_assassination(seed)
        rec, who = to_record(game)
        rebuilt, ambiguous = replay(rec)
        rebuilt, _ = replay(rec, {q: who[q] for q in ambiguous})
        ambiguous_seen += bool(ambiguous)
        assassin = game.seat_of(Role.ASSASSIN)
        assert rebuilt.phase == Phase.ASSASSINATE
        np.testing.assert_array_equal(encode(rebuilt, assassin), encode(game, assassin))
        assert outcome(rec) == "刺杀阶段"
    assert ambiguous_seen  # the two-evil single-Fail case was exercised


def test_replay_rejects_fail_cards_the_evil_seats_cannot_explain():
    rec = {"first_leader": 1, "proposals": [{"team": [1, 2], "approvals": [1, 2, 3, 4]}], "quest_fails": [1],
           "assassin": 5, "morgana": 6}
    with pytest.raises(ValueError, match="矛盾"):
        replay(rec)
    with pytest.raises(ValueError, match="需要 2 人"):
        replay({**rec, "proposals": [{"team": [1, 2, 3], "approvals": []}], "quest_fails": []})


@pytest.fixture(scope="module")
def nets():
    return load_assassins()


def test_advice_is_a_distribution_over_good_seats(nets):
    for seed in range(5):
        game = random_game_to_assassination(seed)
        rec, _ = to_record(game)
        adv = advise(rec, nets)
        mean = np.array(adv["mean"])
        evil = [rec["assassin"] - 1, rec["morgana"] - 1]
        assert mean[evil].sum() == 0 and mean.sum() == pytest.approx(1, abs=1e-3)
        assert adv["pick"] == int(np.argmax(mean)) + 1 and adv["pick"] - 1 not in evil
        assert set(adv["per_seed"]) == {"0", "1", "2"}
        _, ambiguous = replay(rec)
        assert len(adv["variants"]) == (2 ** len(ambiguous) if ambiguous else 0)


def test_store_saves_advises_and_scores(tmp_path, nets):
    store = Store(tmp_path, nets)
    rec, _ = to_record(random_game_to_assassination(1))
    rec.update(id="0123456789abcdef", note="test")
    saved = store.advise(rec, "me")
    assert "owner" not in saved and store.load(rec["id"], "me")["advice"] == saved["advice"]
    merlin = saved["advice"]["ranking"][0]
    store.save({**saved, "actual_merlin": merlin, "human_target": saved["advice"]["ranking"][1]}, "me")
    summary = store.summary("me")
    stats = summary["stats"]
    assert {k: v["n"] for k, v in stats.pop("per_seed").items()} == {"0": 1, "1": 1, "2": 1}
    assert stats == {"games": 1, "judged": 1, "ai_hits": 1, "human_judged": 1, "human_hits": 0,
                     "mean_p_true": pytest.approx(saved["advice"]["mean"][merlin - 1], abs=1e-4)}
    assert summary["shared"] == {"games": 0, "judged": 0}  # not shared yet
    with pytest.raises(ValueError):
        store.save({**rec, "id": "../escape"}, "me")


def test_visitors_see_only_their_games_and_shared_stats_pool_everyone(tmp_path, nets):
    store = Store(tmp_path, nets)
    rec, _ = to_record(random_game_to_assassination(2))
    for i, (owner, share) in enumerate([("a", True), ("b", True), ("b", False)]):
        saved = store.advise({**rec, "id": f"{i:016x}"}, owner)
        store.save({**saved, "actual_merlin": saved["advice"]["pick"], "share": share}, owner)
    assert [g["id"] for g in store.summary("a")["games"]] == [f"{0:016x}"]
    assert store.summary("b")["stats"]["games"] == 2
    assert store.summary("c")["shared"]["judged"] == 2 and store.summary("c")["games"] == []
    with pytest.raises(FileNotFoundError):
        store.load(f"{1:016x}", "a")
    with pytest.raises(ValueError, match="找不到"):
        store.save({**rec, "id": f"{1:016x}"}, "a")  # cannot overwrite b's game
    with pytest.raises(FileNotFoundError):
        store.delete(f"{1:016x}", "a")
    with pytest.raises(ValueError, match="共享"):
        store.save({**rec, "id": f"{3:016x}", "share": "yes"}, "a")


def test_orphan_records_get_one_legacy_code(tmp_path):
    games = tmp_path / "games"
    games.mkdir()
    (games / "00000000000000aa.json").write_text(json.dumps({"id": "00000000000000aa", "proposals": []}))
    store = Store(games, {})
    code = store.adopt_orphans(tmp_path / "code.txt")
    assert store.load("00000000000000aa", owner_of(code))["id"] == "00000000000000aa"
    assert store.adopt_orphans(tmp_path / "code.txt") == code


def test_http_cookie_gives_each_browser_its_own_records(tmp_path):
    store = Store(tmp_path, {})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(store))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"

    def call(path, body=None, cookie=None):
        req = Request(base + path, data=None if body is None else json.dumps(body).encode(),
                      headers={"Cookie": cookie} if cookie else {})
        with urlopen(req) as r:
            return json.loads(r.read()), r.headers.get("Set-Cookie")
    try:
        me, set_cookie = call("/api/me")
        cookie = set_cookie.split(";")[0]
        assert cookie == f"advisor_code={me['code']}"
        rec = {"id": "00000000000000bb", "first_leader": 1, "proposals": [], "quest_fails": []}
        call("/api/save", rec, cookie)
        assert len(call("/api/games", cookie=cookie)[0]["games"]) == 1
        assert call("/api/games")[0]["games"] == []  # a fresh browser
        _, switched = call("/api/code", {"code": me["code"].upper()}, "advisor_code=" + "1" * 16)
        assert switched.startswith(cookie)
    finally:
        server.shutdown()
        server.server_close()
