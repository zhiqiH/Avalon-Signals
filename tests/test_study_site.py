import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from scripts.assassin_advisor import Store
from scripts.play_web import TableSessions
from scripts.study_site import SiteLog, handler_for
from src.agents import RuleController


@pytest.fixture
def site(tmp_path):
    sessions = TableSessions(0, log=SiteLog(tmp_path / "games.jsonl"))
    rule = RuleController()
    sessions.models[0] = {"good": rule, "evil": rule}  # no checkpoint needed for the table itself
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(sessions, Store(tmp_path / "advisor", {})))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", tmp_path
    server.shutdown()
    server.server_close()


class Browser:
    """Keeps cookies like a browser would (one cookie per name)."""

    def __init__(self, base):
        self.base, self.cookies = base, {}

    def raw(self, path, body=None, headers=None):
        hdrs = {"Cookie": "; ".join(f"{k}={v}" for k, v in self.cookies.items())}
        if body is not None:
            hdrs["Content-Type"] = "application/json"
        hdrs.update(headers or {})
        req = Request(self.base + path, data=None if body is None else json.dumps(body).encode(), headers=hdrs)
        try:
            r = urlopen(req)
        except HTTPError as e:
            r = e
        for c in r.headers.get_all("Set-Cookie") or []:
            k, v = c.split(";")[0].split("=", 1)
            self.cookies[k] = v
        return r.status, r.read(), r.headers

    def json(self, path, body=None):
        status, data, _ = self.raw(path, body)
        return status, json.loads(data)


def test_pages_and_routes(site):
    base, _ = site
    b = Browser(base)
    status, home, headers = b.raw("/")
    assert status == 200 and "共享数据用于研究".encode() in home and b'src="/i18n.js"' in home
    assert "'unsafe-inline'" not in headers["Content-Security-Policy"]
    for path in ("/ui.css", "/i18n.js", "/home.js", "/play/app.js", "/advisor/advisor.js"):
        assert b.raw(path)[0] == 200, path
    status, page, _ = b.raw("/play/")
    assert status == 200 and b'href="/ui.css"' in page and b'src="app.js"' in page
    status, page, _ = b.raw("/advisor/")
    assert status == 200 and b'src="advisor.js"' in page
    assert b.json("/api/health") == (200, {"service": "avalon-site", "ready": True})
    assert b.raw("/play/api/new")[0] == 404  # starting a game is POST only
    assert b.raw("/nope")[0] == 404


def test_shared_table_game_is_logged_with_site_tag(site):
    base, tmp = site
    b = Browser(base)
    status, state = b.json("/play/api/new", {"role": "servant", "seat": 2, "share": True, "baseline": 0})
    assert status == 200 and state["active"]
    while not state["done"]:
        req = {"id": state["id"], "revision": state["revision"]}
        if state["human_turn"]:
            status, state = b.json("/play/api/action", {**req, "action": state["options"][0]["action"]})
        else:
            status, state = b.json("/play/api/advance", req)
        assert status == 200, state
    record = json.loads((tmp / "games.jsonl").read_text().splitlines()[0])
    assert record["entry"] == "site" and record["human_role"] == "SERVANT"


def test_advisor_under_prefix_and_cross_origin_post_is_refused(site):
    base, _ = site
    b = Browser(base)
    status, me = b.json("/advisor/api/me")
    assert status == 200 and b.cookies["advisor_code"] == me["code"]
    rec = {"id": "00000000000000cc", "first_leader": 1, "proposals": [], "quest_fails": [], "share": True}
    assert b.json("/advisor/api/save", rec)[0] == 200
    assert len(b.json("/advisor/api/games")[1]["games"]) == 1
    status, _, _ = b.raw("/advisor/api/save", rec, {"Origin": "https://evil.example"})
    assert status == 403
