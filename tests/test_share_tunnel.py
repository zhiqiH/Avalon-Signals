from io import StringIO
from unittest.mock import Mock
from urllib.error import URLError

import pytest

from scripts.share_tunnel import QuickTunnel


def test_share_uses_assigned_https_origin_and_owns_child_lifecycle(monkeypatch):
    process = Mock()
    process.stdout = StringIO("Starting tunnel\nhttps://small-quiet-table.trycloudflare.com\n")
    process.poll.return_value = None
    popen = Mock(return_value=process)
    monkeypatch.setattr("scripts.share_tunnel.shutil.which", lambda _: "/bin/cloudflared")
    monkeypatch.setattr("scripts.share_tunnel.subprocess.Popen", popen)
    tunnel = QuickTunnel("http://127.0.0.1:8765")
    assert tunnel.start() == "https://small-quiet-table.trycloudflare.com"
    assert popen.call_args.args[0][-1] == "http://127.0.0.1:8765"
    tunnel.close()
    process.terminate.assert_called_once()
    process.wait.assert_called_once()


def test_failed_share_reports_failure_and_stops_child(monkeypatch):
    process = Mock()
    process.stdout = StringIO("Could not connect to tunnel service\n")
    process.poll.return_value = None
    monkeypatch.setattr("scripts.share_tunnel.shutil.which", lambda _: "/bin/cloudflared")
    monkeypatch.setattr("scripts.share_tunnel.subprocess.Popen", lambda *a, **kw: process)
    with pytest.raises(RuntimeError, match="Could not connect"):
        QuickTunnel("http://127.0.0.1:8765").start()
    process.terminate.assert_called_once()


def test_missing_tunnel_binary_explains_installation(monkeypatch):
    monkeypatch.setattr("scripts.share_tunnel.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="brew install cloudflared"):
        QuickTunnel("http://127.0.0.1:8765").start()


def test_ready_waits_through_initial_dns_failure(monkeypatch):
    tunnel = QuickTunnel("http://127.0.0.1:8765")
    tunnel.url = "https://small-quiet-table.trycloudflare.com"
    tunnel.process = Mock()
    tunnel.process.poll.return_value = None
    response = Mock()
    response.status = 200
    response.read.return_value = b'{"service":"avalon-human-table","ready":true}'
    context = Mock()
    context.__enter__ = Mock(return_value=response)
    context.__exit__ = Mock(return_value=False)
    fetch = Mock(side_effect=[URLError("Could not resolve host"), context])
    monkeypatch.setattr("scripts.share_tunnel.urlopen", fetch)
    monkeypatch.setattr("scripts.share_tunnel.time.sleep", lambda _: None)
    tunnel.wait_ready(timeout=1)
    assert fetch.call_count == 2
    assert fetch.call_args.args[0].endswith("/api/health")


def test_ready_rejects_a_tunnel_that_exited(monkeypatch):
    tunnel = QuickTunnel("http://127.0.0.1:8765")
    tunnel.process = Mock()
    tunnel.process.poll.return_value = 1
    fetch = Mock()
    monkeypatch.setattr("scripts.share_tunnel.urlopen", fetch)
    with pytest.raises(RuntimeError, match="已退出"):
        tunnel.wait_ready()
    fetch.assert_not_called()
