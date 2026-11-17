"""Lifecycle of the optional Cloudflare Quick Tunnel used by play_web --share."""
from __future__ import annotations

import re
import json
import shutil
import subprocess
import threading
import time
from collections import deque
from urllib.request import urlopen


class QuickTunnel:
    def __init__(self, local_url):
        self.local_url = local_url
        self.process = None
        self.url = None
        self.lines = deque(maxlen=12)
        self.discovered = threading.Event()

    def start(self, timeout=45):
        executable = shutil.which("cloudflared")
        if not executable:
            raise RuntimeError("未找到 cloudflared。macOS 可先运行 brew install cloudflared，再使用 --share。")
        self.process = subprocess.Popen(
            [executable, "tunnel", "--no-autoupdate", "--url", self.local_url],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", start_new_session=True,
        )

        def read_output():
            for line in self.process.stdout:
                self.lines.append(line.strip())
                match = re.search(r"https://[a-z0-9]+(?:-[a-z0-9]+)*\.trycloudflare\.com\b", line)
                if match:
                    self.url = match.group(0)
                    self.discovered.set()
            self.discovered.set()  # Also wake the caller if the process failed early.

        threading.Thread(target=read_output, daemon=True).start()
        self.discovered.wait(timeout)
        if not self.url:
            self.close()
            detail = "\n".join(self.lines)
            raise RuntimeError("无法创建公网隧道，请检查网络后重试。\n" + detail)
        return self.url

    def wait_ready(self, timeout=60):
        """A hostname being assigned does not mean DNS and the public route are ready."""
        deadline = time.monotonic() + timeout
        last_error = "等待公网入口响应"
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("公网隧道已退出：\n" + "\n".join(self.lines))
            try:
                remaining = max(0.1, deadline - time.monotonic())
                with urlopen(self.url + "/api/health", timeout=min(5, remaining)) as response:
                    health = json.loads(response.read(4096))
                    if response.status == 200 and health == {"service": "avalon-human-table", "ready": True}:
                        return
                    last_error = "入口尚未返回牌桌的就绪响应"
            except (OSError, ValueError) as exc:
                last_error = str(exc)
            time.sleep(min(1, max(0, deadline - time.monotonic())))
        raise RuntimeError(f"公网地址已分配，但尚未验证可用：{last_error}。请稍后重试。")

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)

