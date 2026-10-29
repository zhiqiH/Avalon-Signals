"""One training process on the GPU at a time.

Several runs share the GPU of the Windows laptop. When the PPO updates of six runs overlapped there, the
driver (RTX 5070 Ti Laptop, driver 581.42) hit a firmware fault ("Graphics FECS Exception: UCODE Fatal
Error", then a GPU reset) that killed every run; the same six runs taking turns through this lock trained
without faults. A run alone on the machine never waits.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch

LOCK_PATH = Path(__file__).resolve().parents[3] / ".gpu.lock"  # shared by every run of this checkout


@contextmanager
def file_lock(path: Path, poll: float = 0.02) -> Iterator[None]:
    """Machine-wide exclusive lock on `path`; the OS releases it if the holder dies."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    try:
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(poll)
            try:
                yield
            finally:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@contextmanager
def gpu_turn(device: torch.device, path: Path = LOCK_PATH) -> Iterator[None]:
    """Hold the GPU for the enclosed work (no-op off CUDA); its queued kernels finish before the lock is released."""
    if device.type != "cuda":
        yield
        return
    with file_lock(path):
        try:
            yield
        finally:
            torch.cuda.synchronize(device)
