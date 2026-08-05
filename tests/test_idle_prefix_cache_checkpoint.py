"""Runtime idle-checkpoint regression tests."""

from __future__ import annotations

import asyncio
import threading

from vllm_mlx import server
from vllm_mlx.runtime import cache as runtime_cache


def test_runtime_checkpoint_limits_snapshot_to_one_candidate(monkeypatch, tmp_path):
    predicates = []

    class Engine:
        def save_cache_to_disk(self, _path, should_abort=None):
            predicates.append(should_abort)
            assert should_abort(99.0) is False
            assert should_abort(0.0) is True
            return True

    class Config:
        engine = Engine()

    monkeypatch.setattr(runtime_cache, "get_config", lambda: Config())
    monkeypatch.setattr(runtime_cache, "get_cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(runtime_cache, "_save_radix_index_after_cache", lambda *_: None)

    assert runtime_cache.checkpoint_prefix_cache_to_disk() is True
    assert len(predicates) == 1


def test_runtime_checkpoint_skips_legacy_engine_that_cannot_limit_entries(
    monkeypatch,
):
    calls = []

    class Engine:
        def save_cache_to_disk(self, _path):
            calls.append(True)
            return True

    class Config:
        engine = Engine()

    monkeypatch.setattr(runtime_cache, "get_config", lambda: Config())
    assert runtime_cache.checkpoint_prefix_cache_to_disk() is False
    assert calls == []


def test_idle_checkpoint_runs_only_for_dirty_idle_cache(monkeypatch):
    calls = []
    loop_thread = threading.get_ident()

    class Engine:
        def get_stats(self):
            return {"num_running": 0, "num_waiting": 0}

        def get_cache_stats(self):
            return {
                "content_generation": 2,
                "persisted_generation": 1,
                "entry_count": 1,
            }

    def checkpoint():
        calls.append(threading.get_ident())
        stop.set()
        return True

    async def run():
        nonlocal stop
        stop = asyncio.Event()
        monkeypatch.setattr(server, "_engine", Engine())
        monkeypatch.setattr(server, "_prefix_cache_load_task", None)
        monkeypatch.setattr(server, "_prefix_cache_snapshot_exists", lambda: True)
        monkeypatch.setattr(server, "_checkpoint_prefix_cache_to_disk", checkpoint)
        await server._idle_prefix_cache_checkpoint_loop(stop, 0.01)

    stop = None
    asyncio.run(run())
    assert len(calls) == 1
    assert calls[0] != loop_thread


def test_idle_checkpoint_skips_clean_and_busy_cache(monkeypatch):
    calls = []

    class Engine:
        def __init__(self):
            self.polls = 0

        def get_stats(self):
            self.polls += 1
            if self.polls == 1:
                return {"num_running": 1, "num_waiting": 0}
            stop.set()
            return {"num_running": 0, "num_waiting": 0}

        def get_cache_stats(self):
            return {
                "content_generation": 3,
                "persisted_generation": 3,
                "entry_count": 1,
            }

    async def run():
        nonlocal stop
        stop = asyncio.Event()
        monkeypatch.setattr(server, "_engine", Engine())
        monkeypatch.setattr(server, "_prefix_cache_load_task", None)
        monkeypatch.setattr(server, "_prefix_cache_snapshot_exists", lambda: True)
        monkeypatch.setattr(
            server,
            "_checkpoint_prefix_cache_to_disk",
            lambda: calls.append(True),
        )
        await server._idle_prefix_cache_checkpoint_loop(stop, 0.01)

    stop = None
    asyncio.run(run())
    assert calls == []


def test_idle_checkpoint_rebuilds_missing_committed_snapshot(monkeypatch):
    calls = []

    class Engine:
        def get_stats(self):
            return {"num_running": 0, "num_waiting": 0}

        def get_cache_stats(self):
            return {
                "content_generation": 1,
                "persisted_generation": 1,
                "entry_count": 1,
            }

    def checkpoint():
        calls.append(True)
        stop.set()

    async def run():
        nonlocal stop
        stop = asyncio.Event()
        monkeypatch.setattr(server, "_engine", Engine())
        monkeypatch.setattr(server, "_prefix_cache_load_task", None)
        monkeypatch.setattr(server, "_prefix_cache_snapshot_exists", lambda: False)
        monkeypatch.setattr(server, "_checkpoint_prefix_cache_to_disk", checkpoint)
        await server._idle_prefix_cache_checkpoint_loop(stop, 0.01)

    stop = None
    asyncio.run(run())
    assert calls == [True]


def test_checkpoint_interval_invalid_value_falls_back(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_PREFIX_CACHE_CHECKPOINT_SECONDS", "invalid")
    assert server._prefix_cache_checkpoint_interval() == 300.0
