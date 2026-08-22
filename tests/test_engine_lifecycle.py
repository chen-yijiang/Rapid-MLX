from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from vllm_mlx.engine.batched import BatchedEngine
from vllm_mlx.mllm_scheduler import MLLMScheduler
from vllm_mlx.scheduler import BackpressureError, Scheduler


def _engine(*, reservations: int = 0, running: dict | None = None):
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._is_mllm = False
    engine._mllm_scheduler = None
    engine._admission_lock = threading.Lock()
    engine._admission_reservations = reservations
    engine._generation_paused = False
    engine._generation_pause_mode = None
    scheduler = SimpleNamespace(
        requests=running or {},
        running=running or {},
        waiting=[],
        config=SimpleNamespace(max_concurrent_requests=8),
    )

    def set_generation_paused(paused, *, add_allowance=0):
        scheduler.generation_paused = paused
        scheduler.add_allowance = add_allowance if paused else 0

    scheduler.set_generation_paused = set_generation_paused
    engine._engine = SimpleNamespace(engine=SimpleNamespace(scheduler=scheduler))
    engine.get_stats = lambda: {
        "num_running": len(scheduler.running),
        "num_waiting": len(scheduler.waiting),
    }
    return engine, scheduler


@pytest.mark.asyncio
async def test_wait_pause_closes_admission_then_drains_existing_request():
    engine, _ = _engine(reservations=1)

    pause = asyncio.create_task(engine.pause_generation("wait"))
    await asyncio.sleep(0)

    with pytest.raises(BackpressureError, match="paused"):
        engine.check_admission()
    assert not pause.done()

    engine.release_admission_reservation()
    status = await asyncio.wait_for(pause, timeout=1)
    assert status["paused"] is True
    assert status["active_requests"] == 0

    await engine.resume_generation()
    engine.check_admission()
    engine.release_admission_reservation()


@pytest.mark.asyncio
async def test_abort_pause_rechecks_requests_that_arrive_after_pause_edge():
    engine, scheduler = _engine(reservations=1)
    aborted = []

    async def abort_request(request_id):
        aborted.append(request_id)
        scheduler.requests.pop(request_id, None)
        scheduler.running.pop(request_id, None)
        engine.release_admission_reservation()
        return True

    engine.abort_request = abort_request
    pause = asyncio.create_task(engine.pause_generation("abort"))
    await asyncio.sleep(0)

    # Simulate a route that reserved just before pause and reached the
    # scheduler just after it. Abort mode must discover it on a later scan.
    request = SimpleNamespace(request_id="late")
    scheduler.requests["late"] = request
    scheduler.running["late"] = request

    status = await asyncio.wait_for(pause, timeout=1)
    assert aborted == ["late"]
    assert status["running_requests"] == 0
    assert status["active_requests"] == 0


@pytest.mark.asyncio
async def test_wait_pause_allows_request_reserved_before_pause_to_enter_scheduler():
    engine, scheduler = _engine(reservations=1)

    pause = asyncio.create_task(engine.pause_generation("wait"))
    await asyncio.sleep(0)

    assert scheduler.generation_paused is True
    assert scheduler.add_allowance == 1

    # This request owns the one reservation captured at the pause edge.
    scheduler.add_allowance -= 1
    request = SimpleNamespace(request_id="reserved-before-pause")
    scheduler.requests[request.request_id] = request
    scheduler.running[request.request_id] = request
    await asyncio.sleep(0)
    assert not pause.done()

    scheduler.requests.clear()
    scheduler.running.clear()
    engine.release_admission_reservation()
    await asyncio.wait_for(pause, timeout=1)


@pytest.mark.asyncio
async def test_zero_timeout_atomically_pauses_an_idle_engine():
    engine, _ = _engine()

    status = await engine.pause_generation("wait", timeout=0)

    assert status["paused"] is True
    with pytest.raises(BackpressureError, match="paused"):
        engine.check_admission()


def test_text_scheduler_rejects_direct_add_while_paused():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._generation_paused = True

    with pytest.raises(BackpressureError, match="paused"):
        scheduler.add_request(SimpleNamespace(request_id="direct"))


def test_mllm_scheduler_rejects_direct_add_while_paused():
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler._generation_paused = True

    with pytest.raises(BackpressureError, match="paused"):
        scheduler.add_request("prompt", request_id="direct")


def test_paused_engine_rejects_even_when_concurrency_cap_is_unlimited():
    engine, scheduler = _engine()
    scheduler.config.max_concurrent_requests = None
    engine._generation_paused = True

    with pytest.raises(BackpressureError, match="paused"):
        engine.check_admission()
