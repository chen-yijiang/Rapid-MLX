from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from vllm_mlx.engine.batched import BatchedEngine, _admission_token_context
from vllm_mlx.engine_core import EngineCore
from vllm_mlx.mllm_scheduler import MLLMScheduler
from vllm_mlx.output_collector import RequestOutputCollector
from vllm_mlx.scheduler import BackpressureError, Scheduler


def _engine(*, reservations: int = 0, running: dict | None = None):
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._is_mllm = False
    engine._mllm_scheduler = None
    engine._admission_lock = threading.Lock()
    engine._admission_reservations = reservations
    engine._admission_tokens = {index + 1 for index in range(reservations)}
    if reservations:
        _admission_token_context.set((id(engine), (1,)))
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


@pytest.mark.asyncio
async def test_resume_reopens_scheduler_before_outer_engine_gate():
    engine, scheduler = _engine()
    engine._generation_paused = True

    def set_generation_paused(paused, *, add_allowance=0):
        assert engine._generation_paused is True
        scheduler.generation_paused = paused

    scheduler.set_generation_paused = set_generation_paused

    await engine.resume_generation()

    assert scheduler.generation_paused is False
    assert engine._generation_paused is False


@pytest.mark.asyncio
async def test_pause_timeout_automatically_reopens_both_gates():
    running = {"busy": SimpleNamespace(request_id="busy")}
    engine, scheduler = _engine(running=running)

    with pytest.raises(TimeoutError):
        await engine.pause_generation("wait", timeout=0)

    assert engine._generation_paused is False
    assert scheduler.generation_paused is False


@pytest.mark.asyncio
async def test_concurrent_pause_transactions_are_serialized():
    engine, _ = _engine(reservations=1)

    first = asyncio.create_task(engine.pause_generation("wait"))
    await asyncio.sleep(0)
    second = asyncio.create_task(engine.pause_generation("wait"))
    await asyncio.sleep(0)
    assert not second.done()

    engine.release_admission_reservation()
    await asyncio.wait_for(first, timeout=1)
    await asyncio.wait_for(second, timeout=1)
    await engine.resume_generation()


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


@pytest.mark.parametrize("scheduler_type", [Scheduler, MLLMScheduler])
def test_scheduler_pause_atomically_accounts_for_already_owned_requests(
    scheduler_type,
):
    scheduler = scheduler_type.__new__(scheduler_type)
    scheduler._cancel_counter_lock = threading.Lock()
    scheduler.requests = {"already-owned": SimpleNamespace(lifecycle_admission_token=1)}

    scheduler.pause_generation_admission({1, 2}, "wait")

    assert scheduler._generation_paused is True
    assert scheduler._paused_add_allowance == 1
    assert scheduler._paused_admission_tokens == {2}

    scheduler.set_generation_paused(False)
    assert scheduler._generation_paused is False
    assert scheduler._paused_add_allowance == 0


def test_paused_engine_rejects_even_when_concurrency_cap_is_unlimited():
    engine, scheduler = _engine()
    scheduler.config.max_concurrent_requests = None
    engine._generation_paused = True

    with pytest.raises(BackpressureError, match="paused"):
        engine.check_admission()


def test_unlimited_cap_still_tracks_lifecycle_reservation():
    engine, _ = _engine()
    engine._engine.engine.scheduler.config.max_concurrent_requests = None

    engine.check_admission()

    assert engine._admission_reservations == 1
    engine.release_admission_reservation()
    assert engine._admission_reservations == 0


def test_same_context_admissions_release_as_a_token_stack():
    engine, _ = _engine()
    engine._engine.engine.scheduler.config.max_concurrent_requests = None

    engine.check_admission()
    engine.check_admission()
    assert engine._admission_reservations == 2

    engine.release_admission_reservation()
    assert engine._admission_reservations == 1
    engine.release_admission_reservation()
    assert engine._admission_reservations == 0


def test_scheduler_transfer_releases_route_owned_reservation():
    engine, _ = _engine()
    engine.check_admission()
    context = _admission_token_context.get()
    assert context is not None

    engine._transfer_admission_to_scheduler(context[1][-1])

    assert engine._admission_reservations == 0
    assert engine._admission_tokens == set()


def test_lifecycle_active_requests_is_authoritative_total():
    running = {"one": object(), "two": object()}
    engine, scheduler = _engine(reservations=1, running=running)
    scheduler.waiting.append(object())

    status = engine.lifecycle_status()

    assert status["admitted_requests"] == 1
    assert status["running_requests"] == 2
    assert status["queued_requests"] == 1
    assert status["active_requests"] == 4


@pytest.mark.parametrize("scheduler_type", [Scheduler, MLLMScheduler])
def test_request_id_snapshot_is_safe_during_concurrent_mutation(scheduler_type):
    scheduler = scheduler_type.__new__(scheduler_type)
    scheduler._cancel_counter_lock = threading.Lock()
    scheduler.requests = {}
    result = []
    scheduler._cancel_counter_lock.acquire()
    try:
        reader = threading.Thread(
            target=lambda: result.append(scheduler.request_ids_snapshot())
        )
        reader.start()
        # Snapshot must be blocked on the exact lock used for request commits.
        reader.join(timeout=0.01)
        assert reader.is_alive()
        scheduler.requests.update({"one": 1, "two": 2})
    finally:
        scheduler._cancel_counter_lock.release()

    reader.join(timeout=1)
    assert result == [("one", "two")]


@pytest.mark.asyncio
async def test_text_abort_wakes_non_streaming_consumer_with_terminal_error():
    engine = EngineCore.__new__(EngineCore)
    engine.scheduler = SimpleNamespace(abort_request=lambda _request_id: True)
    engine._output_collectors = {
        "active": RequestOutputCollector(aggregate=True),
    }
    engine._finished_events = {"active": asyncio.Event()}
    engine._idle_event = asyncio.Event()

    assert await engine.abort_request("active") is True
    await asyncio.wait_for(engine._finished_events["active"].wait(), timeout=0.1)

    terminal = engine._output_collectors["active"].get_nowait()
    assert terminal is not None
    assert terminal.finished is True
    assert terminal.error_kind == "lifecycle"
    assert "cancellation" in terminal.error
    # The waiting stream/generate coroutine owns cleanup after consuming the
    # terminal signal. Removing these here recreates the hung HTTP request.
    assert "active" in engine._output_collectors
    assert "active" in engine._finished_events


def test_mllm_abort_delivers_terminal_error_instead_of_empty_success():
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.output_queues = {"active": asyncio.Queue()}
    scheduler._aborted_queue_ids = {"active"}

    scheduler._distribute_outputs(SimpleNamespace(outputs=[]))

    terminal = scheduler.output_queues["active"].get_nowait()
    assert terminal.finished is True
    assert terminal.error_kind == "lifecycle"
    assert "cancellation" in terminal.error


@pytest.mark.asyncio
async def test_mllm_abort_unblocks_consumer_as_inference_error():
    from vllm_mlx.request import InferenceAbortedError, RequestOutput

    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.output_queues = {"active": asyncio.Queue()}
    scheduler.output_queues["active"].put_nowait(
        RequestOutput(
            request_id="active",
            finished=True,
            finish_reason="length",
            error="Inference aborted by a cancellation request",
            error_kind="lifecycle",
        )
    )

    with pytest.raises(InferenceAbortedError, match="cancellation"):
        await anext(scheduler.stream_outputs("active"))
    assert "active" not in scheduler.output_queues
