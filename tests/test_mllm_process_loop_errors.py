# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for scheduler-level MLLM step failures (#1367)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from vllm_mlx.mllm_scheduler import MLLMRequest, MLLMScheduler


@pytest.mark.asyncio
async def test_process_loop_failure_unblocks_every_inflight_request() -> None:
    """Unexpected mlx-vlm/model errors must not be logged and retried forever."""
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    request = MLLMRequest(request_id="broken", prompt="hello")
    scheduler.requests = {request.request_id: request}
    scheduler.waiting = __import__("collections").deque([request])
    scheduler.running = {}
    scheduler.output_queues = {request.request_id: asyncio.Queue()}
    scheduler.request_id_to_uid = {}
    scheduler.uid_to_request_id = {}
    scheduler._detokenizer_pool = {}
    scheduler._pending_abort_ids = set()
    scheduler._aborted_queue_ids = set()
    scheduler.finished_req_ids = set()
    scheduler._running = True
    scheduler._injected_step_executor = None
    scheduler._step_executor = None
    scheduler._owns_step_executor = True
    scheduler.batch_generator = None
    scheduler._step_no_queue = MagicMock(
        side_effect=TypeError("Model.__call__() missing required argument: mask")
    )

    task = asyncio.create_task(scheduler._process_loop())
    try:
        output = await asyncio.wait_for(
            scheduler.output_queues[request.request_id].get(), timeout=0.5
        )
    finally:
        scheduler._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert output.finished is True
    assert output.finish_reason == "length"
    assert "TypeError" in output.error
    assert "mask" in output.error
    assert scheduler._step_no_queue.call_count == 1
    assert not scheduler.requests
    assert not scheduler.waiting
