from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.config import reset_config
from vllm_mlx.middleware.model_lifecycle import ModelLifecycleMiddleware
from vllm_mlx.runtime.model_lifecycle import (
    LifecycleAdmissionClosedError,
    LifecycleOperationConflictError,
    LifecycleOperationNotFoundError,
    LifecyclePhase,
    ModelLifecycleManager,
)


def test_lifecycle_routes_expose_status_and_preserve_auth_contract():
    from vllm_mlx.routes.model_lifecycle import router

    config = reset_config()
    config.api_key = "secret"
    config.lifecycle_manager = ModelLifecycleManager(clock=lambda: 42.0)
    app = FastAPI()
    app.include_router(router)

    with TestClient(app) as client:
        assert client.get("/v1/models/lifecycle").status_code == 401
        headers = {"Authorization": "Bearer secret"}
        initial = client.get("/v1/models/lifecycle", headers=headers)
        assert initial.status_code == 200
        assert initial.json() == {
            "accepting_requests": True,
            "active_requests": 0,
            "operation": None,
        }
        created = client.post(
            "/v1/models/lifecycle/operations",
            headers=headers,
            json={"target_model": "new", "reason": "model_switch"},
        )
        assert created.status_code == 201
        assert created.json()["phase"] == "ready"
        operation_id = created.json()["id"]
        completed = client.post(
            f"/v1/models/lifecycle/operations/{operation_id}/complete",
            headers=headers,
        )
        assert completed.status_code == 200
        assert completed.json()["phase"] == "completed"


@pytest.mark.asyncio
async def test_begin_atomically_closes_admission_and_reports_active_requests():
    manager = ModelLifecycleManager(clock=lambda: 123.0)
    release = asyncio.Event()

    async def active_request():
        async with manager.admit():
            await release.wait()

    request = asyncio.create_task(active_request())
    await asyncio.sleep(0)
    operation = await manager.begin(target_model="new", reason="model_switch")

    assert operation.phase is LifecyclePhase.AWAITING_CONFIRMATION
    assert operation.affected_requests == 1
    with pytest.raises(LifecycleAdmissionClosedError) as closed:
        async with manager.admit():
            pass
    assert closed.value.operation_id == operation.id

    confirmed = await manager.confirm(operation.id)
    assert confirmed.phase is LifecyclePhase.READY
    release.set()
    await request
    assert (await manager.status())["active_requests"] == 0
    completed = await manager.complete(operation.id)
    assert completed.phase is LifecyclePhase.COMPLETED
    assert (await manager.status())["accepting_requests"] is True


@pytest.mark.asyncio
async def test_cancel_reopens_admission_and_new_operation_supersedes_old_token():
    manager = ModelLifecycleManager()
    first = await manager.begin(target_model="a", reason="model_switch")
    second = await manager.begin(target_model="b", reason="model_switch")

    assert first.phase is LifecyclePhase.SUPERSEDED
    with pytest.raises(LifecycleOperationConflictError):
        await manager.confirm(first.id)
    await manager.cancel(second.id)
    async with manager.admit():
        assert (await manager.status())["active_requests"] == 1


@pytest.mark.asyncio
async def test_abandoned_operation_expires_and_reopens_admission():
    now = 10.0
    manager = ModelLifecycleManager(
        clock=lambda: now,
        operation_timeout_seconds=5,
    )
    operation = await manager.begin(target_model="new", reason="model_switch")
    now = 16.0

    assert (await manager.status())["accepting_requests"] is True
    assert operation.phase is LifecyclePhase.EXPIRED
    async with manager.admit():
        pass


@pytest.mark.asyncio
async def test_operation_history_is_bounded_without_evicting_current():
    manager = ModelLifecycleManager(max_history=3)
    operations = []
    for index in range(10):
        operations.append(
            await manager.begin(target_model=str(index), reason="model_switch")
        )

    assert len(manager._history) == 3
    assert operations[-1].id in manager._history
    with pytest.raises(LifecycleOperationNotFoundError):
        await manager.confirm(operations[0].id)


@pytest.mark.asyncio
async def test_asgi_middleware_counts_until_stream_body_finishes():
    config = reset_config()
    manager = ModelLifecycleManager()
    config.lifecycle_manager = manager
    body_release = asyncio.Event()
    response_started = asyncio.Event()
    sent = []

    async def send(message):
        sent.append(message)

    async def streaming_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"one", "more_body": True})
        response_started.set()
        await body_release.wait()
        await send({"type": "http.response.body", "body": b"two"})

    middleware = ModelLifecycleMiddleware(streaming_app)
    scope = {"type": "http", "path": "/v1/chat/completions", "method": "POST"}

    task = asyncio.create_task(middleware(scope, None, send))
    await response_started.wait()
    assert (await manager.status())["active_requests"] == 1
    operation = await manager.begin(target_model="new", reason="model_switch")
    assert operation.affected_requests == 1
    body_release.set()
    await task
    assert (await manager.status())["active_requests"] == 0


@pytest.mark.asyncio
async def test_asgi_middleware_rejects_new_inference_but_not_control_plane():
    config = reset_config()
    manager = ModelLifecycleManager()
    config.lifecycle_manager = manager
    operation = await manager.begin(target_model="new", reason="model_switch")

    calls = []

    async def app(scope, receive, send):
        calls.append(scope["path"])

    middleware = ModelLifecycleMiddleware(app)
    sent = []

    async def send(message):
        sent.append(message)

    await middleware(
        {"type": "http", "path": "/v1/chat/completions", "method": "POST"},
        None,
        send,
    )
    assert calls == []
    assert sent[0]["status"] == 409
    payload = json.loads(sent[1]["body"])
    assert payload["error"]["operation_id"] == operation.id

    await middleware(
        {"type": "http", "path": "/v1/models/lifecycle", "method": "GET"},
        None,
        send,
    )
    assert calls == ["/v1/models/lifecycle"]


@pytest.mark.asyncio
async def test_lifecycle_middleware_preserves_auth_before_drain_rejection():
    config = reset_config()
    config.api_key = "secret"
    manager = ModelLifecycleManager()
    config.lifecycle_manager = manager
    await manager.begin(target_model="new", reason="model_switch")
    calls = []

    async def app(scope, receive, send):
        calls.append(scope)

    middleware = ModelLifecycleMiddleware(app)

    await middleware(
        {"type": "http", "path": "/v1/chat/completions", "headers": []},
        None,
        None,
    )
    assert len(calls) == 1

    sent = []

    async def send(message):
        sent.append(message)

    await middleware(
        {
            "type": "http",
            "path": "/v1/chat/completions",
            "headers": [(b"authorization", b"Bearer secret")],
        },
        None,
        send,
    )
    assert sent[0]["status"] == 409
