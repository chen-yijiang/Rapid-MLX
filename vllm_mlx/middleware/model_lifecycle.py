"""ASGI inference admission tied to the model lifecycle control plane."""

from __future__ import annotations

import json

from fastapi import HTTPException

from ..config import get_config
from ..runtime.model_lifecycle import (
    LifecycleAdmissionClosedError,
    get_lifecycle_manager,
)
from .auth import _extract_bearer_token, _verify_api_key_values

_INFERENCE_PATHS = frozenset(
    {
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/embeddings",
        "/v1/images/generations",
        "/v1/images/edits",
        "/v1/audio/transcriptions",
        "/v1/audio/speech",
        "/v1/audio/translations",
        "/v1/audio/music",
        "/v1/videos",
    }
)


class ModelLifecycleMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http" or scope.get("path") not in _INFERENCE_PATHS:
            await self.app(scope, receive, send)
            return
        if not _request_is_authenticated(scope):
            # Preserve the route dependency's exact 401 envelope. Lifecycle
            # state and operation ids must not be disclosed before auth.
            await self.app(scope, receive, send)
            return
        manager = get_lifecycle_manager()
        try:
            async with manager.admit():
                await self.app(scope, receive, send)
        except LifecycleAdmissionClosedError as exc:
            body = json.dumps(
                {
                    "error": {
                        "code": "model_lifecycle_draining",
                        "message": "The model server is preparing a lifecycle change.",
                        "operation_id": exc.operation_id,
                    }
                },
                separators=(",", ":"),
            ).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 409,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                        (b"retry-after", b"1"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})


def install_model_lifecycle_middleware(app) -> None:
    app.add_middleware(ModelLifecycleMiddleware)


def _request_is_authenticated(scope) -> bool:
    if get_config().api_key is None:
        return True
    headers: dict[bytes, list[str]] = {}
    for name, value in scope.get("headers", []):
        headers.setdefault(name.lower(), []).append(value.decode("latin-1"))
    authorization = (headers.get(b"authorization") or [None])[0]
    bearer = _extract_bearer_token(authorization)
    keys: list[str | None] = [bearer]
    if scope.get("path") in {"/v1/messages", "/v1/messages/count_tokens"}:
        keys.extend(headers.get(b"x-api-key", []))
    try:
        return _verify_api_key_values(*keys)
    except HTTPException:
        return False
