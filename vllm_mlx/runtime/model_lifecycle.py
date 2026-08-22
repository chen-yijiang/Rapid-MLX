"""Authoritative request admission and model lifecycle operations."""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum


class LifecyclePhase(StrEnum):
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    READY = "ready"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"


class LifecycleAdmissionClosedError(RuntimeError):
    def __init__(self, operation_id: str) -> None:
        super().__init__("model lifecycle operation is draining request admission")
        self.operation_id = operation_id


class LifecycleOperationNotFoundError(KeyError):
    pass


class LifecycleOperationConflictError(RuntimeError):
    pass


@dataclass
class LifecycleOperation:
    id: str
    target_model: str | None
    reason: str
    phase: LifecyclePhase
    affected_requests: int
    created_at: float
    expires_at: float

    def payload(self) -> dict:
        return {
            "id": self.id,
            "target_model": self.target_model,
            "reason": self.reason,
            "phase": self.phase.value,
            "affected_requests": self.affected_requests,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "admission_closed": self.phase
            in {LifecyclePhase.AWAITING_CONFIRMATION, LifecyclePhase.READY},
        }


class ModelLifecycleManager:
    """Serialize drain permits with inference admission under one lock.

    The permit does not kill or replace a process. It closes admission
    atomically, reports the requests that were already active, and gives an
    external supervisor an operation id it can confirm/cancel/complete.
    """

    def __init__(
        self,
        *,
        clock=time.time,
        operation_timeout_seconds: float = 300,
        max_history: int = 256,
    ) -> None:
        self._clock = clock
        self._lock = asyncio.Lock()
        self._active_requests = 0
        self._current: LifecycleOperation | None = None
        self._history: dict[str, LifecycleOperation] = {}
        self._operation_timeout_seconds = max(1.0, operation_timeout_seconds)
        self._max_history = max(1, int(max_history))

    @asynccontextmanager
    async def admit(self):
        async with self._lock:
            self._expire_locked()
            current = self._current
            if current is not None and current.phase in {
                LifecyclePhase.AWAITING_CONFIRMATION,
                LifecyclePhase.READY,
            }:
                raise LifecycleAdmissionClosedError(current.id)
            self._active_requests += 1
        try:
            yield
        finally:
            async with self._lock:
                self._active_requests = max(0, self._active_requests - 1)

    async def begin(
        self, *, target_model: str | None, reason: str
    ) -> LifecycleOperation:
        async with self._lock:
            self._expire_locked()
            if self._current is not None and self._current.phase in {
                LifecyclePhase.AWAITING_CONFIRMATION,
                LifecyclePhase.READY,
            }:
                self._current.phase = LifecyclePhase.SUPERSEDED
            created_at = self._clock()
            operation = LifecycleOperation(
                id=str(uuid.uuid4()),
                target_model=target_model,
                reason=reason,
                phase=(
                    LifecyclePhase.AWAITING_CONFIRMATION
                    if self._active_requests > 0
                    else LifecyclePhase.READY
                ),
                affected_requests=self._active_requests,
                created_at=created_at,
                expires_at=created_at + self._operation_timeout_seconds,
            )
            self._current = operation
            self._history[operation.id] = operation
            self._trim_history_locked()
            return operation

    async def confirm(self, operation_id: str) -> LifecycleOperation:
        async with self._lock:
            self._expire_locked()
            operation = self._lookup_current(operation_id)
            if operation.phase is not LifecyclePhase.AWAITING_CONFIRMATION:
                raise LifecycleOperationConflictError(
                    f"operation cannot be confirmed from {operation.phase.value}"
                )
            operation.phase = LifecyclePhase.READY
            return operation

    async def cancel(self, operation_id: str) -> LifecycleOperation:
        async with self._lock:
            self._expire_locked()
            operation = self._lookup_current(operation_id)
            if operation.phase not in {
                LifecyclePhase.AWAITING_CONFIRMATION,
                LifecyclePhase.READY,
            }:
                raise LifecycleOperationConflictError(
                    f"operation cannot be cancelled from {operation.phase.value}"
                )
            operation.phase = LifecyclePhase.CANCELLED
            self._current = None
            return operation

    async def complete(self, operation_id: str) -> LifecycleOperation:
        async with self._lock:
            self._expire_locked()
            operation = self._lookup_current(operation_id)
            if operation.phase is not LifecyclePhase.READY:
                raise LifecycleOperationConflictError(
                    f"operation cannot be completed from {operation.phase.value}"
                )
            operation.phase = LifecyclePhase.COMPLETED
            self._current = None
            return operation

    async def status(self) -> dict:
        async with self._lock:
            self._expire_locked()
            return {
                "accepting_requests": self._current is None,
                "active_requests": self._active_requests,
                "operation": self._current.payload() if self._current else None,
            }

    def _expire_locked(self) -> None:
        if self._current is None:
            return
        if self._clock() < self._current.expires_at:
            return
        self._current.phase = LifecyclePhase.EXPIRED
        self._current = None

    def _trim_history_locked(self) -> None:
        while len(self._history) > self._max_history:
            oldest_id = next(iter(self._history))
            if self._current is not None and oldest_id == self._current.id:
                break
            self._history.pop(oldest_id, None)

    def _lookup_current(self, operation_id: str) -> LifecycleOperation:
        operation = self._history.get(operation_id)
        if operation is None:
            raise LifecycleOperationNotFoundError(operation_id)
        if operation is not self._current:
            raise LifecycleOperationConflictError(
                f"operation is no longer current ({operation.phase.value})"
            )
        return operation


def get_lifecycle_manager() -> ModelLifecycleManager:
    """Return the config-scoped manager, recreating it after test resets."""

    from ..config import get_config

    config = get_config()
    if config.lifecycle_manager is None:
        config.lifecycle_manager = ModelLifecycleManager()
    return config.lifecycle_manager
