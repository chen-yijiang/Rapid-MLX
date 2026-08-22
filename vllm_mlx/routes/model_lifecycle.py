"""Authenticated model lifecycle drain permits and status."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..middleware.auth import verify_api_key
from ..runtime.model_lifecycle import (
    LifecycleOperationConflictError,
    LifecycleOperationNotFoundError,
    get_lifecycle_manager,
)

router = APIRouter(dependencies=[Depends(verify_api_key)])


class LifecycleBeginRequest(BaseModel):
    target_model: str | None = Field(default=None, min_length=1)
    reason: str = Field(default="model_switch", min_length=1, max_length=64)


def _manager():
    return get_lifecycle_manager()


@router.get("/v1/models/lifecycle")
async def lifecycle_status():
    return await _manager().status()


@router.post("/v1/models/lifecycle/operations", status_code=201)
async def begin_lifecycle_operation(request: LifecycleBeginRequest):
    operation = await _manager().begin(
        target_model=request.target_model,
        reason=request.reason,
    )
    return operation.payload()


async def _transition(operation_id: str, action: str):
    try:
        operation = await getattr(_manager(), action)(operation_id)
    except LifecycleOperationNotFoundError as exc:
        raise HTTPException(404, "Lifecycle operation not found") from exc
    except LifecycleOperationConflictError as exc:
        raise HTTPException(409, str(exc)) from exc
    return operation.payload()


@router.post("/v1/models/lifecycle/operations/{operation_id}/confirm")
async def confirm_lifecycle_operation(operation_id: str):
    return await _transition(operation_id, "confirm")


@router.delete("/v1/models/lifecycle/operations/{operation_id}")
async def cancel_lifecycle_operation(operation_id: str):
    return await _transition(operation_id, "cancel")


@router.post("/v1/models/lifecycle/operations/{operation_id}/complete")
async def complete_lifecycle_operation(operation_id: str):
    return await _transition(operation_id, "complete")
