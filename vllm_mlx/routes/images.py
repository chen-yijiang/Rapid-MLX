# SPDX-License-Identifier: Apache-2.0
"""Image generation endpoints (OpenAI ``/v1/images/*`` compatible)."""

import base64
import logging
import time

from fastapi import APIRouter, Body, HTTPException

from ..api.models import ImageGenerationRequest
from ._async_utils import run_to_completion

logger = logging.getLogger(__name__)

router = APIRouter()


def _image_engine():
    """Return the loaded image engine or raise a 409 if this isn't an image server."""
    from ..config import get_config

    engine = get_config().engine
    if engine is None or not getattr(engine, "is_image_gen", False):
        raise HTTPException(
            status_code=409,
            detail={
                "error": {
                    "message": (
                        "This server is not running an image model. Start it with "
                        "`rapid-mlx serve flux-schnell` (or another image alias)."
                    ),
                    "type": "invalid_request_error",
                    "code": "image_model_not_loaded",
                    "param": "model",
                }
            },
        )
    return engine


def _generate_one(engine, request: ImageGenerationRequest, seed: int) -> bytes:
    """Blocking single-image render — runs off the event loop."""
    width, height = request.dimensions()
    return engine.generate(
        prompt=request.prompt,
        width=width,
        height=height,
        num_inference_steps=request.steps if request.steps is not None else 4,
        seed=seed,
        guidance=request.guidance if request.guidance is not None else 4.0,
        negative_prompt=request.negative_prompt,
    )


@router.post("/v1/images/generations")
async def create_image(request: ImageGenerationRequest = Body(...)):
    """Generate one or more images from a text prompt.

    Returns the OpenAI ``{created, data:[{b64_json}]}`` envelope. ``url``
    responses are not offered by the local lane (there is no object store to
    host the bytes) — callers must request ``b64_json``.
    """
    from ..image.engine import ImageRuntimeError

    engine = _image_engine()

    # A single server hosts exactly one image model. When that model is the
    # instruction-edit family, text-to-image generation is the wrong endpoint —
    # 409 toward /v1/images/edits instead of silently ignoring the mismatch.
    if getattr(engine, "is_edit", False):
        raise HTTPException(
            status_code=409,
            detail={
                "error": {
                    "message": (
                        "This server is running an image-edit model; use "
                        "/v1/images/edits with an input image."
                    ),
                    "type": "invalid_request_error",
                    "code": "wrong_image_endpoint",
                    "param": "model",
                }
            },
        )

    if request.response_format == "url":
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (
                        "The local image lane only returns base64 data; request "
                        "response_format='b64_json'."
                    ),
                    "type": "invalid_request_error",
                    "code": "unsupported_response_format",
                    "param": "response_format",
                }
            },
        )

    # When no seed is pinned, derive one from wall-clock so successive calls
    # vary; multi-image (``n``) requests offset per index off that base.
    base_seed = request.seed if request.seed is not None else int(time.time()) & 0x7FFFFFFF

    data = []
    for index in range(request.n):
        try:
            png_bytes = await run_to_completion(
                _generate_one, engine, request, base_seed + index
            )
        except ImageRuntimeError as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": str(exc),
                        "type": "image_generation_error",
                        "code": "image_generation_failed",
                    }
                },
            ) from exc
        data.append({"b64_json": base64.b64encode(png_bytes).decode("ascii")})

    return {"created": int(time.time()), "data": data}
