# SPDX-License-Identifier: Apache-2.0
"""Image generation endpoints (OpenAI ``/v1/images/*`` compatible)."""

import base64
import logging
import os
import tempfile
import time

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile

from ..api.models import ImageGenerationRequest, parse_image_size
from ._async_utils import run_to_completion

logger = logging.getLogger(__name__)

router = APIRouter()

# Cap the uploaded init image so a single edit request can't buffer an
# unbounded body into memory before the size validators run.
_MAX_EDIT_IMAGE_BYTES = 25 * 1024 * 1024

# Default denoise steps for an instruction edit. Qwen-Image-Edit is a large,
# non-distilled model (unlike the 4-step FLUX.1-schnell generator): its edit
# structure needs ~20 steps to resolve, and — because output quality on the
# 4-bit checkpoints is bounded by the quantized VAE rather than by step count —
# pushing past this only costs wall-clock (≈1 min/step at the derived 1024²
# canvas) without a visible gain. Callers can override via ``steps``.
_DEFAULT_EDIT_STEPS = 20


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
    # Step count is family-aware: a distilled model (Klein/schnell, 4 steps)
    # would waste wall-clock at 20 and a non-distilled one (Qwen, 20) would be
    # noise at 4. The engine advertises the right default per family.
    default_steps = getattr(engine, "default_steps", 4)
    return engine.generate(
        prompt=request.prompt,
        width=width,
        height=height,
        num_inference_steps=request.steps if request.steps is not None else default_steps,
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

    from ..image.engine import ImageGenerationCancelled

    data = []
    cancelled = False
    for index in range(request.n):
        try:
            png_bytes = await run_to_completion(
                _generate_one, engine, request, base_seed + index
            )
        except ImageGenerationCancelled:
            # User stopped mid-render: return whatever finished rather than an
            # error, so a cancelled multi-image batch keeps its earlier images.
            cancelled = True
            break
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

    return {"created": int(time.time()), "data": data, "cancelled": cancelled}


@router.get("/v1/images/progress")
async def image_progress():
    """Live denoise progress for the single in-flight render.

    Diffusion has a fixed step count, so this is a *true* ``step / total``
    signal the client polls to drive a determinate progress bar and ETA — no
    streaming parser, and honest on slow hardware (the bar can't outrun the
    real steps). Single-flight: the server renders one image at a time.
    """
    engine = _image_engine()
    snap = engine.progress_snapshot() if hasattr(engine, "progress_snapshot") else {}
    return snap


@router.post("/v1/images/cancel")
async def image_cancel():
    """Ask the in-flight render to stop at the next denoise step."""
    engine = _image_engine()
    if hasattr(engine, "request_cancel"):
        engine.request_cancel()
    return {"ok": True}


def _generate_edit_one(engine, prompt, steps, seed, guidance,
                       negative_prompt, image_path) -> bytes:
    """Blocking single instruction-edit render — runs off the event loop.

    No width/height is threaded: the edit family sizes its output canvas from
    the input image (the engine passes ``None`` to mflux). Forcing a mismatched
    size desyncs the conditioning latents and yields pure noise, so the request
    ``size`` is accepted for OpenAI-API shape but deliberately not honored.
    """
    return engine.generate(
        prompt=prompt,
        num_inference_steps=steps if steps is not None else _DEFAULT_EDIT_STEPS,
        seed=seed,
        guidance=guidance if guidance is not None else 4.0,
        negative_prompt=negative_prompt,
        image_paths=[image_path],
    )


@router.post("/v1/images/edits")
async def edit_image(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    model: str = Form(""),
    n: int = Form(1),
    size: str = Form("1024x1024"),
    response_format: str = Form("b64_json"),
    seed: int | None = Form(None),
    steps: int | None = Form(None),
    guidance: float | None = Form(None),
    negative_prompt: str | None = Form(None),
):
    """Instruction-edit an input image (OpenAI ``/v1/images/edits`` compatible).

    Requires a server running an image-**edit** model (e.g.
    ``rapid-mlx serve qwen-image-edit-4bit``); the uploaded image plus the
    prompt drive a global instruction edit (no mask). Returns the same
    ``{created, data:[{b64_json}]}`` envelope as generations.
    """
    from ..image.engine import ImageRuntimeError

    engine = _image_engine()

    # /v1/images/edits requires the edit family; a txt2img server points the
    # caller at /v1/images/generations instead of silently ignoring the image.
    if not getattr(engine, "is_edit", False):
        raise HTTPException(
            status_code=409,
            detail={
                "error": {
                    "message": (
                        "This server is running a text-to-image model; use "
                        "/v1/images/generations, or start an image-edit model "
                        "(e.g. `rapid-mlx serve qwen-image-edit-4bit`)."
                    ),
                    "type": "invalid_request_error",
                    "code": "wrong_image_endpoint",
                    "param": "model",
                }
            },
        )

    if not prompt or not prompt.strip():
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "prompt must not be empty",
                              "type": "invalid_request_error", "param": "prompt"}},
        )
    if response_format == "url":
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "The local image lane only returns base64 "
                              "data; request response_format='b64_json'.",
                              "type": "invalid_request_error",
                              "code": "unsupported_response_format",
                              "param": "response_format"}},
        )
    if not 1 <= n <= 4:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "n must be between 1 and 4",
                              "type": "invalid_request_error", "param": "n"}},
        )
    # ``size`` is accepted for OpenAI-API compatibility but the edit family
    # derives its output canvas from the input image; a mismatched target size
    # desyncs mflux's conditioning latents into pure noise. We still validate
    # the value so a malformed ``size`` fails loud rather than being silently
    # dropped, then discard it — the engine sizes the render from the image.
    try:
        parse_image_size(size)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": str(exc),
                              "type": "invalid_request_error", "param": "size"}},
        ) from exc

    raw = await image.read()
    if not raw:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "image file is empty",
                              "type": "invalid_request_error", "param": "image"}},
        )
    if len(raw) > _MAX_EDIT_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"error": {"message": f"image exceeds "
                              f"{_MAX_EDIT_IMAGE_BYTES // (1024 * 1024)} MB limit",
                              "type": "invalid_request_error", "param": "image"}},
        )

    suffix = os.path.splitext(image.filename or "")[1] or ".png"
    base_seed = seed if seed is not None else int(time.time()) & 0x7FFFFFFF
    data = []
    # One temp file for the whole request; the process lock in the engine keeps
    # generations serial, so a shared init image is safe across the n renders.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        for index in range(n):
            try:
                png_bytes = await run_to_completion(
                    _generate_edit_one, engine, prompt, steps,
                    base_seed + index, guidance, negative_prompt, tmp_path,
                )
            except ImageRuntimeError as exc:
                raise HTTPException(
                    status_code=500,
                    detail={"error": {"message": str(exc),
                                      "type": "image_generation_error",
                                      "code": "image_generation_failed"}},
                ) from exc
            data.append({"b64_json": base64.b64encode(png_bytes).decode("ascii")})
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return {"created": int(time.time()), "data": data}
