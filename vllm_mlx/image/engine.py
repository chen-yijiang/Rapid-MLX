# SPDX-License-Identifier: Apache-2.0
"""mflux-backed image generation engine.

Thin wrapper over `mflux <https://github.com/filipstrand/mflux>`_ — an
MLX-native, line-by-line port of the FLUX / Qwen-Image model families with
built-in 4/8-bit quantization. Rapid-MLX owns request validation, the lazy
load / process-lock lifecycle and the OpenAI-compatible transport; mflux owns
the diffusion pipeline and weight loading.

Only Apache-2.0-licensed families are wired here so the whole surface stays
commercially clean:

* ``flux2-klein``      — text→image (``FLUX.2-klein-4B``), 4B/4-step, the fast
  default: ~3 s @ 512² / ~10 s @ 1024² on an M3 Ultra, ~4 GB at 4-bit
* ``z-image``          — text→image (``Z-Image-Turbo``), 6B/8-step, the quality
  option (SOTA open-source photorealism), ~5.5 GB at 4-bit
* ``flux-schnell``     — text→image (``black-forest-labs/FLUX.1-schnell``), 12B
* ``qwen-image``       — text→image (``Qwen/Qwen-Image``), strongest text-in-image
* ``qwen-image-edit``  — instruction edit (``Qwen/Qwen-Image-Edit-2509``)

``flux2-klein`` and ``z-image`` supersede the older/larger families for the
interactive tab: Klein is ~3× faster than schnell/z-image at the same
resolution while staying smaller, and Qwen-Image (20B) is too slow to feature.

The mflux model is loaded lazily on the first ``generate`` call: the canonical
repos ship full-precision weights that mflux quantizes at load, so pulling them
at server boot would stall startup on a multi-gigabyte download.
"""

from __future__ import annotations

import io
import re
import threading
from pathlib import Path

# A pre-quantized mflux repo carries a quant tag in its id — either the
# ``<n>bit`` / ``<n>-bit`` convention (``FLUX.1-schnell-mflux-4bit``) or the
# ``q<n>`` convention (``Qwen-Image-Edit-mflux-q4``). Anchored to a separator
# so a base repo like ``Qwen/Qwen-Image`` (no tag) is never misread — the
# leading ``q`` of "Qwen" is not followed by a quant digit.
_QUANT_TAG_RE = re.compile(r"(?:^|[-_./])(?:q[2-8]|[2-8]-?bit)(?:[-_./]|$)", re.IGNORECASE)

# mflux/Metal graphs are not re-entrant — a single process-wide lock serializes
# every generation exactly like the video lane's ``_PROCESS_GENERATION_LOCK``.
_PROCESS_GENERATION_LOCK = threading.Lock()

# Default quantization for the on-load quantize path. 4-bit is the 32GB sweet
# spot (FLUX.1-schnell ~9GB, Qwen-Image ~12GB resident at q4).
_DEFAULT_QUANTIZE = 4


class ImageRuntimeError(RuntimeError):
    """Safe, actionable generation error suitable for the public API."""


def _detect_family(model_name: str) -> str:
    """Map an alias hf_path (or local dir) to a supported mflux family."""
    name = (model_name or "").casefold()
    # Klein first — its repos ("FLUX.2-klein-4B-mflux-4bit") also contain
    # "flux", so the distinctive "klein" / "flux2" token must win before the
    # generic FLUX.1 checks below.
    if "klein" in name or "flux2" in name or "flux.2" in name:
        return "flux2-klein"
    if "z-image" in name or "z_image" in name or "zimage" in name:
        return "z-image"
    if "qwen-image-edit" in name or "qwen_image_edit" in name:
        return "qwen-image-edit"
    if "qwen-image" in name or "qwen_image" in name:
        return "qwen-image"
    if "schnell" in name:
        return "flux-schnell"
    if "flux.1-dev" in name or "flux1-dev" in name:
        return "flux-dev"
    raise ImageRuntimeError(
        f"Unsupported image model '{model_name}'. Supported families: "
        "flux2-klein, z-image, flux-schnell, qwen-image, qwen-image-edit."
    )


# Per-family default denoise steps when the request pins none. Distilled/turbo
# models converge in a handful of steps; a non-distilled model needs many more.
_DEFAULT_STEPS_BY_FAMILY = {
    "flux2-klein": 4,   # distilled turbo
    "z-image": 8,       # turbo, but 8 is the sweet spot for its quality
    "flux-schnell": 4,  # distilled
    "flux-dev": 20,     # non-distilled
    "qwen-image": 20,   # non-distilled 20B
    "qwen-image-edit": 20,
}


def _looks_like_prequantized(model_name: str) -> bool:
    """A pre-quantized mflux repo / local dir is loaded via ``model_path``.

    The canonical BFL / Qwen repos ship full weights that mflux quantizes on
    load (a ~57 GB download). Community mflux repos ship already-quantized
    weights (~9-27 GB) whose id carries a quant tag; those are passed straight
    through as ``model_path`` with ``quantize=None`` — re-quantizing an
    already-quantized checkpoint makes mflux error.
    """
    if model_name and Path(model_name).expanduser().is_dir():
        return True
    return bool(_QUANT_TAG_RE.search(model_name or ""))


class ImageGenerationEngine:
    """Adapter over a single mflux model family.

    One instance owns one lazily-loaded mflux model. ``generate`` is blocking
    (the caller runs it off the event loop) and returns encoded PNG bytes so
    the transport never has to touch the filesystem.
    """

    def __init__(self, model_name: str, *, quantize: int | None = _DEFAULT_QUANTIZE) -> None:
        self.model_name = model_name
        self.family = _detect_family(model_name)
        self.is_edit = self.family == "qwen-image-edit"
        self.default_steps = _DEFAULT_STEPS_BY_FAMILY.get(self.family, 4)
        self._prequantized = _looks_like_prequantized(model_name)
        # ``None`` when the repo is already quantized — passing a quantize width
        # for a pre-quantized checkpoint makes mflux re-quantize and error.
        self._quantize = None if self._prequantized else quantize
        self._model = None
        self._lock = _PROCESS_GENERATION_LOCK

    def _build_model(self):
        """Instantiate the backing mflux model (import-lazy)."""
        from mflux.models.common.config.model_config import ModelConfig

        # A pre-quantized repo / local dir is handed to mflux verbatim as
        # ``model_path``; a canonical repo is selected through ``ModelConfig``
        # so mflux downloads the official weights and quantizes on load.
        model_path = self.model_name if self._prequantized else None

        if self.family == "flux2-klein":
            from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

            return Flux2Klein(
                quantize=self._quantize,
                model_path=model_path,
                model_config=ModelConfig.flux2_klein_4b(),
            )
        if self.family == "z-image":
            from mflux.models.z_image.variants.z_image import ZImage

            return ZImage(
                quantize=self._quantize,
                model_path=model_path,
                model_config=ModelConfig.z_image_turbo(),
            )
        if self.family == "qwen-image-edit":
            from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit

            return QwenImageEdit(quantize=self._quantize, model_path=model_path)
        if self.family == "qwen-image":
            from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

            return QwenImage(quantize=self._quantize, model_path=model_path)

        from mflux.models.flux.variants.txt2img.flux import Flux1

        config = ModelConfig.schnell() if self.family == "flux-schnell" else ModelConfig.dev()
        return Flux1(quantize=self._quantize, model_path=model_path, model_config=config)

    def _ensure_loaded(self):
        if self._model is None:
            try:
                self._model = self._build_model()
            except ImageRuntimeError:
                raise
            except Exception as exc:  # noqa: BLE001 — surface a clean API error
                raise ImageRuntimeError(
                    f"Failed to load image model '{self.model_name}': {exc}"
                ) from exc
        return self._model

    def generate(
        self,
        *,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,
        seed: int = 0,
        guidance: float = 4.0,
        negative_prompt: str | None = None,
        image_paths: list[str] | None = None,
    ) -> bytes:
        """Generate one image and return it as PNG bytes.

        ``image_paths`` is required for the edit family and rejected for the
        text-to-image families so a mis-routed request fails loud instead of
        silently ignoring the conditioning image.
        """
        if self.is_edit and not image_paths:
            raise ImageRuntimeError(
                "qwen-image-edit requires at least one input image (image_paths)."
            )
        if not self.is_edit and image_paths:
            raise ImageRuntimeError(
                f"{self.family} is text-to-image only and does not accept input images; "
                "use the qwen-image-edit model for image editing."
            )

        with self._lock:
            model = self._ensure_loaded()
            try:
                if self.is_edit:
                    # Edit derives its output canvas from the input image and
                    # must NOT be given an explicit width/height. mflux fixes the
                    # VAE conditioning latents to a 1024²-area canvas of the input
                    # aspect ratio (``_compute_dimensions``), while the denoised
                    # latents use ``config.width/height``. Any mismatch between
                    # the two — e.g. forcing 512×512 against 1024²-derived
                    # conditioning — desyncs the RoPE position ids and the model
                    # emits pure noise (a valid, correctly-sized PNG of static).
                    # Passing ``None`` lets mflux size the target to match the
                    # conditioning, exactly like its edit CLI.
                    result = model.generate_image(
                        seed=seed,
                        prompt=prompt,
                        image_paths=image_paths,
                        num_inference_steps=num_inference_steps,
                        height=None,
                        width=None,
                        guidance=guidance,
                        negative_prompt=negative_prompt,
                    )
                else:
                    result = model.generate_image(
                        seed=seed,
                        prompt=prompt,
                        num_inference_steps=num_inference_steps,
                        height=height,
                        width=width,
                        guidance=guidance,
                        negative_prompt=negative_prompt,
                    )
            except ImageRuntimeError:
                raise
            except Exception as exc:  # noqa: BLE001 — surface a clean API error
                raise ImageRuntimeError(f"Image generation failed: {exc}") from exc

        return self._encode_png(result)

    @staticmethod
    def _encode_png(result) -> bytes:
        """Encode an mflux ``GeneratedImage`` to PNG bytes without touching disk."""
        pil_image = getattr(result, "image", None)
        if pil_image is None:
            raise ImageRuntimeError("Image backend returned no image data.")
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return buffer.getvalue()
