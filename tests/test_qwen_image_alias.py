# SPDX-License-Identifier: Apache-2.0
"""Contract pins for the ``qwen-image`` image-gen alias.

The mflux backend already understands the ``qwen-image`` family
(``vllm_mlx/image/engine.py``); this alias is what makes
``rapid-mlx serve qwen-image`` resolve to it without the caller having
to type the raw Hugging Face path.

Repo choice, and why it is NOT ``filipstrand/Qwen-Image-mflux-6bit``
(the community's most-downloaded Qwen-Image mflux conversion): that repo
quantizes the text encoder to 6-bit (packed ``embed_tokens.weight`` shape
``[152064, 672]`` instead of the full-precision ``[152064, 3584]``), but
the mflux version this project pins (0.18.1) hardcodes
``skip_quantization=True`` for the Qwen text encoder ("quantization causes
significant semantic degradation" per mflux's own weight definition) and
never dequantizes it on load. The result is silent runtime corruption, not
a load-time error: the model builds, ``generate()`` runs, and crashes deep
in the text encoder's RMSNorm with a shape mismatch
(``[broadcast_shapes] Shapes (3584) and (1,50,672)``) — reproduced live
against this project's pinned mflux before this alias was pointed
elsewhere. ``mflux-community/qwen-image-mflux-q6`` (uploaded 2026-08-16,
current at the time of pinning) keeps the text encoder full-precision
(verified: zero ``scale``/``bias`` keys, correct ``[152064, 3584]`` embed
shape) and was confirmed end-to-end: both a plain text-to-image prompt and
a text-rendering prompt ("RAPID MLX" on a sign) produced correct images
with this project's pinned mflux.

These tests guard the things a data-only alias can still get wrong:

  1. It routes to the mflux image lane (``modality == "image-gen"``),
     not the default text lane.
  2. The pinned repo is a *pre-quantized* mflux checkpoint, so the
     loader takes the ``model_path`` branch instead of pulling the
     ~57 GB canonical ``Qwen/Qwen-Image`` weights and quantizing on
     load (which would also error against an already-quantized repo).
  3. The pinned repo is specifically NOT one with a quantized text
     encoder — see above. A future re-pin (e.g. chasing a smaller
     download) must not silently reintroduce the corruption.
"""

from __future__ import annotations

from vllm_mlx.image.engine import _detect_family, _looks_like_prequantized
from vllm_mlx.model_aliases import resolve_profile
from vllm_mlx.runtime.resident_models import estimate_model_bytes

_GIB = 1024**3

# The specific repo this alias must NOT point at: its text encoder is
# quantized in a way the pinned mflux version cannot load correctly (see
# module docstring). Regression pin for "someone re-pins to a smaller
# download and silently reintroduces broken generation".
_KNOWN_INCOMPATIBLE_REPO = "filipstrand/Qwen-Image-mflux-6bit"


def test_qwen_image_alias_routes_to_image_lane() -> None:
    profile = resolve_profile("qwen-image")
    assert profile is not None
    assert profile.modality == "image-gen"
    assert profile.hf_path == "mflux-community/qwen-image-mflux-q6"


def test_qwen_image_alias_does_not_point_at_the_known_broken_repo() -> None:
    profile = resolve_profile("qwen-image")
    assert profile.hf_path != _KNOWN_INCOMPATIBLE_REPO


def test_qwen_image_alias_declares_a_memory_floor() -> None:
    # Full-precision text encoder + 6-bit transformer measured ~40.2 GiB
    # peak RSS during a real generation (`/usr/bin/time -l`, 512x512).
    # The floor gates it off Macs that cannot hold that resident.
    profile = resolve_profile("qwen-image")
    assert getattr(profile, "min_memory_gb", None) == 48


def test_qwen_image_repo_is_detected_as_qwen_image_family() -> None:
    hf_path = resolve_profile("qwen-image").hf_path
    assert _detect_family(hf_path) == "qwen-image"


def test_qwen_image_repo_is_prequantized_so_no_full_weight_pull() -> None:
    hf_path = resolve_profile("qwen-image").hf_path
    assert _looks_like_prequantized(hf_path) is True


def test_qwen_image_resident_estimate_is_not_the_bare_default() -> None:
    # The alias carries no digit params, so without a known-image entry the
    # fallback would charge the 4 GB default and mis-admit a ~29 GB model.
    # Both the alias and the pinned repo id must resolve to the real charge.
    for name in ("qwen-image", resolve_profile("qwen-image").hf_path):
        assert estimate_model_bytes(name) == int(40.2 * _GIB)
