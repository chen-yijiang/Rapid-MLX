# SPDX-License-Identifier: Apache-2.0
"""Contract pins for the ``qwen-image`` image-gen alias.

The mflux backend already understands the ``qwen-image`` family
(``vllm_mlx/image/engine.py``); this alias is what makes
``rapid-mlx serve qwen-image`` resolve to it without the caller having
to type the raw Hugging Face path. These tests guard the two things a
data-only alias can still get wrong:

  1. It routes to the mflux image lane (``modality == "image-gen"``),
     not the default text lane.
  2. The pinned repo is a *pre-quantized* mflux checkpoint, so the
     loader takes the ``model_path`` branch instead of pulling the
     ~57 GB canonical ``Qwen/Qwen-Image`` weights and quantizing on
     load (which would also error against an already-quantized repo).
"""

from __future__ import annotations

from vllm_mlx.image.engine import _detect_family, _looks_like_prequantized
from vllm_mlx.model_aliases import resolve_profile
from vllm_mlx.runtime.resident_models import estimate_model_bytes

_GIB = 1024**3


def test_qwen_image_alias_routes_to_image_lane() -> None:
    profile = resolve_profile("qwen-image")
    assert profile is not None
    assert profile.modality == "image-gen"
    assert profile.hf_path == "filipstrand/Qwen-Image-mflux-6bit"


def test_qwen_image_alias_declares_a_memory_floor() -> None:
    # 6-bit Qwen-Image (~23 GB on disk) needs headroom to stay resident;
    # the gate keeps it off Macs that cannot hold it.
    profile = resolve_profile("qwen-image")
    assert getattr(profile, "min_memory_gb", None) == 24


def test_qwen_image_repo_is_detected_as_qwen_image_family() -> None:
    hf_path = resolve_profile("qwen-image").hf_path
    assert _detect_family(hf_path) == "qwen-image"


def test_qwen_image_repo_is_prequantized_so_no_full_weight_pull() -> None:
    hf_path = resolve_profile("qwen-image").hf_path
    assert _looks_like_prequantized(hf_path) is True


def test_qwen_image_resident_estimate_is_not_the_bare_default() -> None:
    # The alias carries no digit params, so without a known-image entry the
    # fallback would charge the 4 GB default and mis-admit a ~23 GB model.
    # Both the alias and the pinned repo id must resolve to the real charge.
    for name in ("qwen-image", resolve_profile("qwen-image").hf_path):
        assert estimate_model_bytes(name) == int(18.0 * _GIB)
