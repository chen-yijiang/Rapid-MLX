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
commit ``c628fe4392d963557c3013c2709e6d3b67bca79d`` at the time of
pinning — the alias tracks the mutable ``main`` ref, not this SHA;
``aliases.json`` has no revision-pin field today, and adding one is a
system-wide change out of scope here) keeps the text encoder
full-precision (verified: zero ``scale``/``bias`` keys, correct
``[152064, 3584]`` embed shape) and was confirmed end-to-end: both a
plain text-to-image prompt and a text-rendering prompt ("RAPID MLX" on a
sign) produced correct images with this project's pinned mflux.

Provenance: the ``mflux-community`` HF account has published ~50 repos
as a systematic multi-family, multi-bitwidth conversion pipeline (krea-2,
z-image-turbo, flux2-klein-9b, each at q3-q8/bf16) — a pattern, not a
one-off upload — though it carries no README/license file and is not a
Hugging Face-verified org; this is a real supply-chain tradeoff against
filipstrand (the mflux maintainer personally) that a future maintainer
should weigh if a more authoritative source appears.

Guarded going forward, not just for this one repo:
``ImageGenerationEngine._verify_text_encoder_not_quantized`` is a
structural preflight that reads ONLY the text encoder's safetensors
header (no tensor data, no network beyond what is already cached) and
refuses to build the model if ``embed_tokens`` is packed/quantized — so a
FUTURE re-pin to a different repo with this same defect fails loudly
before generation starts, not three prompts later.

These tests guard the things a data-only alias can still get wrong:

  1. It routes to the mflux image lane (``modality == "image-gen"``),
     not the default text lane.
  2. The pinned repo is a *pre-quantized* mflux checkpoint, so the
     loader takes the ``model_path`` branch instead of pulling the
     ~57 GB canonical ``Qwen/Qwen-Image`` weights and quantizing on
     load (which would also error against an already-quantized repo).
  3. The pinned repo is specifically NOT one with a quantized text
     encoder — see above. A future re-pin (e.g. chasing a smaller
     download) must not silently reintroduce the corruption, whether by
     shape (structurally guarded now) or by name (guarded by the
     regression pin below).
"""

from __future__ import annotations

import json
import struct

import pytest

from vllm_mlx.image.engine import (
    ImageGenerationEngine,
    ImageRuntimeError,
    _detect_family,
    _looks_like_prequantized,
)
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
    # Full-precision text encoder + 6-bit transformer measured ~55.7 GiB
    # peak RSS during a real generation (`/usr/bin/time -l`) at 1024x1024 —
    # the API/GUI default resolution, not the smaller size this was first
    # measured at. The floor gates it off Macs that cannot hold that
    # resident.
    profile = resolve_profile("qwen-image")
    assert getattr(profile, "min_memory_gb", None) == 64


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
        assert estimate_model_bytes(name) == int(55.7 * _GIB)


# MARK: - Structural guard against a quantized text encoder
#
# The regression these tests actually need to pin is not "this one repo name
# is banned" (trivially defeated by a re-pin to a different broken repo) but
# "a Qwen-Image checkpoint whose text encoder is packed/quantized is refused
# BEFORE generation starts", using nothing but a synthetic safetensors header
# — no network, no multi-gigabyte download, no real mflux model. See
# ``ImageGenerationEngine._verify_text_encoder_not_quantized``.


def _write_safetensors_header(path, header: dict) -> None:
    """Write a syntactically valid ``.safetensors`` file carrying only
    ``header`` — enough for header-only readers; the tensor data section is
    empty because nothing under test reads past the header."""
    payload = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)


def _make_snapshot(
    tmp_path, embed_header_entry: dict, *, extra_keys: dict | None = None
):
    snapshot = tmp_path / "snapshot"
    text_encoder = snapshot / "text_encoder"
    text_encoder.mkdir(parents=True)
    shard_name = "0.safetensors"
    (text_encoder / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"encoder.embed_tokens.weight": shard_name}})
    )
    header = {"encoder.embed_tokens.weight": embed_header_entry}
    if extra_keys:
        header.update(extra_keys)
    _write_safetensors_header(text_encoder / shard_name, header)
    return snapshot


def _engine_for_family_check(monkeypatch, snapshot) -> ImageGenerationEngine:
    # ``ImageGenerationEngine.__init__`` does real work (family detection,
    # prequantized sniffing) that only needs a plausible qwen-image repo
    # name — no engine method under test touches the network or mflux itself.
    engine = ImageGenerationEngine("mflux-community/qwen-image-mflux-q6")
    # ``_verify_text_encoder_not_quantized`` imports ``mflux_local_snapshot``
    # from ``_download_gate`` inside the method body, so patching the
    # source module's attribute is what the late-bound import picks up.
    monkeypatch.setattr(
        "vllm_mlx._download_gate.mflux_local_snapshot",
        lambda _repo: str(snapshot),
    )
    return engine


def test_full_precision_text_encoder_passes(tmp_path, monkeypatch) -> None:
    snapshot = _make_snapshot(tmp_path, {"shape": [152064, 3584], "dtype": "F16"})
    engine = _engine_for_family_check(monkeypatch, snapshot)
    engine._verify_text_encoder_not_quantized()  # must not raise


def test_packed_shape_is_refused(tmp_path, monkeypatch) -> None:
    # The exact failure mode reproduced against filipstrand's repo: a packed
    # last dimension instead of the real hidden size.
    snapshot = _make_snapshot(tmp_path, {"shape": [152064, 672], "dtype": "U32"})
    engine = _engine_for_family_check(monkeypatch, snapshot)
    with pytest.raises(ImageRuntimeError, match="quantized text encoder"):
        engine._verify_text_encoder_not_quantized()


def test_scales_sibling_is_refused_even_with_a_plausible_shape(
    tmp_path, monkeypatch
) -> None:
    # Belt-and-suspenders: a quantization scheme that happens to keep the
    # embedding's own shape at [vocab, hidden] is still caught by the
    # scales/biases siblings every quantized linear/embedding carries.
    snapshot = _make_snapshot(
        tmp_path,
        {"shape": [152064, 3584], "dtype": "F16"},
        extra_keys={
            "encoder.embed_tokens.weight.scales": {
                "shape": [152064, 56],
                "dtype": "F16",
            }
        },
    )
    engine = _engine_for_family_check(monkeypatch, snapshot)
    with pytest.raises(ImageRuntimeError, match="quantized text encoder"):
        engine._verify_text_encoder_not_quantized()


def test_non_qwen_family_is_not_checked(tmp_path, monkeypatch) -> None:
    # The 3584 hidden-size expectation is Qwen2-specific; z-image/flux2-klein
    # text encoders have entirely different architectures and must not be
    # held to it.
    engine = ImageGenerationEngine("filipstrand/Z-Image-Turbo-mflux-4bit")
    assert engine.family != "qwen-image"
    engine._verify_text_encoder_not_quantized()  # no-op, must not raise


def test_missing_index_does_not_raise(tmp_path, monkeypatch) -> None:
    # No index (or an unreadable one) is outside what this check can vouch
    # for — ``_verify_weights_complete``'s completeness check is what must
    # catch that; this guard degrades to a no-op rather than a false claim.
    empty_snapshot = tmp_path / "empty"
    empty_snapshot.mkdir()
    engine = _engine_for_family_check(monkeypatch, empty_snapshot)
    engine._verify_text_encoder_not_quantized()  # no-op, must not raise
