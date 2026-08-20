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
    tmp_path,
    embed_header_entry: dict | None,
    *,
    weight_map_extra: dict | None = None,
    write_shard: bool = True,
):
    """A synthetic ``text_encoder/`` layout: an index naming
    ``encoder.embed_tokens.weight``'s shard, plus (usually) that shard's
    header. ``weight_map_extra`` adds sibling entries to the INDEX (real
    quantization scale/bias keys land there, possibly in a different shard
    than the weight itself — see the check's docstring); ``write_shard=False``
    exercises the index naming a shard that never materializes.
    """
    snapshot = tmp_path / "snapshot"
    text_encoder = snapshot / "text_encoder"
    text_encoder.mkdir(parents=True)
    shard_name = "0.safetensors"
    weight_map = {"encoder.embed_tokens.weight": shard_name}
    if weight_map_extra:
        weight_map.update(weight_map_extra)
    (text_encoder / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    if write_shard:
        header = {}
        if embed_header_entry is not None:
            header["encoder.embed_tokens.weight"] = embed_header_entry
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
        weight_map_extra={"encoder.embed_tokens.weight.scales": "0.safetensors"},
    )
    engine = _engine_for_family_check(monkeypatch, snapshot)
    with pytest.raises(ImageRuntimeError, match="quantized text encoder"):
        engine._verify_text_encoder_not_quantized()


def test_scales_sibling_in_a_different_shard_is_still_refused(
    tmp_path, monkeypatch
) -> None:
    # The sibling check reads the INDEX's full weight_map, not one shard's
    # header — a quantized embedding whose scales/biases were split into a
    # separate shard from the weight itself must not slip past.
    snapshot = _make_snapshot(
        tmp_path,
        {"shape": [152064, 3584], "dtype": "F16"},
        weight_map_extra={"encoder.embed_tokens.weight.scales": "1.safetensors"},
    )
    engine = _engine_for_family_check(monkeypatch, snapshot)
    with pytest.raises(ImageRuntimeError, match="quantized text encoder"):
        engine._verify_text_encoder_not_quantized()


def test_shard_named_by_index_but_missing_on_disk_fails_closed(
    tmp_path, monkeypatch
) -> None:
    # The index makes an explicit claim about where this tensor lives; if
    # that claim doesn't resolve to a real file, that is itself suspicious
    # and must refuse rather than silently skip the check.
    snapshot = _make_snapshot(tmp_path, None, write_shard=False)
    engine = _engine_for_family_check(monkeypatch, snapshot)
    with pytest.raises(ImageRuntimeError, match="quantized text encoder"):
        engine._verify_text_encoder_not_quantized()


def test_shard_present_but_key_absent_fails_closed(tmp_path, monkeypatch) -> None:
    # The index names a shard for embed_tokens.weight, but that shard's own
    # header doesn't actually contain it — inconsistent with what the index
    # promised, so this must not be read as "nothing to check".
    snapshot = _make_snapshot(tmp_path, embed_header_entry=None)
    engine = _engine_for_family_check(monkeypatch, snapshot)
    with pytest.raises(ImageRuntimeError, match="quantized text encoder"):
        engine._verify_text_encoder_not_quantized()


def test_oversized_header_length_is_rejected_without_reading_it(tmp_path) -> None:
    # A corrupt or hostile shard could claim a multi-gigabyte header to force
    # a multi-gigabyte read before JSON parsing ever runs. The length prefix
    # is checked against an absolute cap BEFORE the file is read further.
    shard = tmp_path / "0.safetensors"
    oversized = ImageGenerationEngine._SAFETENSORS_HEADER_MAX_BYTES + 1
    shard.write_bytes(struct.pack("<Q", oversized))
    assert ImageGenerationEngine._read_safetensors_header(str(shard)) is None


def test_shard_reached_via_symlink_is_accepted(tmp_path, monkeypatch) -> None:
    # A Hugging Face hub cache's snapshot files ARE symlinks into a sibling
    # blobs/ directory — the real shape this check runs against in
    # production. A resolve()-then-compare-parents guard walks straight out
    # of text_encoder_dir through that symlink and false-positives on every
    # real cached checkpoint (caught live against mflux-community's repo);
    # the string-only guard must accept this shape.
    snapshot = tmp_path / "snapshot"
    text_encoder = snapshot / "text_encoder"
    text_encoder.mkdir(parents=True)
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    real_file = blobs / "deadbeef"
    _write_safetensors_header(
        real_file,
        {"encoder.embed_tokens.weight": {"shape": [152064, 3584], "dtype": "F16"}},
    )
    (text_encoder / "0.safetensors").symlink_to(real_file)
    (text_encoder / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"encoder.embed_tokens.weight": "0.safetensors"}})
    )
    engine = _engine_for_family_check(monkeypatch, snapshot)
    engine._verify_text_encoder_not_quantized()  # must not raise


def test_shard_path_traversal_is_rejected(tmp_path, monkeypatch) -> None:
    # A shard name escaping text_encoder/ (path traversal, or an absolute
    # path elsewhere on disk) must not be trusted even though it would
    # resolve to a real file.
    outside = tmp_path / "outside.safetensors"
    _write_safetensors_header(
        outside,
        {"encoder.embed_tokens.weight": {"shape": [152064, 3584], "dtype": "F16"}},
    )
    snapshot = _make_snapshot(
        tmp_path, {"shape": [152064, 3584], "dtype": "F16"}
    )  # baseline valid shard, then override the index to point outside
    (snapshot / "text_encoder" / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"encoder.embed_tokens.weight": "../outside.safetensors"}}
        )
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
