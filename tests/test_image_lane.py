# SPDX-License-Identifier: Apache-2.0
"""Hermetic tests for the mflux image-generation lane.

No network, no disk, no real weights: the mflux model is replaced by a fake
that returns a tiny in-memory PIL image, so these exercise Rapid-MLX's own
validation / dispatch / transport contract rather than the diffusion pipeline.
"""

import base64
import io
import types

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from vllm_mlx.api.models import ImageGenerationRequest
from vllm_mlx.image.engine import ImageGenerationEngine, ImageRuntimeError
from vllm_mlx.runtime.image_lane import ImageEngine

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class _FakeGeneratedImage:
    def __init__(self, image):
        self.image = image


class _FakeModel:
    """Stand-in for an mflux Flux1 / QwenImage model."""

    def __init__(self):
        self.calls = []

    def generate_image(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeGeneratedImage(Image.new("RGB", (8, 8), (200, 40, 40)))


# --------------------------------------------------------------------------- #
# Family detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "hf_path,expected_family,is_edit",
    [
        ("black-forest-labs/FLUX.1-schnell", "flux-schnell", False),
        ("Qwen/Qwen-Image", "qwen-image", False),
        ("Qwen/Qwen-Image-Edit-2509", "qwen-image-edit", True),
    ],
)
def test_detect_family(hf_path, expected_family, is_edit):
    engine = ImageGenerationEngine(hf_path)
    assert engine.family == expected_family
    assert engine.is_edit is is_edit


def test_unknown_family_raises():
    with pytest.raises(ImageRuntimeError, match="Unsupported image model"):
        ImageGenerationEngine("stabilityai/some-unwired-model")


@pytest.mark.parametrize(
    "hf_path,family",
    [
        # ``<n>bit`` convention — the repos the -4bit aliases point at
        ("dhairyashil/FLUX.1-schnell-mflux-4bit", "flux-schnell"),
        ("OsaurusAI/Qwen-Image-mflux-4bit", "qwen-image"),
        ("filipstrand/Qwen-Image-mflux-6bit", "qwen-image"),
        # ``q<n>`` convention — the repo the qwen-image-edit-4bit alias points at
        ("OsaurusAI/Qwen-Image-Edit-mflux-q4", "qwen-image-edit"),
    ],
)
def test_prequantized_repo_disables_onload_quantize(hf_path, family):
    engine = ImageGenerationEngine(hf_path)
    assert engine._prequantized is True
    assert engine._quantize is None  # never re-quantize an already-quantized repo
    assert engine.family == family


@pytest.mark.parametrize(
    "hf_path",
    [
        # Base repos carry no quant tag — the leading "q" of "Qwen" must not be
        # mistaken for a "q<n>" tag, so on-load quantization stays enabled.
        "black-forest-labs/FLUX.1-schnell",
        "Qwen/Qwen-Image",
        "Qwen/Qwen-Image-Edit-2509",
    ],
)
def test_canonical_repo_keeps_onload_quantize(hf_path):
    engine = ImageGenerationEngine(hf_path)
    assert engine._prequantized is False
    assert engine._quantize == 4


# --------------------------------------------------------------------------- #
# ImageGenerationEngine.generate
# --------------------------------------------------------------------------- #
def test_generate_returns_png_bytes():
    engine = ImageGenerationEngine("black-forest-labs/FLUX.1-schnell")
    engine._model = _FakeModel()  # bypass lazy load / real weights
    png = engine.generate(prompt="a red fox", width=512, height=512, seed=7)
    assert png.startswith(_PNG_MAGIC)
    # It round-trips as a real PNG.
    assert Image.open(io.BytesIO(png)).size == (8, 8)


def test_generate_is_lazy(monkeypatch):
    engine = ImageGenerationEngine("black-forest-labs/FLUX.1-schnell")
    built = _FakeModel()
    monkeypatch.setattr(engine, "_build_model", lambda: built)
    assert engine._model is None  # not loaded at construction
    engine.generate(prompt="x", seed=1)
    assert engine._model is built  # loaded on first generate
    assert built.calls[0]["prompt"] == "x"


def test_edit_family_requires_image_paths():
    engine = ImageGenerationEngine("Qwen/Qwen-Image-Edit-2509")
    engine._model = _FakeModel()
    with pytest.raises(ImageRuntimeError, match="requires at least one input image"):
        engine.generate(prompt="make it blue", image_paths=None)


def test_txt2img_family_rejects_image_paths():
    engine = ImageGenerationEngine("Qwen/Qwen-Image")
    engine._model = _FakeModel()
    with pytest.raises(ImageRuntimeError, match="text-to-image only"):
        engine.generate(prompt="a cat", image_paths=["/tmp/x.png"])


def test_edit_family_passes_image_paths_through():
    engine = ImageGenerationEngine("Qwen/Qwen-Image-Edit-2509")
    fake = _FakeModel()
    engine._model = fake
    engine.generate(prompt="add a hat", image_paths=["/tmp/in.png"], seed=3)
    assert fake.calls[0]["image_paths"] == ["/tmp/in.png"]


def test_edit_forces_none_dimensions_even_when_size_requested():
    # Regression: mflux edit fixes the conditioning latents to a 1024²-area
    # canvas of the input; forcing a mismatched width/height (e.g. 512×512)
    # desyncs the RoPE ids and yields pure noise. The engine must hand mflux
    # None so it sizes the target to match the conditioning.
    engine = ImageGenerationEngine("Qwen/Qwen-Image-Edit-2509")
    fake = _FakeModel()
    engine._model = fake
    engine.generate(
        prompt="add a hat", image_paths=["/tmp/in.png"],
        width=512, height=512, seed=3,
    )
    assert fake.calls[0]["width"] is None
    assert fake.calls[0]["height"] is None


def test_txt2img_family_honors_requested_dimensions():
    # The noise trap is edit-only: text-to-image must still respect width/height.
    engine = ImageGenerationEngine("Qwen/Qwen-Image")
    fake = _FakeModel()
    engine._model = fake
    engine.generate(prompt="a cat", width=768, height=512, seed=3)
    assert fake.calls[0]["width"] == 768
    assert fake.calls[0]["height"] == 512


def test_backend_failure_becomes_runtime_error():
    engine = ImageGenerationEngine("Qwen/Qwen-Image")

    class _Boom:
        def generate_image(self, **kwargs):
            raise ValueError("metal exploded")

    engine._model = _Boom()
    with pytest.raises(ImageRuntimeError, match="Image generation failed"):
        engine.generate(prompt="x")


# --------------------------------------------------------------------------- #
# ImageEngine adapter
# --------------------------------------------------------------------------- #
def test_image_engine_adapter_is_duck_typed():
    engine = ImageEngine("black-forest-labs/FLUX.1-schnell")
    assert engine.is_image_gen is True
    assert engine._loaded is True
    assert engine.family == "flux-schnell"
    assert engine.is_edit is False


# --------------------------------------------------------------------------- #
# Route: /v1/images/generations
# --------------------------------------------------------------------------- #
class _FakeImageEngine:
    is_image_gen = True

    def __init__(self, is_edit=False):
        self.is_edit = is_edit
        self.seeds = []
        self.image_paths_seen = []
        self.dims_seen = []
        self.steps_seen = []

    def generate(self, *, prompt, num_inference_steps, seed, guidance,
                 negative_prompt, width=None, height=None, image_paths=None):
        # The edit route omits width/height (the engine derives them from the
        # input image); the generations route always supplies them.
        self.seeds.append(seed)
        self.image_paths_seen.append(image_paths)
        self.dims_seen.append((width, height))
        self.steps_seen.append(num_inference_steps)
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4), (10, 20, 30)).save(buffer, format="PNG")
        return buffer.getvalue()


def _png_upload_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (90, 90, 90)).save(buffer, format="PNG")
    return buffer.getvalue()


def _patch_engine(monkeypatch, engine):
    monkeypatch.setattr(
        "vllm_mlx.config.get_config", lambda: types.SimpleNamespace(engine=engine)
    )


@pytest.fixture
def client():
    from vllm_mlx.server import app

    return TestClient(app)


def test_route_409_when_no_image_model(client, monkeypatch):
    _patch_engine(monkeypatch, None)
    resp = client.post("/v1/images/generations", json={"prompt": "a fox"})
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "image_model_not_loaded"


def test_route_409_when_edit_model_loaded(client, monkeypatch):
    _patch_engine(monkeypatch, _FakeImageEngine(is_edit=True))
    resp = client.post("/v1/images/generations", json={"prompt": "a fox"})
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "wrong_image_endpoint"


def test_route_400_url_response_format(client, monkeypatch):
    _patch_engine(monkeypatch, _FakeImageEngine())
    resp = client.post(
        "/v1/images/generations",
        json={"prompt": "a fox", "response_format": "url"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "unsupported_response_format"


def test_route_happy_path_returns_b64(client, monkeypatch):
    _patch_engine(monkeypatch, _FakeImageEngine())
    resp = client.post(
        "/v1/images/generations",
        json={"prompt": "a red fox", "size": "512x512", "seed": 42},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "created" in body and len(body["data"]) == 1
    raw = base64.b64decode(body["data"][0]["b64_json"])
    assert raw.startswith(_PNG_MAGIC)


def test_route_multi_image_offsets_seed(client, monkeypatch):
    engine = _FakeImageEngine()
    _patch_engine(monkeypatch, engine)
    resp = client.post(
        "/v1/images/generations",
        json={"prompt": "a fox", "n": 3, "seed": 100},
    )
    assert resp.status_code == 200
    assert len(resp.json()["data"]) == 3
    assert engine.seeds == [100, 101, 102]  # per-index seed offset


@pytest.mark.parametrize("bad_size", ["1023x1024", "100x100", "3000x512", "oops"])
def test_route_rejects_bad_size(client, monkeypatch, bad_size):
    _patch_engine(monkeypatch, _FakeImageEngine())
    resp = client.post(
        "/v1/images/generations", json={"prompt": "x", "size": bad_size}
    )
    assert resp.status_code in (400, 422)


@pytest.mark.parametrize("bad_guidance", [float("nan"), float("inf"), float("-inf")])
def test_request_model_rejects_nonfinite_guidance(bad_guidance):
    # NaN/inf fail the ge=0 / le=20 comparisons, so the bounds reject them.
    with pytest.raises(ValueError):
        ImageGenerationRequest(prompt="x", guidance=bad_guidance)


# --------------------------------------------------------------------------- #
# Route: /v1/images/edits
# --------------------------------------------------------------------------- #
def test_edit_409_when_no_image_model(client, monkeypatch):
    _patch_engine(monkeypatch, None)
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", _png_upload_bytes(), "image/png")},
        data={"prompt": "add a hat"},
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "image_model_not_loaded"


def test_edit_409_when_txt2img_model_loaded(client, monkeypatch):
    _patch_engine(monkeypatch, _FakeImageEngine(is_edit=False))
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", _png_upload_bytes(), "image/png")},
        data={"prompt": "add a hat"},
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "wrong_image_endpoint"


def test_edit_happy_path_returns_b64_and_passes_image(client, monkeypatch):
    engine = _FakeImageEngine(is_edit=True)
    _patch_engine(monkeypatch, engine)
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", _png_upload_bytes(), "image/png")},
        data={"prompt": "make the sky blue", "size": "512x512", "seed": "5"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["data"]) == 1
    assert base64.b64decode(body["data"][0]["b64_json"]).startswith(_PNG_MAGIC)
    # The uploaded image was written to a temp file and passed to the engine.
    assert engine.image_paths_seen[0] is not None
    assert len(engine.image_paths_seen[0]) == 1
    # Even though the request carried size=512x512, the edit route does NOT
    # thread dimensions — the engine derives them from the input image.
    assert engine.dims_seen[0] == (None, None)


def test_edit_defaults_to_20_steps_when_unspecified(client, monkeypatch):
    # FLUX.1-schnell generation defaults to 4 distilled steps, but a
    # non-distilled edit needs ~20 to resolve structure; the edit route must
    # not inherit the 4-step generation default.
    engine = _FakeImageEngine(is_edit=True)
    _patch_engine(monkeypatch, engine)
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", _png_upload_bytes(), "image/png")},
        data={"prompt": "add a hat"},
    )
    assert resp.status_code == 200
    assert engine.steps_seen == [20]


def test_edit_honors_explicit_steps(client, monkeypatch):
    engine = _FakeImageEngine(is_edit=True)
    _patch_engine(monkeypatch, engine)
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", _png_upload_bytes(), "image/png")},
        data={"prompt": "add a hat", "steps": "8"},
    )
    assert resp.status_code == 200
    assert engine.steps_seen == [8]


def test_edit_multi_offsets_seed(client, monkeypatch):
    engine = _FakeImageEngine(is_edit=True)
    _patch_engine(monkeypatch, engine)
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", _png_upload_bytes(), "image/png")},
        data={"prompt": "x", "n": "2", "seed": "50"},
    )
    assert resp.status_code == 200
    assert len(resp.json()["data"]) == 2
    assert engine.seeds == [50, 51]


def test_edit_400_empty_prompt(client, monkeypatch):
    _patch_engine(monkeypatch, _FakeImageEngine(is_edit=True))
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", _png_upload_bytes(), "image/png")},
        data={"prompt": "   "},
    )
    assert resp.status_code == 400


def test_edit_400_bad_size(client, monkeypatch):
    _patch_engine(monkeypatch, _FakeImageEngine(is_edit=True))
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", _png_upload_bytes(), "image/png")},
        data={"prompt": "x", "size": "100x100"},
    )
    assert resp.status_code == 400


def test_edit_400_empty_image(client, monkeypatch):
    _patch_engine(monkeypatch, _FakeImageEngine(is_edit=True))
    resp = client.post(
        "/v1/images/edits",
        files={"image": ("in.png", b"", "image/png")},
        data={"prompt": "x"},
    )
    assert resp.status_code == 400
