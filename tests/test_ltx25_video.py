"""LTX-2.5 external MLX runtime integration tests."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from vllm_mlx.model_aliases import resolve_profile
from vllm_mlx.runtime import video_lane
from vllm_mlx.runtime.video_lane import VideoEngine, VideoRuntimeError
from vllm_mlx.video import ltx25


def test_ltx25_alias_routes_to_video_lane() -> None:
    profile = resolve_profile("ltx-2.5-mlx-q8")
    assert profile is not None
    assert profile.hf_path == "MrMofer/ltx-2.5-mlx-q8"
    assert profile.modality == "video-gen"
    assert profile.min_memory_gb == 24


def test_ltx25_capabilities_match_distilled_controls() -> None:
    from vllm_mlx.routes.video import _video_capabilities

    capabilities = _video_capabilities(
        SimpleNamespace(model_name="MrMofer/ltx-2.5-mlx-q8", video_family="ltx-2.5")
    )
    assert capabilities["family"] == "ltx-2.5"
    assert capabilities["limits"]["size"]["width"]["multiple_of"] == 32
    assert capabilities["limits"]["workload"]["dimension_rounding"] == "ceil_to_32"
    assert capabilities["controls"]["guidance_scale"] is None
    assert capabilities["controls"]["negative_prompt"] is False
    assert capabilities["controls"]["conditioning_strength"] == {
        "minimum": 0.0,
        "maximum": 1.0,
    }


def test_ltx25_runtime_preflight_fails_before_download(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ltx25, "resolve_ltx25_runtime", lambda: None)
    monkeypatch.setattr(
        video_lane, "_resolve_ffmpeg", lambda: "/opt/homebrew/bin/ffmpeg"
    )

    with pytest.raises(SystemExit, match="2"):
        video_lane.require_video_runtime_or_exit("MrMofer/ltx-2.5-mlx-q8")

    error = capsys.readouterr().err
    assert ltx25.LTX25_RUNTIME_COMMIT in error
    assert "video generation guide" in error


def test_ltx25_runtime_override_must_be_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "ltx-2-mlx"
    runtime.write_text("#!/bin/sh\n")
    monkeypatch.setenv("RAPID_MLX_LTX25_RUNTIME", str(runtime))
    monkeypatch.setattr(
        ltx25.shutil,
        "which",
        lambda _: pytest.fail(
            "an invalid explicit override must not fall back to PATH"
        ),
    )

    assert ltx25.resolve_ltx25_runtime() is None

    runtime.chmod(0o755)
    assert ltx25.resolve_ltx25_runtime() == str(runtime)


def test_serve_routes_ltx25_model_to_specific_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_mlx import cli

    class PreflightReachedError(RuntimeError):
        pass

    def stop_at_preflight(model_name: str) -> None:
        assert model_name == "ltx-2.5-mlx-q8"
        raise PreflightReachedError

    monkeypatch.setattr(video_lane, "require_video_runtime_or_exit", stop_at_preflight)
    args = SimpleNamespace(model="ltx-2.5-mlx-q8", max_tokens=None, watchdog_ppid=None)
    with pytest.raises(PreflightReachedError):
        cli.serve_command(args)


def test_ltx25_engine_invokes_pinned_runtime_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.mp4"
    image = tmp_path / "reference.png"
    image.write_bytes(b"png")
    calls: list[list[str]] = []
    monkeypatch.setattr(ltx25, "resolve_ltx25_runtime", lambda: "/trusted/ltx-2-mlx")

    def run(command: list[str], **kwargs) -> None:
        calls.append(command)
        output.write_bytes(b"mp4-with-audio")

    monkeypatch.setattr(ltx25.subprocess, "run", run)
    ltx25.LTX25VideoEngine("MrMofer/ltx-2.5-mlx-q8").generate(
        prompt="a fox",
        output_path=output,
        width=704,
        height=480,
        num_frames=97,
        fps=24,
        seed=7,
        image=image,
        conditioning_strength=0.6,
    )

    command = calls[0]
    assert command[0] == "/trusted/ltx-2-mlx"
    assert command[1:3] == ["generate", "--model"]
    assert "--distilled" in command
    assert "--low-ram" in command
    assert command[command.index("--image") + 1 :] == [str(image), "0", "0.6"]
    assert output.read_bytes() == b"mp4-with-audio"


def test_ltx25_engine_reports_subprocess_failure_without_leaking_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ltx25, "resolve_ltx25_runtime", lambda: "/trusted/ltx-2-mlx")

    def fail(*args, **kwargs) -> None:
        raise subprocess.CalledProcessError(1, ["ltx-2-mlx"], stderr="secret")

    monkeypatch.setattr(ltx25.subprocess, "run", fail)
    engine = ltx25.LTX25VideoEngine("MrMofer/ltx-2.5-mlx-q8")
    with pytest.raises(ltx25.LTX25BackendError) as exc:
        engine.generate(
            prompt="a fox",
            output_path=tmp_path / "result.mp4",
            width=704,
            height=480,
            num_frames=97,
            fps=24,
            seed=7,
            image=None,
        )
    assert "secret" not in str(exc.value)


def test_video_engine_selects_ltx25_and_preserves_generated_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def generate(self, **kwargs) -> None:
        captured.update(kwargs)
        kwargs["output_path"].write_bytes(b"mp4-with-audio")

    monkeypatch.setattr(ltx25.LTX25VideoEngine, "generate", generate)
    monkeypatch.setattr(
        VideoEngine,
        "_crop_generated_output",
        lambda *args, **kwargs: captured.update(crop=kwargs),
    )
    engine = VideoEngine("MrMofer/ltx-2.5-mlx-q8")
    output = tmp_path / "result.mp4"
    engine.generate(
        prompt="a fox",
        output_path=output,
        width=704,
        height=512,
        num_frames=97,
        fps=24,
        seed=7,
        image=None,
    )

    assert engine.video_family == "ltx-2.5"
    assert output.read_bytes() == b"mp4-with-audio"
    assert captured["crop"]["family"] == "LTX-2.5"


def test_ltx25_distilled_rejects_cfg_controls() -> None:
    engine = VideoEngine("MrMofer/ltx-2.5-mlx-q8")
    with pytest.raises(VideoRuntimeError, match="does not support"):
        engine.generate(
            prompt="a fox",
            output_path=Path("result.mp4"),
            width=704,
            height=480,
            num_frames=97,
            fps=24,
            seed=7,
            image=None,
            guidance_scale=4.0,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unsupported",
    [{"guidance_scale": 4.0}, {"negative_prompt": "blurry"}],
)
async def test_ltx25_route_rejects_unsupported_cfg_before_queueing(
    monkeypatch: pytest.MonkeyPatch, unsupported: dict[str, object]
) -> None:
    from vllm_mlx.routes import video

    engine = SimpleNamespace(
        model_name="MrMofer/ltx-2.5-mlx-q8", video_family="ltx-2.5"
    )
    monkeypatch.setattr(video, "_video_engine", lambda: engine)
    monkeypatch.setattr(video, "_accepting_jobs", True)
    before = set(video._jobs)

    with pytest.raises(HTTPException, match="does not support") as exc:
        await video.create_video(
            prompt="a fox",
            model="ltx-2.5-mlx-q8",
            seconds="1",
            size="704x512",
            seed=7,
            input_reference=None,
            **unsupported,
        )

    assert exc.value.status_code == 400
    assert set(video._jobs) == before
