# SPDX-License-Identifier: Apache-2.0
"""Adapter for the standalone LTX-2.5 MLX runtime."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

LTX25_RUNTIME_COMMIT = "57952288076766abe27dda3a774b2c24f7346977"
LTX25_RUNTIME_REPOSITORY = "https://github.com/MrMoferFRAN/ltx-2-mlx.git"


def is_ltx25_model(model_name: str | None) -> bool:
    """Return whether a model identifier explicitly selects LTX-2.5."""
    if not model_name:
        return False
    normalized = model_name.casefold().replace("_", "-")
    return "ltx-2.5" in normalized or "ltx25" in normalized


def resolve_ltx25_runtime() -> str | None:
    """Resolve the separately installed, pinned LTX-2.5 CLI."""
    override = os.environ.get("RAPID_MLX_LTX25_RUNTIME", "").strip()
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.absolute())
        return None
    executable = shutil.which("ltx-2-mlx")
    return str(Path(executable).absolute()) if executable else None


class LTX25BackendError(RuntimeError):
    """Safe, public-facing error from the LTX-2.5 backend."""


class LTX25VideoEngine:
    """Run LTX-2.5 through its native MLX command-line runtime."""

    native_fps = 24

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(
        self,
        *,
        prompt: str,
        output_path: Path,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        seed: int,
        image: Path | None,
        conditioning_strength: float | None = None,
    ) -> None:
        executable = resolve_ltx25_runtime()
        if executable is None:
            raise LTX25BackendError(
                "LTX-2.5 support requires the pinned ltx-2-mlx runtime. "
                "See the LTX-2.5 setup in the video generation guide."
            )

        command = [
            executable,
            "generate",
            "--model",
            self.model_name,
            "--distilled",
            "--low-ram",
            "--quiet",
            "--prompt",
            prompt,
            "--height",
            str(height),
            "--width",
            str(width),
            "--frames",
            str(num_frames),
            "--frame-rate",
            str(fps),
            "--seed",
            str(seed),
            "--output",
            str(output_path),
        ]
        if image is not None:
            command.extend(
                [
                    "--image",
                    str(image),
                    "0",
                    str(
                        1.0 if conditioning_strength is None else conditioning_strength
                    ),
                ]
            )
        try:
            subprocess.run(command, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise LTX25BackendError(
                "LTX-2.5 generation failed; check the server logs for runtime details."
            ) from exc
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise LTX25BackendError(
                "LTX-2.5 generation completed without an MP4 output."
            )
