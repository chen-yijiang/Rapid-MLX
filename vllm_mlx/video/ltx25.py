# SPDX-License-Identifier: Apache-2.0
"""Adapter for the standalone LTX-2.5 MLX runtime."""

from __future__ import annotations

import io
import os
import shutil
import signal
import subprocess
import tarfile
import tempfile
import threading
from pathlib import Path

LTX25_RUNTIME_COMMIT = "57952288076766abe27dda3a774b2c24f7346977"
LTX25_RUNTIME_REPOSITORY = "https://github.com/MrMoferFRAN/ltx-2-mlx.git"
_DEFAULT_TIMEOUT_SECONDS = 7200
_TERMINATE_GRACE_SECONDS = 10
_STDIN_PROMPT_RUNNER = """\
import sys
from ltx_pipelines_mlx.cli import main

sys.argv = ["ltx-2-mlx", *sys.argv[1:], "--prompt", sys.stdin.read()]
main()
"""


def is_ltx25_model(model_name: str | None) -> bool:
    """Return whether a model identifier explicitly selects LTX-2.5."""
    if not model_name:
        return False
    normalized = model_name.casefold().replace("_", "-")
    return "ltx-2.5" in normalized or "ltx25" in normalized


def resolve_ltx25_runtime() -> str | None:
    """Resolve the CLI only when its checkout is at the audited revision."""
    override = os.environ.get("RAPID_MLX_LTX25_RUNTIME", "").strip()
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            absolute = str(candidate.absolute())
            return (
                absolute
                if _runtime_revision(absolute) == LTX25_RUNTIME_COMMIT
                else None
            )
        return None
    executable = shutil.which("ltx-2-mlx")
    if executable is None:
        return None
    absolute = str(Path(executable).absolute())
    return absolute if _runtime_revision(absolute) == LTX25_RUNTIME_COMMIT else None


def _runtime_revision(executable: str) -> str | None:
    """Verify the documented workspace points at the pinned Git revision."""
    path = Path(executable)
    try:
        repository = path.parents[2]
    except IndexError:
        return None
    if not (repository / ".git").exists():
        return None
    if path.absolute() != (repository / ".venv" / "bin" / "ltx-2-mlx").absolute():
        return None
    try:

        def git(*arguments: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                ["git", "-C", str(repository), *arguments],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )

        result = git("rev-parse", "HEAD")
        git("ls-files", "--error-unmatch", "uv.lock")
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def _materialize_runtime(repository: Path, destination: Path) -> None:
    """Extract only tracked files from the audited commit into a fresh tree."""
    try:
        archive = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "archive",
                "--format=tar",
                LTX25_RUNTIME_COMMIT,
            ],
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
        root = destination.resolve()
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
            members = source.getmembers()
            for member in members:
                target = (destination / member.name).resolve()
                if not target.is_relative_to(root) or not (
                    member.isfile() or member.isdir()
                ):
                    raise LTX25BackendError(
                        "The pinned LTX-2.5 source archive contains an unsafe entry."
                    )
            source.extractall(destination, members=members)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise LTX25BackendError(
            "The pinned LTX-2.5 source snapshot could not be materialized."
        ) from exc


def _generation_timeout_seconds() -> int:
    raw = os.environ.get("RAPID_MLX_LTX25_TIMEOUT_SEC", "").strip()
    if not raw:
        return _DEFAULT_TIMEOUT_SECONDS
    try:
        value = int(raw)
    except ValueError as exc:
        raise LTX25BackendError(
            "The LTX-2.5 timeout must be an integer number of seconds."
        ) from exc
    if value < 60:
        raise LTX25BackendError("The LTX-2.5 timeout must be at least 60 seconds.")
    return value


class LTX25BackendError(RuntimeError):
    """Safe, public-facing error from the LTX-2.5 backend."""


class LTX25VideoEngine:
    """Run LTX-2.5 through its native MLX command-line runtime."""

    native_fps = 24

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._process_lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._stopping = False

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            try:
                process.wait(timeout=_TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired as exc:
                raise LTX25BackendError(
                    "The LTX-2.5 runtime process group could not be reaped."
                ) from exc

    def stop(self) -> None:
        """Stop an active external generation during bounded shutdown."""
        with self._process_lock:
            self._stopping = True
            process = self._process
        if process is not None:
            self._terminate_process(process)

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
        repository = Path(executable).parents[2]
        uv = shutil.which("uv")
        if uv is None:
            raise LTX25BackendError(
                "LTX-2.5 support requires uv to build its pinned isolated runtime."
            )
        timeout = _generation_timeout_seconds()
        snapshot = tempfile.TemporaryDirectory(prefix="rapidmlx-ltx25-runtime-")
        try:
            snapshot_path = Path(snapshot.name)
            _materialize_runtime(repository, snapshot_path)
        except Exception:
            snapshot.cleanup()
            raise

        command = [
            str(Path(uv).absolute()),
            "run",
            "--isolated",
            "--frozen",
            "--project",
            str(snapshot_path),
            "python",
            "-c",
            _STDIN_PROMPT_RUNNER,
            "generate",
            "--model",
            self.model_name,
            "--distilled",
            "--low-ram",
            "--quiet",
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
        process: subprocess.Popen[str] | None = None
        try:
            # Prompts may contain private user data. Keep them out of argv and
            # local process listings by feeding the isolated runtime over stdin.
            with self._process_lock:
                if self._stopping:
                    raise LTX25BackendError(
                        "LTX-2.5 generation cannot start while the server is stopping."
                    )
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    start_new_session=True,
                )
                self._process = process
            process.communicate(input=prompt, timeout=timeout)
            if process.returncode:
                raise LTX25BackendError(
                    f"LTX-2.5 runtime exited with code {process.returncode}; "
                    "runtime output is not retained because it may contain request data."
                )
        except subprocess.TimeoutExpired as exc:
            if process is not None:
                self._terminate_process(process)
            raise LTX25BackendError(
                "LTX-2.5 generation exceeded its configured time limit."
            ) from exc
        except OSError as exc:
            raise LTX25BackendError(
                "LTX-2.5 generation could not start its isolated runtime."
            ) from exc
        finally:
            with self._process_lock:
                if self._process is process:
                    self._process = None
            snapshot.cleanup()
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise LTX25BackendError(
                "LTX-2.5 generation completed without an MP4 output."
            )
