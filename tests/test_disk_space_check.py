# SPDX-License-Identifier: Apache-2.0
"""Tests for the pre-flight disk-space check in cli._check_disk_space.

The check must: hard-fail when disk is provably insufficient, return
silently when it can't determine size or the model is already cached,
and respect HF_HOME via huggingface_hub.constants.HF_HUB_CACHE.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from vllm_mlx.cli import _check_disk_space


def _make_info(file_sizes_bytes: list[int]) -> SimpleNamespace:
    """Build a fake huggingface_hub.ModelInfo with sibling file sizes."""
    siblings = [
        SimpleNamespace(size=sz, rfilename=f"file-{index}.safetensors")
        for index, sz in enumerate(file_sizes_bytes)
    ]
    return SimpleNamespace(siblings=siblings, safetensors=None)


def _fake_statvfs(free_bytes: int):
    """Build a fake os.statvfs result with the requested free space."""
    return SimpleNamespace(f_bavail=free_bytes // 4096, f_frsize=4096)


class TestDiskSpaceCheck:
    def test_aborts_when_model_too_large(self):
        """141 GB model on 8.8 GB disk — the bug we're fixing."""
        info = _make_info([int(141 * 1024**3)])
        with (
            patch("huggingface_hub.try_to_load_from_cache", return_value=None),
            patch("huggingface_hub.model_info", return_value=info),
            patch("os.statvfs", return_value=_fake_statvfs(int(8.8 * 1024**3))),
        ):
            with pytest.raises(SystemExit) as exc:
                _check_disk_space("mlx-community/DeepSeek-V4-Flash-4bit")
            assert exc.value.code == 1

    def test_passes_when_disk_has_room(self):
        info = _make_info([int(2 * 1024**3)])  # 2 GB model
        with (
            patch("huggingface_hub.try_to_load_from_cache", return_value=None),
            patch("huggingface_hub.model_info", return_value=info),
            patch("os.statvfs", return_value=_fake_statvfs(int(50 * 1024**3))),
        ):
            # Should not raise.
            _check_disk_space("mlx-community/Qwen3-0.6B-8bit")

    def test_force_skips_abort(self):
        """With --force-disk-check (force=True), insufficient disk warns
        but does not abort."""
        info = _make_info([int(141 * 1024**3)])
        with (
            patch("huggingface_hub.try_to_load_from_cache", return_value=None),
            patch("huggingface_hub.model_info", return_value=info),
            patch("os.statvfs", return_value=_fake_statvfs(int(8.8 * 1024**3))),
        ):
            # Should not raise.
            _check_disk_space("mlx-community/DeepSeek-V4-Flash-4bit", force=True)

    def test_returns_silently_when_model_size_unknown(self):
        """If HF doesn't return file sizes (gated repo, weird config),
        we can't math the disk requirement — skip rather than guess."""
        info = _make_info([])  # no siblings
        with (
            patch("huggingface_hub.try_to_load_from_cache", return_value=None),
            patch("huggingface_hub.model_info", return_value=info),
        ):
            _check_disk_space("mlx-community/Some-Model")  # no raise

    def test_returns_silently_when_hf_api_fails(self):
        """Network errors / 404s during the size query must not block
        startup — the loader has its own 404 handler."""
        with (
            patch("huggingface_hub.try_to_load_from_cache", return_value=None),
            patch(
                "huggingface_hub.model_info",
                side_effect=ConnectionError("offline"),
            ),
        ):
            _check_disk_space("mlx-community/Some-Model")  # no raise

    def test_skips_local_path(self, tmp_path):
        """Local model directories don't need disk checking."""
        local = tmp_path / "my-model"
        local.mkdir()
        # Should not raise even without mocking model_info — must short-circuit.
        _check_disk_space(str(local))

    def test_skips_already_cached(self):
        """If every remote file is cached, skip the size check entirely."""
        info = _make_info([int(5 * 1024**3)])
        disk_probe = Mock(return_value=_fake_statvfs(100 * 1024**3))
        with (
            patch(
                "huggingface_hub.try_to_load_from_cache",
                return_value="/fake/path/file-0.safetensors",
            ),
            patch("huggingface_hub.model_info", return_value=info),
            patch(
                "os.path.exists",
                side_effect=lambda path: str(path).startswith("/fake/path/"),
            ),
            patch("os.statvfs", disk_probe),
        ):
            _check_disk_space("mlx-community/Some-Model")
        disk_probe.assert_not_called()

    def test_fully_cached_nested_mflux_layout_needs_no_free_space(self):
        """#1772: mflux has no root config, but all nested files are cached."""
        repo = "filipstrand/Z-Image-Turbo-mflux-4bit"
        siblings = [
            SimpleNamespace(
                size=int(4 * 1024**3),
                rfilename="transformer/model-00001-of-00002.safetensors",
            ),
            SimpleNamespace(
                size=int(1.5 * 1024**3),
                rfilename="text_encoder/model.safetensors",
            ),
            SimpleNamespace(size=4096, rfilename="vae/config.json"),
        ]
        info = SimpleNamespace(siblings=siblings, safetensors=None)
        disk_probe = Mock(return_value=_fake_statvfs(100 * 1024**3))

        cached_files = {sibling.rfilename for sibling in siblings}

        def cached(_repo, filename):
            if filename in cached_files:
                return f"/fake/snapshot/{filename}"
            return None

        with (
            patch("huggingface_hub.try_to_load_from_cache", side_effect=cached),
            patch("huggingface_hub.model_info", return_value=info),
            patch(
                "os.path.exists",
                side_effect=lambda path: str(path).startswith("/fake/snapshot/"),
            ),
            patch("os.statvfs", disk_probe),
        ):
            _check_disk_space(repo)
        disk_probe.assert_not_called()

    def test_partial_nested_cache_checks_only_missing_bytes(self, capsys):
        """A partial mflux cache must retain the gate for uncached payload."""
        repo = "filipstrand/Z-Image-Turbo-mflux-4bit"
        gib = 1024**3
        info = SimpleNamespace(
            siblings=[
                SimpleNamespace(
                    size=int(4 * gib),
                    rfilename="transformer/model.safetensors",
                ),
                SimpleNamespace(
                    size=int(1.5 * gib),
                    rfilename="text_encoder/model.safetensors",
                ),
            ],
            safetensors=None,
        )

        def cached(_repo, filename):
            if filename.startswith("transformer/"):
                return f"/fake/snapshot/{filename}"
            return None

        with (
            patch("huggingface_hub.try_to_load_from_cache", side_effect=cached),
            patch("huggingface_hub.model_info", return_value=info),
            patch(
                "os.path.exists",
                side_effect=lambda path: str(path).startswith("/fake/snapshot/"),
            ),
            patch("os.statvfs", return_value=_fake_statvfs(int(1.0 * gib))),
        ):
            with pytest.raises(SystemExit) as exc:
                _check_disk_space(repo)
            assert exc.value.code == 1
        assert "Download size:     1.5 GB" in capsys.readouterr().out

    def test_config_only_text_cache_does_not_bypass_missing_weights(self):
        """A cached config is metadata, not proof that model weights exist."""
        gib = 1024**3
        info = SimpleNamespace(
            siblings=[
                SimpleNamespace(size=4096, rfilename="config.json"),
                SimpleNamespace(size=int(5 * gib), rfilename="model.safetensors"),
            ],
            safetensors=None,
        )

        def cached(_repo, filename):
            return "/fake/snapshot/config.json" if filename == "config.json" else None

        with (
            patch("huggingface_hub.try_to_load_from_cache", side_effect=cached),
            patch("huggingface_hub.model_info", return_value=info),
            patch(
                "os.path.exists",
                side_effect=lambda path: str(path) == "/fake/snapshot/config.json",
            ),
            patch("os.statvfs", return_value=_fake_statvfs(int(1.0 * gib))),
        ):
            with pytest.raises(SystemExit) as exc:
                _check_disk_space("mlx-community/Partial-Model")
            assert exc.value.code == 1

    def test_uses_hf_hub_cache_for_statvfs(self):
        """The probe must use HF_HUB_CACHE (respects HF_HOME), not the
        hard-coded ~/.cache/huggingface.

        Asserts strictly on /Volumes — the HOME fallback only kicks in when
        the walk-up loop runs out of ancestors, which would mask a
        regression that drops the HF_HUB_CACHE pivot entirely.
        """
        info = _make_info([int(2 * 1024**3)])
        seen_paths = []

        def capture_statvfs(path):
            seen_paths.append(path)
            return _fake_statvfs(int(50 * 1024**3))

        with (
            patch("huggingface_hub.try_to_load_from_cache", return_value=None),
            patch("huggingface_hub.model_info", return_value=info),
            patch(
                "huggingface_hub.constants.HF_HUB_CACHE",
                "/Volumes/external/hf",
            ),
            patch("os.statvfs", side_effect=capture_statvfs),
            patch(
                "os.path.exists",
                side_effect=lambda p: p == "/Volumes/external" or p.startswith("/"),
            ),
        ):
            _check_disk_space("mlx-community/Qwen3-0.6B-8bit")

        assert seen_paths, "statvfs was never called"
        assert any("/Volumes" in p for p in seen_paths), (
            f"statvfs probe path {seen_paths!r} didn't include HF_HUB_CACHE; "
            "the HF_HOME pivot regressed and the probe fell back to $HOME."
        )
