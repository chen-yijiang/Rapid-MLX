from __future__ import annotations

import collections
import subprocess

import pytest

from scripts import check_mypy_new_errors
from scripts.check_mypy_new_errors import ErrorSignature, added_errors, parse_errors


def test_parse_errors_ignores_locations_notes_and_summary() -> None:
    output = """\
vllm_mlx/a.py:10:5: error: Bad assignment  [assignment]
vllm_mlx/a.py:99: error: Bad assignment  [assignment]
vllm_mlx/a.py:10: note: This is context
Found 2 errors in 1 file (checked 2 source files)
"""

    assert parse_errors(output) == collections.Counter(
        {ErrorSignature("vllm_mlx/a.py", "assignment", "Bad assignment"): 2}
    )


def test_parse_errors_retains_unclassified_errors() -> None:
    assert parse_errors("./vllm_mlx/a.py:7: error: Broken") == collections.Counter(
        {ErrorSignature("vllm_mlx/a.py", "unclassified", "Broken"): 1}
    )


def test_moved_existing_error_does_not_count_as_new() -> None:
    before = parse_errors("vllm_mlx/a.py:1: error: Broken  [assignment]")
    after = parse_errors("vllm_mlx/a.py:200: error: Broken  [assignment]")

    assert not added_errors(before, after)


def test_additional_copy_of_existing_error_is_new() -> None:
    before = parse_errors("vllm_mlx/a.py:1: error: Broken  [assignment]")
    after = parse_errors(
        "vllm_mlx/a.py:20: error: Broken  [assignment]\n"
        "vllm_mlx/a.py:30: error: Broken  [assignment]"
    )

    assert added_errors(before, after) == collections.Counter(
        {ErrorSignature("vllm_mlx/a.py", "assignment", "Broken"): 1}
    )


def test_changed_error_message_is_new_and_old_one_is_resolved() -> None:
    before = parse_errors("vllm_mlx/a.py:1: error: Old  [assignment]")
    after = parse_errors("vllm_mlx/a.py:1: error: New  [assignment]")

    assert added_errors(before, after) == collections.Counter(
        {ErrorSignature("vllm_mlx/a.py", "assignment", "New"): 1}
    )


def test_unparseable_mypy_failure_fails_closed(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        check_mypy_new_errors,
        "_run",
        lambda command, cwd: subprocess.CompletedProcess(
            command, returncode=1, stdout="unexpected output format"
        ),
    )

    with pytest.raises(RuntimeError, match="no parseable errors"):
        check_mypy_new_errors._mypy_errors(tmp_path, ["mypy"])
