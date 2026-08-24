#!/usr/bin/env python3
"""Reject mypy errors introduced by a candidate relative to its base commit.

The repository has existing type-checking debt. This gate runs the same pinned
mypy toolchain against the candidate and its base, removes unstable line/column
numbers, and compares the remaining error multiset. Existing errors may move
within a file without blocking a PR; an additional or changed error does block.
"""

from __future__ import annotations

import argparse
import collections
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

_ERROR_RE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+)(?::(?P<column>\d+))?: error: "
    r"(?P<message>.*?)(?:  \[(?P<code>[^]]+)\])?$"
)
_ACCEPTED_MYPY_EXITS = {0, 1}


@dataclass(frozen=True, order=True)
class ErrorSignature:
    path: str
    code: str
    message: str

    def render(self, count: int = 1) -> str:
        suffix = f" (x{count})" if count > 1 else ""
        return f"{self.path}: [{self.code}] {self.message}{suffix}"


def parse_errors(output: str) -> collections.Counter[ErrorSignature]:
    """Return stable error signatures, intentionally ignoring notes/locations."""

    errors: collections.Counter[ErrorSignature] = collections.Counter()
    for raw_line in output.splitlines():
        match = _ERROR_RE.match(raw_line.strip())
        if match is None:
            continue
        errors[
            ErrorSignature(
                path=match.group("path").removeprefix("./"),
                code=match.group("code") or "unclassified",
                message=match.group("message"),
            )
        ] += 1
    return errors


def added_errors(
    base: collections.Counter[ErrorSignature],
    candidate: collections.Counter[ErrorSignature],
) -> collections.Counter[ErrorSignature]:
    return candidate - base


def _run(command: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _mypy_errors(
    root: Path, command: Sequence[str]
) -> collections.Counter[ErrorSignature]:
    result = _run(command, cwd=root)
    if result.returncode not in _ACCEPTED_MYPY_EXITS:
        print(result.stdout, file=sys.stderr)
        raise RuntimeError(f"mypy failed operationally with exit {result.returncode}")
    errors = parse_errors(result.stdout)
    if result.returncode == 1 and not errors:
        print(result.stdout, file=sys.stderr)
        raise RuntimeError("mypy reported failure but produced no parseable errors")
    return errors


def _resolve_commit(repo: Path, revision: str) -> str:
    result = _run(["git", "rev-parse", "--verify", f"{revision}^{{commit}}"], cwd=repo)
    if result.returncode != 0:
        raise RuntimeError(f"base revision is unavailable: {revision}")
    return result.stdout.strip()


def _comparison(
    repo: Path,
    base_revision: str,
    mypy_command: Sequence[str],
) -> tuple[collections.Counter[ErrorSignature], collections.Counter[ErrorSignature]]:
    base_commit = _resolve_commit(repo, base_revision)
    with tempfile.TemporaryDirectory(prefix="rapid-mypy-base-") as directory:
        base_root = Path(directory)
        add = _run(
            ["git", "worktree", "add", "--detach", str(base_root), base_commit],
            cwd=repo,
        )
        if add.returncode != 0:
            raise RuntimeError(f"could not create base worktree:\n{add.stdout}")
        try:
            base = _mypy_errors(base_root, mypy_command)
        finally:
            # Best effort here; TemporaryDirectory still removes the files, and
            # prune clears metadata if git reports an already-removed worktree.
            _run(["git", "worktree", "remove", "--force", str(base_root)], cwd=repo)
            _run(["git", "worktree", "prune"], cwd=repo)

    candidate = _mypy_errors(repo, mypy_command)
    return base, candidate


def _default_mypy_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "mypy",
        "vllm_mlx/",
        "--ignore-missing-imports",
        "--no-error-summary",
        "--show-error-codes",
        "--no-pretty",
    ]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Git revision to compare against")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo = args.repo.resolve()
    if shutil.which("git") is None:
        parser.error("git is required")

    try:
        base, candidate = _comparison(repo, args.base, _default_mypy_command())
    except RuntimeError as error:
        print(f"mypy ratchet could not produce a trustworthy comparison: {error}")
        return 2

    added = added_errors(base, candidate)
    resolved = base - candidate
    print(
        f"mypy ratchet: base={sum(base.values())}, "
        f"candidate={sum(candidate.values())}, "
        f"new={sum(added.values())}, resolved={sum(resolved.values())}"
    )
    if not added:
        return 0

    print("New mypy errors:")
    for signature, count in sorted(added.items()):
        print(f"  - {signature.render(count)}")
    print("Fix the new errors; unrelated historical mypy debt remains non-blocking.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
