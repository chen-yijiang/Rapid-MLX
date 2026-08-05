# SPDX-License-Identifier: Apache-2.0
"""Perf-gate floor resolution — the reviewed-floor layer that lets
``evals/perf_gate.py`` actually BLOCK a release when decode throughput
regresses (previously the gate was never wired into the M3 gauntlet, so its
floor mechanism was unreachable; see docs/development/releasing.md G8b).

Hermetic: no server, no model — every case exercises the pure resolution /
file-parsing helpers against ``tmp_path`` fixtures and monkeypatched env.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import perf_gate
from evals.perf_gate import FloorsFileError, _load_floor_from_file, resolve_floor

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_floors(tmp_path: Path, floors: object) -> str:
    p = tmp_path / "perf_floors.json"
    p.write_text(json.dumps({"schema": 1, "floors": floors}))
    return str(p)


# --------------------------- resolve_floor precedence ---------------------


def test_cli_beats_env_and_file(tmp_path: Path) -> None:
    floors_file = _write_floors(tmp_path, {"m": 10.0})
    got = resolve_floor(
        cli_min_tps=99.0, env_min_tps=50.0, floors_file=floors_file, alias="m"
    )
    assert got == 99.0


def test_env_beats_file(tmp_path: Path) -> None:
    floors_file = _write_floors(tmp_path, {"m": 10.0})
    got = resolve_floor(
        cli_min_tps=None, env_min_tps=50.0, floors_file=floors_file, alias="m"
    )
    assert got == 50.0


def test_file_used_when_no_cli_or_env(tmp_path: Path) -> None:
    floors_file = _write_floors(tmp_path, {"m": 12.5})
    got = resolve_floor(
        cli_min_tps=None, env_min_tps=None, floors_file=floors_file, alias="m"
    )
    assert got == 12.5


def test_advisory_when_nothing_supplies_a_floor() -> None:
    assert (
        resolve_floor(cli_min_tps=None, env_min_tps=None, floors_file=None, alias="m")
        is None
    )


def test_advisory_when_alias_absent_from_file(tmp_path: Path) -> None:
    floors_file = _write_floors(tmp_path, {"other": 10.0})
    assert (
        resolve_floor(
            cli_min_tps=None, env_min_tps=None, floors_file=floors_file, alias="m"
        )
        is None
    )


# --------------------------- _load_floor_from_file ------------------------


def test_load_present_alias(tmp_path: Path) -> None:
    floors_file = _write_floors(tmp_path, {"qwen3.5-9b-4bit": 29.5})
    assert _load_floor_from_file(floors_file, "qwen3.5-9b-4bit") == 29.5


def test_load_absent_alias_is_none(tmp_path: Path) -> None:
    floors_file = _write_floors(tmp_path, {"a": 1.0})
    assert _load_floor_from_file(floors_file, "b") is None


def test_load_blank_alias_is_none(tmp_path: Path) -> None:
    # No alias to look up → no file floor (caller passes the file
    # unconditionally and lets the alias decide whether it matters).
    floors_file = _write_floors(tmp_path, {"a": 1.0})
    assert _load_floor_from_file(floors_file, "") is None


def test_load_empty_floors_object(tmp_path: Path) -> None:
    floors_file = _write_floors(tmp_path, {})
    assert _load_floor_from_file(floors_file, "a") is None


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FloorsFileError, match="cannot read"):
        _load_floor_from_file(str(tmp_path / "nope.json"), "a")


def test_load_malformed_json_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(FloorsFileError, match="not valid JSON"):
        _load_floor_from_file(str(p), "a")


def test_load_non_object_top_level_raises(tmp_path: Path) -> None:
    p = tmp_path / "arr.json"
    p.write_text("[1, 2, 3]")
    with pytest.raises(FloorsFileError, match="must be a JSON object"):
        _load_floor_from_file(str(p), "a")


def test_load_floors_not_object_raises(tmp_path: Path) -> None:
    p = tmp_path / "f.json"
    p.write_text(json.dumps({"schema": 1, "floors": [1, 2]}))
    with pytest.raises(FloorsFileError, match="'floors' must be an object"):
        _load_floor_from_file(str(p), "a")


def test_load_non_numeric_value_raises(tmp_path: Path) -> None:
    floors_file = _write_floors(tmp_path, {"a": "fast"})
    with pytest.raises(FloorsFileError, match="must be a number"):
        _load_floor_from_file(floors_file, "a")


def test_load_bool_value_raises(tmp_path: Path) -> None:
    # bool is an int subclass — a stray `true` must not be read as 1.0.
    floors_file = _write_floors(tmp_path, {"a": True})
    with pytest.raises(FloorsFileError, match="must be a number"):
        _load_floor_from_file(floors_file, "a")


def test_load_int_value_coerced_to_float(tmp_path: Path) -> None:
    floors_file = _write_floors(tmp_path, {"a": 30})
    got = _load_floor_from_file(floors_file, "a")
    assert got == 30.0
    assert isinstance(got, float)


# --------------------------- committed registry ---------------------------


def test_committed_perf_floors_is_valid() -> None:
    """The committed harness/perf_floors.json must be parseable and shaped as
    the gate expects — a broken commit here would fail every release run at
    the G8b gate (return code 2)."""
    path = REPO_ROOT / "harness" / "perf_floors.json"
    data = json.loads(path.read_text())
    assert isinstance(data, dict)
    assert isinstance(data.get("floors"), dict)
    # Every committed floor must be a real, enforceable number (finite > 0) so
    # the gate never trips its own "floor that cannot enforce anything" guard.
    for alias, floor in data["floors"].items():
        assert isinstance(alias, str) and alias
        assert isinstance(floor, (int, float)) and not isinstance(floor, bool)
        assert floor > 0
    # And the loader agrees with a straight parse for any committed alias.
    for alias in data["floors"]:
        assert _load_floor_from_file(str(path), alias) == float(data["floors"][alias])


def test_module_exposes_resolution_api() -> None:
    # Pins the public surface the gauntlet + tests depend on.
    assert callable(perf_gate.resolve_floor)
    assert callable(perf_gate._load_floor_from_file)
    assert issubclass(perf_gate.FloorsFileError, RuntimeError)
