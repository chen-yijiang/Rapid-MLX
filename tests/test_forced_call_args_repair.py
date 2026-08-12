# SPDX-License-Identifier: Apache-2.0
"""Forced ``tool_choice`` calls must carry OBJECT arguments on the wire.

Dogfood 2026-08-12 (post #1866/#1874/#1878 merge, pre-existing on main):
the hermes forced-choice prefix hands the model ``..."arguments": ``
mid-envelope and Qwen3.5-35B-A3B-8bit at temp 0 continues with a bare
``1`` — the parser surfaces ``arguments="1"``, and every OpenAI client
``json.loads``\\ es a non-dict and breaks. The repair mirrors the #571
synthesis fallback (recover from raw text, else ``"{}"``) and holds the
result to the #1256 schema gate.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from vllm_mlx.routes.chat import _repair_forced_call_arguments


def _call(arguments):
    return SimpleNamespace(
        id="call_x",
        type="function",
        function=SimpleNamespace(name="release_probe", arguments=arguments),
    )


def _tool(name: str = "release_probe", required: list[str] | None = None):
    schema: dict[str, Any] = {"type": "object", "properties": {}}
    if required is not None:
        schema["required"] = required
    return SimpleNamespace(
        type="function", function={"name": name, "parameters": schema}
    )


def test_int_arguments_repaired_to_empty_object():
    tc = _call("1")
    err = _repair_forced_call_arguments([tc], "", "release_probe", [_tool()])
    assert err is None
    assert tc.function.arguments == "{}"


def test_valid_object_arguments_untouched():
    tc = _call('{"note": "ok"}')
    err = _repair_forced_call_arguments([tc], "", "release_probe", [_tool()])
    assert err is None
    assert tc.function.arguments == '{"note": "ok"}'


def test_unparseable_arguments_repaired():
    tc = _call("not json at all {")
    err = _repair_forced_call_arguments([tc], "", "release_probe", [_tool()])
    assert err is None
    assert tc.function.arguments == "{}"


def test_recovery_from_raw_text_preferred_over_empty():
    tc = _call("1")
    raw = '<tool_call>{"name": "release_probe", "arguments": {"note": "hi"}}'
    err = _repair_forced_call_arguments([tc], raw, "release_probe", [_tool()])
    assert err is None
    assert "note" in tc.function.arguments


def test_schema_required_violation_surfaces_error():
    """The repair must not fabricate a passing call when the tool's
    schema requires properties the repaired arguments cannot provide —
    same #1256 contract as the synthesis fallback."""
    tc = _call("1")
    err = _repair_forced_call_arguments(
        [tc], "", "release_probe", [_tool(required=["level"])]
    )
    assert err is not None
    assert "level" in err or "required" in err.lower()
