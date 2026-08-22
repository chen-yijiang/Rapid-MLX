"""Nemotron 3 contract tests for the shared Qwen3 XML parser."""

import json

from vllm_mlx.tool_parsers import NemotronToolParser, ToolParserManager
from vllm_mlx.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser


def _request(*names: str) -> dict:
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    },
                },
            }
            for name in names
        ]
    }


def _call(name: str, value: str) -> str:
    return (
        f"<tool_call><function={name}><parameter=value>{value}</parameter>"
        "</function></tool_call>"
    )


def test_nemotron_names_reuse_qwen3_xml_parser():
    assert issubclass(NemotronToolParser, Qwen3CoderToolParser)
    assert ToolParserManager.get_tool_parser("nemotron") is NemotronToolParser
    assert ToolParserManager.get_tool_parser("nemotron3") is NemotronToolParser


def test_canonical_parameter_wrapper():
    result = NemotronToolParser().extract_tool_calls(
        _call("lookup", "Paris"), _request("lookup")
    )

    assert result.tools_called
    assert [call["name"] for call in result.tool_calls] == ["lookup"]
    assert json.loads(result.tool_calls[0]["arguments"]) == {"value": "Paris"}


def test_multiple_wrapped_calls_preserve_wire_order():
    result = NemotronToolParser().extract_tool_calls(
        _call("first", "a") + _call("second", "b"),
        _request("first", "second"),
    )

    assert [call["name"] for call in result.tool_calls] == ["first", "second"]


def test_undeclared_tool_is_content_not_execution():
    text = _call("run_shell", "id")
    result = NemotronToolParser().extract_tool_calls(text, _request("lookup"))

    assert not result.tools_called
    assert result.content == text


def test_streaming_emits_declared_call():
    parser = NemotronToolParser()
    previous = ""
    emitted = []
    for chunk in (
        "<tool_call>",
        "<function=lookup>",
        "<parameter=value>Paris</parameter>",
        "</function>",
        "</tool_call>",
    ):
        current = previous + chunk
        delta = parser.extract_tool_calls_streaming(
            previous,
            current,
            chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=_request("lookup"),
        )
        emitted.extend((delta or {}).get("tool_calls") or [])
        previous = current

    assert len(emitted) == 1
    assert emitted[0]["function"]["name"] == "lookup"
