# SPDX-License-Identifier: Apache-2.0
"""Incremental-delta streaming for ``Qwen3CoderToolParser``.

Closes the bug surfaced in #479 (kenizhou): the parser used to buffer the
entire string-typed parameter value until ``</parameter>`` arrived, then
dump it as a single ``function.arguments`` delta. CopilotKit / LangChain
inspectors that render argument values live stalled for multiple seconds
on long summaries / key-points.

The three tests below pin the new behavior:

* ``test_long_string_param_emits_multiple_deltas`` — granularity guard.
* ``test_close_tag_never_leaks_into_emitted_fragment`` — the
  ``len("</parameter>")`` tail-buffer guarantees no half-tag escapes
  into a JSON fragment.
* ``test_streaming_json_matches_non_streaming`` — concatenating all
  streamed argument fragments and parsing the result yields the same
  Python object as ``extract_tool_calls`` returns for the full text.
"""

from __future__ import annotations

import json

import pytest

from vllm_mlx.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

_LONG_SUMMARY = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do "
    "eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut "
    "enim ad minim veniam, quis nostrud exercitation ullamco laboris "
    "nisi ut aliquip ex ea commodo consequat."
)


def _request_with_tool(name: str, properties: dict) -> dict:
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {"type": "object", "properties": properties},
                },
            }
        ]
    }


def _feed(parser: Qwen3CoderToolParser, chunks: list[str], request: dict | None):
    """Stream ``chunks`` through the parser; return non-None delta dicts."""
    parser.reset()
    deltas: list[dict] = []
    previous = ""
    for chunk in chunks:
        if not chunk:
            continue
        current = previous + chunk
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=chunk,
            request=request,
        )
        if delta is not None:
            deltas.append(delta)
        previous = current
    return deltas


def _argument_fragments(deltas: list[dict]) -> list[str]:
    """Flatten ``function.arguments`` strings out of streamed tool_call deltas."""
    out: list[str] = []
    for d in deltas:
        for tc in d.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if args:
                out.append(args)
    return out


def test_long_string_param_emits_multiple_deltas():
    """For a long string param, at least 2 ``function.arguments`` deltas with
    non-empty content must arrive BEFORE the ``</parameter>`` close.

    Without #479's incremental emit the parser only produced a single
    delta containing the whole value once the close tag arrived.
    """
    parser = Qwen3CoderToolParser(tokenizer=None)
    request = _request_with_tool(
        "summarize",
        {"summary": {"type": "string"}},
    )

    head = [
        "<tool_call>\n",
        "<function=summarize>\n",
        "<parameter=summary>\n",
    ]
    # Split the long summary into 32-char body chunks so the in-flight
    # branch fires several times before the close tag arrives.
    value_chunks = [_LONG_SUMMARY[i : i + 32] for i in range(0, len(_LONG_SUMMARY), 32)]
    pre_close_chunks = head + value_chunks

    parser.reset()
    deltas_before_close: list[dict] = []
    previous = ""
    for chunk in pre_close_chunks:
        current = previous + chunk
        d = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=chunk,
            request=request,
        )
        if d is not None:
            deltas_before_close.append(d)
        previous = current

    arg_fragments = [
        f
        for f in _argument_fragments(deltas_before_close)
        # Skip the structural openers ("{", "") so we only count real
        # value-bearing fragments — those are what the UI renders live.
        if f not in ("{", "")
    ]

    assert len(arg_fragments) >= 2, (
        "#479 regression: long string params should stream incrementally; "
        f"got {len(arg_fragments)} value-bearing argument deltas before "
        f"</parameter>. fragments={arg_fragments!r}"
    )


def test_close_tag_never_leaks_into_emitted_fragment():
    """Feeding ``...value</par`` then ``ameter>`` across two chunks must
    never produce a ``function.arguments`` fragment containing the literal
    substring ``</par`` or ``</parameter>``.

    Guards the tail-buffer: the parser must hold back the last
    ``len("</parameter>")`` chars of unread tail so a partial close tag
    straddling a chunk boundary can't be flushed prematurely.
    """
    parser = Qwen3CoderToolParser(tokenizer=None)
    request = _request_with_tool("echo", {"value": {"type": "string"}})

    # Long enough that incremental emission will fire before the close.
    value = "A" * 200
    # Split mid-close-tag (``</par`` | ``ameter>``) to exercise the
    # tail-buffer guard at a chunk boundary.
    chunks = [
        "<tool_call>\n",
        "<function=echo>\n",
        "<parameter=value>\n",
        value[:80],
        value[80:160],
        value[160:] + "\n</par",
        "ameter>\n",
        "</function>\n",
        "</tool_call>",
    ]

    deltas = _feed(parser, chunks, request)
    fragments = _argument_fragments(deltas)
    for frag in fragments:
        assert "</par" not in frag, f"close-tag leaked into streamed fragment: {frag!r}"
        assert "</parameter>" not in frag, (
            f"close-tag leaked into streamed fragment: {frag!r}"
        )

    # Belt + braces: concatenate everything, parse the JSON, and assert
    # the decoded value is exactly the original — catches escaped /
    # split-across-fragments leaks that a substring scan alone would miss.
    combined = "".join(fragments)
    decoded = json.loads(combined)
    assert decoded == {"value": value}, (
        f"streamed args decoded to {decoded!r}, expected {{'value': {value!r}}}"
    )


def test_streaming_json_matches_non_streaming():
    """Concatenating all streamed ``function.arguments`` fragments and
    ``json.loads``-ing the result must equal the arguments dict
    ``extract_tool_calls`` returns for the same complete input.

    Covers mixed param types — string (mid-flight emit), int (buffered
    emit), and a second short string (back-to-back string).
    """
    parser = Qwen3CoderToolParser(tokenizer=None)
    request = _request_with_tool(
        "report",
        {
            "summary": {"type": "string"},
            "score": {"type": "integer"},
            "owner": {"type": "string"},
        },
    )

    full_text = (
        "<tool_call>\n"
        "<function=report>\n"
        f"<parameter=summary>\n{_LONG_SUMMARY}\n</parameter>\n"
        "<parameter=score>\n42\n</parameter>\n"
        "<parameter=owner>\nken\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )

    # Non-streaming reference
    ns_result = parser.extract_tool_calls(full_text, request=request)
    assert ns_result.tools_called, "non-stream extract should detect tool call"
    expected_args = json.loads(ns_result.tool_calls[0]["arguments"])

    # Streaming run with small body chunks so the in-flight emit fires
    # repeatedly and back-to-back string params exercise param-separator
    # logic in ``_close_string_increment``.
    chunks = [
        "<tool_call>\n",
        "<function=report>\n",
        "<parameter=summary>\n",
    ]
    summary_body = _LONG_SUMMARY + "\n"
    chunks.extend(summary_body[i : i + 24] for i in range(0, len(summary_body), 24))
    chunks.extend(
        [
            "</parameter>\n",
            "<parameter=score>\n42\n</parameter>\n",
            "<parameter=owner>\nken\n</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]
    )
    deltas = _feed(parser, chunks, request)
    fragments = _argument_fragments(deltas)
    combined = "".join(fragments)

    streamed_args = json.loads(combined)
    assert streamed_args == expected_args, (
        f"streamed JSON does not match non-streamed JSON.\n"
        f"  combined raw    = {combined!r}\n"
        f"  streamed parsed = {streamed_args!r}\n"
        f"  expected        = {expected_args!r}"
    )


@pytest.mark.parametrize(
    "value",
    [
        "text with a literal </parameter> inside",
        "text with a literal </function> inside",
        "has <parameter=fake> and </parameter> both",
        "has <function=fake> and </function> both",
    ],
)
def test_marker_text_inside_string_value_round_trips(value):
    """Wire markers inside a string argument are payload, not boundaries.

    The XML format has no escape syntax, so the parser must wait for a real
    sibling/function boundary and use the final closer before that boundary.
    This is the streaming twin of the release-blocking non-streaming repro.
    """
    parser = Qwen3CoderToolParser(tokenizer=None)
    request = _request_with_tool("note_write", {"body": {"type": "string"}})
    chunks = [
        "<tool_call>\n",
        "<function=note_write>\n",
        "<parameter=body>\n",
        value[: max(1, len(value) // 2)],
        value[max(1, len(value) // 2) :] + "\n",
        "</parameter>\n",
        "</function>\n",
        "</tool_call>",
    ]

    deltas = _feed(parser, chunks, request)
    combined = "".join(_argument_fragments(deltas))
    assert json.loads(combined) == {"body": value}, (combined, deltas)


def test_marker_text_round_trips_when_complete_call_arrives_in_one_delta():
    """The header fast path must use the same last-closer rule."""
    value = "text with a literal </function> inside"
    wire = (
        "<tool_call>\n<function=note_write>\n<parameter=body>\n"
        f"{value}\n</parameter>\n</function>\n</tool_call>"
    )
    request = _request_with_tool("note_write", {"body": {"type": "string"}})
    deltas = _feed(Qwen3CoderToolParser(tokenizer=None), [wire], request)
    assert json.loads("".join(_argument_fragments(deltas))) == {"body": value}


def test_truncated_final_call_is_recovered_after_complete_call():
    """A complete earlier call must not hide max-token recovery for the last."""
    request = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {
                        "type": "object",
                        "properties": {key: {"type": "string"}},
                    },
                },
            }
            for name, key in (("first", "one"), ("second", "two"))
        ]
    }
    text = (
        "<tool_call><function=first><parameter=one>1</parameter>"
        "</function></tool_call>"
        "<tool_call><function=second><parameter=two>2</parameter>"
    )
    result = Qwen3CoderToolParser(tokenizer=None).extract_tool_calls(text, request)
    assert [call["name"] for call in result.tool_calls] == ["first", "second"]
    assert [json.loads(call["arguments"]) for call in result.tool_calls] == [
        {"one": "1"},
        {"two": "2"},
    ]


def test_complete_multi_function_wrapper_in_one_delta_emits_every_call():
    request = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {
                        "type": "object",
                        "properties": {key: {"type": "string"}},
                    },
                },
            }
            for name, key in (("first", "one"), ("second", "two"))
        ]
    }
    wire = (
        "<tool_call><function=first><parameter=one>1</parameter></function>"
        "<function=second><parameter=two>2</parameter></function></tool_call>"
    )
    deltas = _feed(Qwen3CoderToolParser(tokenizer=None), [wire], request)
    calls = deltas[0]["tool_calls"]
    assert [call["function"]["name"] for call in calls] == ["first", "second"]
    assert [json.loads(call["function"]["arguments"]) for call in calls] == [
        {"one": "1"},
        {"two": "2"},
    ]


def test_complete_multi_function_wrapper_in_token_sized_deltas_emits_every_call():
    request = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {
                        "type": "object",
                        "properties": {key: {"type": "string"}},
                    },
                },
            }
            for name, key in (("first", "one"), ("second", "two"))
        ]
    }
    chunks = [
        "<tool_call>",
        "<function=first>",
        "<parameter=one>1</parameter>",
        "</function>",
        "<function=second>",
        "<parameter=two>2</parameter>",
        "</function>",
        "</tool_call>",
    ]
    deltas = _feed(Qwen3CoderToolParser(tokenizer=None), chunks, request)
    calls = [call for delta in deltas for call in delta.get("tool_calls", [])]
    by_index: dict[int, str] = {}
    names: dict[int, str] = {}
    for call in calls:
        index = call["index"]
        function = call["function"]
        if function.get("name"):
            names[index] = function["name"]
        by_index[index] = by_index.get(index, "") + function.get("arguments", "")
    assert names == {0: "first", 1: "second"}
    assert {index: json.loads(arguments) for index, arguments in by_index.items()} == {
        0: {"one": "1"},
        1: {"two": "2"},
    }


def test_separately_wrapped_calls_do_not_leak_second_wrapper_close():
    request = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {
                        "type": "object",
                        "properties": {key: {"type": "string"}},
                    },
                },
            }
            for name, key in (("first", "one"), ("second", "two"))
        ]
    }
    chunks = [
        "<tool_call>",
        "<function=first><parameter=one>1</parameter></function>",
        "</tool_call>",
        "<tool_call>",
        "<function=second><parameter=two>2</parameter></function>",
        "</tool_call>",
    ]
    deltas = _feed(Qwen3CoderToolParser(tokenizer=None), chunks, request)
    assert all("tool_call" not in delta.get("content", "") for delta in deltas)
    calls = [call for delta in deltas for call in delta.get("tool_calls", [])]
    names = {
        call["index"]: call["function"]["name"]
        for call in calls
        if call["function"].get("name")
    }
    arguments: dict[int, str] = {}
    for call in calls:
        index = call["index"]
        arguments[index] = arguments.get(index, "") + call["function"].get(
            "arguments", ""
        )
    assert names == {0: "first", 1: "second"}
    assert {index: json.loads(value) for index, value in arguments.items()} == {
        0: {"one": "1"},
        1: {"two": "2"},
    }


@pytest.mark.parametrize(
    "value_chunks",
    [
        ["short"],
        ["abc", "defghijklmnopqrstuvwxyz"],
    ],
)
def test_missing_final_parameter_close_recovers_valid_stream_json(value_chunks):
    """A structural function close is also the fallback parameter boundary."""
    value = "".join(value_chunks)
    request = _request_with_tool("note_write", {"body": {"type": "string"}})
    chunks = [
        "<tool_call>\n",
        "<function=note_write>\n",
        "<parameter=body>",
        *value_chunks,
        "</function>\n</tool_call>",
    ]
    deltas = _feed(Qwen3CoderToolParser(tokenizer=None), chunks, request)
    assert json.loads("".join(_argument_fragments(deltas))) == {"body": value}


def test_literal_declared_parameter_opener_does_not_fabricate_argument():
    parser = Qwen3CoderToolParser(tokenizer=None)
    request = _request_with_tool(
        "note_write",
        {"body": {"type": "string"}, "other": {"type": "string"}},
    )
    text = (
        "<tool_call><function=note_write>"
        "<parameter=body>text <parameter=other> literal</parameter>"
        "</function></tool_call>"
    )

    result = parser.extract_tool_calls(text, request)

    assert result.tools_called
    assert json.loads(result.tool_calls[0]["arguments"]) == {
        "body": "text <parameter=other> literal"
    }


def test_two_literal_declared_parameter_blocks_do_not_fabricate_arguments():
    request = _request_with_tool(
        "note_write",
        {
            "body": {"type": "string"},
            "other": {"type": "string"},
            "third": {"type": "string"},
        },
    )
    value = (
        "examples <parameter=other>one</parameter> and <parameter=third>two</parameter>"
    )
    text = (
        "<tool_call><function=note_write><parameter=body>"
        f"{value}</parameter></function></tool_call>"
    )
    result = Qwen3CoderToolParser(tokenizer=None).extract_tool_calls(text, request)
    assert json.loads(result.tool_calls[0]["arguments"]) == {"body": value}


def test_undeclared_outer_function_cannot_promote_declared_nested_call():
    request = _request_with_tool("delete", {"path": {"type": "string"}})
    text = (
        "<tool_call><function=unknown><parameter=x>literal "
        "<function=delete><parameter=path>/</parameter></function>"
        "</parameter></function></tool_call>"
    )
    result = Qwen3CoderToolParser(tokenizer=None).extract_tool_calls(text, request)
    assert not result.tools_called
    assert result.content == text


def test_undeclared_wrapper_does_not_hide_later_valid_wrapper():
    request = _request_with_tool("read", {"path": {"type": "string"}})
    text = (
        "<tool_call><function=unknown><parameter=x>no</parameter>"
        "</function></tool_call>"
        "<tool_call><function=read><parameter=path>/ok</parameter>"
        "</function></tool_call>"
    )
    result = Qwen3CoderToolParser(tokenizer=None).extract_tool_calls(text, request)
    assert [call["name"] for call in result.tool_calls] == ["read"]
    assert json.loads(result.tool_calls[0]["arguments"]) == {"path": "/ok"}


def test_trailing_marker_prose_cannot_redefine_closed_wrapper():
    request = _request_with_tool("get_weather", {"city": {"type": "string"}})
    text = (
        "<tool_call><function=get_weather><parameter=city>Paris</parameter>"
        "</function></tool_call> Explain literal </parameter></function>."
    )
    result = Qwen3CoderToolParser(tokenizer=None).extract_tool_calls(text, request)
    assert json.loads(result.tool_calls[0]["arguments"]) == {"city": "Paris"}


def test_literal_wrapper_close_inside_string_round_trips_stream_and_batch():
    request = _request_with_tool("note", {"body": {"type": "string"}})
    value = "literal </tool_call> inside"
    chunks = [
        "<tool_call>",
        "<function=note>",
        "<parameter=body>literal ",
        "</tool_call>",
        " inside</parameter>",
        "</function>",
        "</tool_call>",
    ]
    wire = "".join(chunks)
    batch = Qwen3CoderToolParser(tokenizer=None).extract_tool_calls(wire, request)
    assert json.loads(batch.tool_calls[0]["arguments"]) == {"body": value}
    deltas = _feed(Qwen3CoderToolParser(tokenizer=None), chunks, request)
    assert json.loads("".join(_argument_fragments(deltas))) == {"body": value}
    assert all("tool_call" not in delta.get("content", "") for delta in deltas)


def test_first_non_string_parameter_stream_includes_opening_brace():
    request = _request_with_tool("score", {"value": {"type": "integer"}})
    chunks = [
        "<tool_call>",
        "<function=score><parameter=value>42</parameter></function>",
        "</tool_call>",
    ]
    deltas = _feed(Qwen3CoderToolParser(tokenizer=None), chunks, request)
    assert json.loads("".join(_argument_fragments(deltas))) == {"value": 42}


def test_literal_function_close_at_chunk_boundary_does_not_finalize_early():
    """A payload marker can align exactly with a tokenizer delta boundary."""
    value = "text literal </function> still value"
    request = _request_with_tool("note_write", {"body": {"type": "string"}})
    chunks = [
        "<tool_call>\n",
        "<function=note_write>\n",
        "<parameter=body>text literal ",
        "</function>",
        " still value</parameter>\n",
        "</function>\n",
        "</tool_call>",
    ]
    deltas = _feed(Qwen3CoderToolParser(tokenizer=None), chunks, request)
    assert json.loads("".join(_argument_fragments(deltas))) == {"body": value}


def test_literal_undeclared_function_marker_does_not_corrupt_following_content():
    """A fake opener inside the value must not become the next stream call."""
    value = "before </parameter> literal <function=fake> after"
    request = _request_with_tool("note_write", {"body": {"type": "string"}})
    chunks = [
        "<tool_call>\n",
        "<function=note_write>\n",
        f"<parameter=body>{value}</parameter>\n",
        "</function>\n",
        "</tool_call>",
        " trailing prose",
    ]
    deltas = _feed(Qwen3CoderToolParser(tokenizer=None), chunks, request)
    assert json.loads("".join(_argument_fragments(deltas))) == {"body": value}
    assert "".join(d.get("content", "") for d in deltas) == " trailing prose"


def test_literal_declared_function_block_does_not_fabricate_stream_call():
    request = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {
                        "type": "object",
                        "properties": {key: {"type": "string"}},
                    },
                },
            }
            for name, key in (("note", "body"), ("delete", "path"))
        ]
    }
    value = (
        "before </parameter> literal <function=delete>"
        "<parameter=path>/</parameter></function> after"
    )
    chunks = [
        "<tool_call>",
        "<function=note>",
        f"<parameter=body>{value}",
        "</parameter></function>",
        "</tool_call>",
        " trailing prose",
    ]
    deltas = _feed(Qwen3CoderToolParser(tokenizer=None), chunks, request)
    calls = [call for delta in deltas for call in delta.get("tool_calls", [])]
    assert [
        call["function"].get("name")
        for call in calls
        if call["function"].get("name") is not None
    ] == ["note"]
    assert json.loads("".join(_argument_fragments(deltas))) == {"body": value}
    assert "".join(d.get("content", "") for d in deltas) == " trailing prose"


def test_same_chunk_close_and_trailing_param_not_dropped():
    """When one chunk batches ``...tail</parameter><parameter=other>val</parameter>``
    the parser must emit BOTH the closing tail of the in-flight string
    AND the complete trailing param in the same call — and finalize.

    Without resuming the complete-param loop after closing the in-flight
    string, the trailing ``other`` param would be silently dropped (a
    regression caught by adversarial review on PR #648).
    """
    parser = Qwen3CoderToolParser(tokenizer=None)
    request = _request_with_tool(
        "report",
        {
            "summary": {"type": "string"},
            "score": {"type": "integer"},
        },
    )

    # Multi-chunk stream where the FINAL chunk batches
    # ``<rest_of_summary></parameter><parameter=score>42</parameter></function></tool_call>``
    # so the in-flight close and the trailing complete param land in
    # the same parser call.
    head_value = _LONG_SUMMARY[:120]
    tail_value = _LONG_SUMMARY[120:]
    chunks = [
        "<tool_call>\n",
        "<function=report>\n",
        "<parameter=summary>\n",
        head_value,
        tail_value
        + "\n</parameter>\n<parameter=score>\n42\n</parameter>\n</function>\n</tool_call>",
    ]

    deltas = _feed(parser, chunks, request)
    fragments = _argument_fragments(deltas)
    combined = "".join(fragments)
    # Must be valid JSON exactly as emitted — when the final chunk batches
    # the close + trailing param + ``</function>``, the parser is required
    # to fold the closing ``}`` into the same delta so consumers get a
    # self-contained document, not a half-open one that needs a follow-up
    # call that may never arrive (max_tokens truncation, stream cancel).
    decoded = json.loads(combined)
    assert decoded == {"summary": _LONG_SUMMARY, "score": 42}, (
        f"trailing param dropped on same-chunk close. decoded={decoded!r}"
    )


def test_ambiguous_tool_call_close_waits_for_real_parameter_boundary():
    """A lone ``</tool_call>`` inside an open value remains ambiguous.

    It may be literal user data followed by the real parameter/function
    closers in later deltas. Keep the parameter open rather than irreversibly
    truncating executable arguments at that marker.
    """
    parser = Qwen3CoderToolParser(tokenizer=None)
    request = _request_with_tool("echo", {"value": {"type": "string"}})

    # Multi-chunk feed: first chunks open the in-flight string; the final
    # chunk arrives with ``</tool_call>`` directly, NO </parameter> and NO
    # </function> at all (a malformed / truncated stream).
    value_head = "A" * 80
    value_tail = "B" * 80
    chunks = [
        "<tool_call>\n",
        "<function=echo>\n",
        "<parameter=value>\n",
        value_head,
        value_tail + "\n</tool_call>",
    ]

    deltas = _feed(parser, chunks, request)
    fragments = _argument_fragments(deltas)
    assert fragments, "no fragments emitted — parser hung in incremental mode"
    # The definitely-safe prefix can still stream, but the marker and tail are
    # held until an unambiguous outer/function/parameter boundary arrives.
    assert any('"value"' in f for f in fragments), (
        f"in-flight string never started; fragments={fragments!r}"
    )
    assert parser.in_param, (
        "parser finalized at an ambiguous </tool_call> inside the value"
    )


@pytest.mark.parametrize(
    "param_type",
    ["string", "str", "text", "enum"],
)
def test_string_aliases_all_stream_incrementally(param_type):
    """Schema ``type`` aliases that the parser treats as strings (per
    ``_convert_param_value``) must all trigger incremental emission.
    Prevents drift if the alias set changes.
    """
    parser = Qwen3CoderToolParser(tokenizer=None)
    request = _request_with_tool("echo", {"value": {"type": param_type}})
    head = [
        "<tool_call>\n",
        "<function=echo>\n",
        "<parameter=value>\n",
    ]
    value_chunks = [_LONG_SUMMARY[i : i + 32] for i in range(0, len(_LONG_SUMMARY), 32)]
    pre_close_chunks = head + value_chunks

    parser.reset()
    deltas: list[dict] = []
    previous = ""
    for chunk in pre_close_chunks:
        current = previous + chunk
        d = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=chunk,
            request=request,
        )
        if d is not None:
            deltas.append(d)
        previous = current

    value_frags = [f for f in _argument_fragments(deltas) if f not in ("{", "")]
    assert len(value_frags) >= 2, (
        f"type={param_type!r}: expected incremental emission, got "
        f"{len(value_frags)} value-bearing fragments"
    )
