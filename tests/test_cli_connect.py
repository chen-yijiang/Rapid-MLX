# SPDX-License-Identifier: Apache-2.0
"""Tests for the unified server ready/connect output SSOT.

``vllm_mlx.connect`` is the single source of truth for how a running server's
endpoints are rendered — both the serve lifespan banner and ``rapid-mlx
connect`` (human + ``--json``) consume it. These tests lock:

* the endpoint URL derivations (base vs OpenAI ``/v1`` vs Anthropic),
* the rendered banner shape (Ready / OpenAI / Anthropic / Model / Connect),
* the machine-readable JSON shape (for the desktop / other tooling),
* the socket-activation (inherited-fd) fallback shape,
* the ``connect_command`` plumbing for all three target forms.
"""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout

import pytest

from vllm_mlx import connect
from vllm_mlx.cli import connect_command


def _endpoints(host="localhost", port=8000, model="qwen3.6-35b-4bit"):
    return connect.ServerEndpoints(host, port, model=model)


def _run_connect(*, target=None, json_=False, host=None, port=None, model=None):
    """Invoke ``connect_command`` the way argparse drives it, capturing stdout."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        connect_command(
            argparse.Namespace(
                target=target,
                json=json_,
                host=host,
                port=port,
                model=model,
            )
        )
    return buf.getvalue()


# --- SSOT endpoint derivations ----------------------------------------------
def test_endpoint_urls():
    ep = _endpoints(host="127.0.0.1", port=9000)
    assert ep.base_url == "http://127.0.0.1:9000"
    assert ep.openai_url == "http://127.0.0.1:9000/v1"
    # Anthropic base has NO trailing /v1 (the SDK appends /v1/messages itself).
    assert ep.anthropic_url == "http://127.0.0.1:9000"


def test_json_shape_matches_banner_endpoints():
    ep = _endpoints()
    d = ep.to_dict()
    assert d == {
        "ready": "http://localhost:8000",
        "openai": "http://localhost:8000/v1",
        "anthropic": "http://localhost:8000",
        "model": "qwen3.6-35b-4bit",
    }


def test_render_banner_matches_spec():
    out = connect.render_banner(_endpoints())
    assert "Ready: http://localhost:8000" in out
    assert "OpenAI:    http://localhost:8000/v1" in out
    assert "Anthropic: http://localhost:8000" in out
    assert "Model:     qwen3.6-35b-4bit" in out
    assert "rapid-mlx agents claude-code --setup" in out
    assert "rapid-mlx agents continue --setup" in out
    assert "rapid-mlx connect openai-python" in out


def test_listen_fd_shape():
    ep = connect.ServerEndpoints("", 0, model=None, listen_fd=7)
    out = connect.render_banner(ep)
    assert "Ready: inherited fd 7" in out
    assert "OpenAI:" not in out  # no known address → no endpoint rows
    d = ep.to_dict()
    assert d["ready"] == "inherited fd 7"
    assert d["openai"] is None
    assert d["anthropic"] is None
    assert d["listen_fd"] == 7


# --- endpoints_from_bind (serve-lifespan wiring) ----------------------------
def test_endpoints_from_bind_host_port():
    ep = connect.endpoints_from_bind("localhost", 9123, model="gpt-oss-20b")
    assert ep.base_url == "http://localhost:9123"
    assert ep.model == "gpt-oss-20b"
    assert ep.listen_fd is None


def test_endpoints_from_bind_prefers_fd_when_no_host_port():
    # Mirrors the serve superisor handoff: fd set but no host/port known.
    ep = connect.endpoints_from_bind(None, None, listen_fd=5)
    assert ep.listen_fd == 5


def test_endpoints_from_bind_host_port_wins_over_fd():
    ep = connect.endpoints_from_bind("localhost", 8000, listen_fd=5)
    assert ep.listen_fd is None
    assert ep.base_url == "http://localhost:8000"


# --- connect_command plumbing ------------------------------------------------
def test_connect_no_target_renders_banner(monkeypatch):
    monkeypatch.setattr(
        connect,
        "resolve_endpoints",
        lambda **kw: connect.ServerEndpoints("localhost", 8000, model="m1"),
    )
    out = _run_connect()
    assert "Ready: http://localhost:8000" in out
    assert "Connect:" in out
    assert "OpenAI:" in out


def test_connect_json_is_valid_and_stable(monkeypatch):
    monkeypatch.setattr(
        connect,
        "resolve_endpoints",
        lambda **kw: connect.ServerEndpoints("localhost", 8000, model="m1"),
    )
    out = _run_connect(json_=True)
    payload = json.loads(out)
    assert payload["ready"] == "http://localhost:8000"
    assert payload["openai"] == "http://localhost:8000/v1"
    assert payload["anthropic"] == "http://localhost:8000"
    assert payload["model"] == "m1"


def test_connect_openai_python_snippet(monkeypatch):
    monkeypatch.setattr(
        connect,
        "resolve_endpoints",
        lambda **kw: connect.ServerEndpoints("localhost", 8000, model="m1"),
    )
    out = _run_connect(target="openai-python")
    assert "http://localhost:8000/v1" in out
    assert "OpenAI(" in out
    assert "m1" in out


def test_connect_claude_code_points_at_setup(monkeypatch):
    monkeypatch.setattr(
        connect,
        "resolve_endpoints",
        lambda **kw: connect.ServerEndpoints("localhost", 8000, model=None),
    )
    out = _run_connect(target="claude-code")
    assert "rapid-mlx agents claude-code --setup" in out


def test_connect_continue_points_at_setup(monkeypatch):
    monkeypatch.setattr(
        connect,
        "resolve_endpoints",
        lambda **kw: connect.ServerEndpoints("localhost", 8000, model=None),
    )
    out = _run_connect(target="continue")
    assert "rapid-mlx agents continue --setup" in out


def test_connect_unknown_target_exits(monkeypatch):
    monkeypatch.setattr(
        connect,
        "resolve_endpoints",
        lambda **kw: connect.ServerEndpoints("localhost", 8000, model=None),
    )
    buf = io.StringIO()
    with pytest.raises(SystemExit) as exc, redirect_stdout(buf):
        connect_command(
            argparse.Namespace(
                target="nope", json=False, host=None, port=None, model=None
            )
        )
    assert exc.value.code == 1
    # The helpful supported-target list is printed before exiting.
    assert "Supported: claude-code, continue, openai-python" in buf.getvalue()


# --- P1 fixes applied after #1872 revert ------------------------------------
# (Reviewer feedback on #1871: remote `--host/--port` setup commands must
# carry `--base-url`; IPv6 literals/scoped addresses must be bracket-wrapped;
# `connect --port` must reuse the validated `_port_arg`.)


def test_claude_remote_endpoint_passthrough(monkeypatch):
    """`--host/--port` must flow into the suggested `--base-url` for claude."""
    monkeypatch.setattr(
        connect,
        "resolve_endpoints",
        lambda **kw: connect.ServerEndpoints("mini.local", 9000, model=None),
    )
    out = _run_connect(target="claude-code")
    assert "http://mini.local:9000/v1" in out
    assert "rapid-mlx agents claude-code --setup" in out
    assert "--base-url http://mini.local:9000/v1" in out


def test_continue_remote_endpoint_passthrough(monkeypatch):
    """`--host/--port` must flow into the suggested `--base-url` for continue."""
    monkeypatch.setattr(
        connect,
        "resolve_endpoints",
        lambda **kw: connect.ServerEndpoints("mini.local", 9000, model=None),
    )
    out = _run_connect(target="continue")
    assert "http://mini.local:9000/v1" in out
    assert "--base-url http://mini.local:9000/v1" in out


def test_claude_localhost_keeps_default_base_url(monkeypatch):
    """Default localhost:8000 setup command still emits an explicit base url."""
    monkeypatch.setattr(
        connect,
        "resolve_endpoints",
        lambda **kw: connect.ServerEndpoints("localhost", 8000, model=None),
    )
    out = _run_connect(target="claude-code")
    assert "--base-url http://localhost:8000/v1" in out


def test_ipv6_literal_bracketing():
    ep = _endpoints(host="::1", port=8000)
    assert ep.base_url == "http://[::1]:8000"
    assert ep.openai_url == "http://[::1]:8000/v1"
    assert ep.anthropic_url == "http://[::1]:8000"


def test_ipv6_scoped_address_bracketing():
    # Scoped zone-id address must still be bracket-wrapped.
    ep = _endpoints(host="fe80::1%en0", port=8000)
    assert ep.base_url == "http://[fe80::1%en0]:8000"
    assert ep.openai_url == "http://[fe80::1%en0]:8000/v1"


def test_ipv6_banner_renders_bracketed(monkeypatch):
    monkeypatch.setattr(
        connect,
        "resolve_endpoints",
        lambda **kw: connect.ServerEndpoints("::1", 8000, model=None),
    )
    out = _run_connect()
    assert "Ready: http://[::1]:8000" in out
    assert "OpenAI:    http://[::1]:8000/v1" in out


def test_ipv6_json_renders_bracketed(monkeypatch):
    monkeypatch.setattr(
        connect,
        "resolve_endpoints",
        lambda **kw: connect.ServerEndpoints("::1", 8000, model="m1"),
    )
    payload = json.loads(_run_connect(json_=True))
    assert payload["ready"] == "http://[::1]:8000"
    assert payload["openai"] == "http://[::1]:8000/v1"
    assert payload["anthropic"] == "http://[::1]:8000"


def test_connect_invalid_port_rejected():
    """`connect --port` must reuse `_port_arg`, rejecting 0 / out-of-range."""
    from vllm_mlx.cli import build_parser

    parser = build_parser()
    for bad in ("0", "70000", "-1", "65536"):
        with pytest.raises(SystemExit):
            parser.parse_args(["connect", "--port", bad])


def test_connect_valid_port_accepted():
    """`connect --port` accepts a legitimate in-range value."""
    from vllm_mlx.cli import build_parser

    args = build_parser().parse_args(["connect", "--port", "9000", "--host", "x"])
    assert args.port == 9000
    assert args.host == "x"
