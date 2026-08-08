# SPDX-License-Identifier: Apache-2.0
"""A golden flow may only claim something is ABSENT from an observation it has.

``gui-golden-flows.sh`` proves that things are gone: that Settings closed, that
no video-generation alias reached a chat surface. Written as

    jq -e '[.data.ui_elements[]? | select(...)] | length == 0'

that claim is also satisfied by never having looked. ``rapid-ax`` walks the
accessibility tree with three silent ways to fall short of a full inventory — an
``AXChildren`` read that fails, the depth cap, the record cap — and each removes
a subtree while the dump still says ``success: true``.

It matters because the flow it guards is the one standing between users and
#1603: eight video-generation aliases reaching the picker and dead-ending at
"Couldn't start … Try again" *after* a download of up to 64 GB. A test that can
pass without looking is not a guard against that returning.

The fix is a completeness signal (``data.walk.complete``) the assertion gates
on, and a helper that refuses to answer while it is false. These tests pin the
helper's three-outcome contract, and lint the flows so the raw idiom cannot come
back — it had already been copied to a third site (#1673) before it was fixed
once.

Pure bash + jq, no GUI, no Swift, no GPU: the functions are extracted from the
real script so a copy here cannot drift away from what actually runs.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FLOWS = _REPO_ROOT / "apps" / "rapid-mac" / "scripts" / "gui-golden-flows.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("jq") is None or shutil.which("bash") is None,
    reason="needs bash and jq, which the golden flows require anyway",
)


def _extract(name: str) -> str:
    """Pull one shell function out of the flows script, verbatim."""
    source = _FLOWS.read_text()
    match = re.search(
        rf"^{re.escape(name)}\(\) \{{\n.*?^\}}$", source, re.MULTILINE | re.DOTALL
    )
    assert match, f"{name}() not found in {_FLOWS} — did it get renamed?"
    return match.group(0)


# The filter every catalog assertion uses, in the "does it match?" polarity the
# helper expects.
PRESENT_FILTER = (
    '[.data.ui_elements[]? | select(.identifier == "fake-video-alias")] | length > 0'
)


def _dump(elements, *, complete=True, success=True, reasons=None, walk=True):
    payload = {
        "success": success,
        "data": {
            "pid": 4242,
            "ui_elements": elements,
            "windows": {"titles": ["Rapid-MLX"], "complete": True},
        },
    }
    if walk:
        payload["data"]["walk"] = {"complete": complete, "reasons": reasons or []}
    return payload


def _match(tmp_path, payload, filter_=PRESENT_FILTER) -> int:
    """Exit status of ax_elements_match against ``payload``."""
    dump = tmp_path / "dump.json"
    dump.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    script = textwrap.dedent(
        f"""
        set -uo pipefail
        {_extract("ax_elements_match")}
        ax_elements_match "$1" "$2"
        """
    )
    return subprocess.run(
        ["bash", "-c", script, "bash", str(dump), filter_],
        capture_output=True,
        text=True,
    ).returncode


# ---------------------------------------------------------------------------
# The three outcomes. Folding the third into "absent" is the whole bug.
# ---------------------------------------------------------------------------


def test_a_match_in_a_complete_dump_is_present(tmp_path):
    assert _match(tmp_path, _dump([{"identifier": "fake-video-alias"}])) == 0


def test_no_match_in_a_complete_dump_is_absent(tmp_path):
    assert _match(tmp_path, _dump([{"identifier": "rapid.chat.compose"}])) == 1


def test_an_incomplete_walk_cannot_answer(tmp_path):
    """The case the whole change exists for: nothing matched, but a subtree is
    missing, so "nothing matched" is not an observation of absence."""
    payload = _dump(
        [{"identifier": "rapid.chat.compose"}],
        complete=False,
        reasons=["AXChildren was unreadable on 1 element(s) (last AXError -25204)"],
    )
    assert _match(tmp_path, payload) == 2


def test_a_dump_without_a_walk_signal_cannot_answer(tmp_path):
    """An older driver, or one whose output shape drifted, proves nothing."""
    assert _match(tmp_path, _dump([{"identifier": "x"}], walk=False)) == 2


def test_an_unsuccessful_dump_cannot_answer(tmp_path):
    assert _match(tmp_path, _dump([], success=False)) == 2


def test_ui_elements_that_is_not_an_array_cannot_answer(tmp_path):
    """``[]?`` swallows a structural failure, so without the type check a
    malformed dump reads as a confident "absent"."""
    payload = _dump([])
    payload["data"]["ui_elements"] = {"oops": True}
    assert _match(tmp_path, payload) == 2


def test_unparseable_json_cannot_answer(tmp_path):
    assert _match(tmp_path, "<html>not a dump</html>") == 2


def test_a_broken_query_cannot_answer(tmp_path):
    """jq exits 1 for "false" and 3 for "does not compile". Only the first is an
    answer; treating the second as absence is how a typo becomes a green test."""
    assert _match(tmp_path, _dump([]), filter_=".data.ui_elements[") == 2


def test_an_empty_element_array_is_still_a_real_absence(tmp_path):
    """A complete walk that found nothing is a legitimate "absent" — the gate
    must not be so strict that no flow can ever prove anything."""
    assert _match(tmp_path, _dump([])) == 1


# ---------------------------------------------------------------------------
# assert_ax_absent turns those three outcomes into pass / fail / fail-loudly.
# ---------------------------------------------------------------------------


def _assert_absent(tmp_path, payload) -> subprocess.CompletedProcess:
    dump = tmp_path / "dump.json"
    dump.write_text(json.dumps(payload))
    # `sleep` is shadowed so the retry loop does not cost 5 s. AX_DRIVER is
    # `false` — a driver that cannot observe, which is also the case that proves
    # the retry does not destroy the caller's evidence.
    script = textwrap.dedent(
        f"""
        set -uo pipefail
        sleep() {{ :; }}
        die() {{ echo "DIE: $*" >&2; exit 9; }}
        AX_DRIVER=false
        APP_PID=4242
        {_extract("ax_elements_match")}
        {_extract("assert_ax_absent")}
        {_extract("walk_reasons")}
        assert_ax_absent "$1" "$2" "a video-gen alias reached the chat surface"
        """
    )
    return subprocess.run(
        ["bash", "-c", script, "bash", str(dump), PRESENT_FILTER],
        capture_output=True,
        text=True,
    )


def test_absence_in_a_complete_dump_passes(tmp_path):
    result = _assert_absent(tmp_path, _dump([{"identifier": "rapid.chat.compose"}]))
    assert result.returncode == 0, result.stderr


def test_a_present_element_fails_with_the_callers_message(tmp_path):
    result = _assert_absent(tmp_path, _dump([{"identifier": "fake-video-alias"}]))
    assert result.returncode == 9
    assert "a video-gen alias reached the chat surface" in result.stderr


def test_an_incomplete_dump_fails_loudly_rather_than_passing(tmp_path):
    """The regression this guards: an unobservable dump must never be reported
    as a clean bill of health, and the reason must reach the log."""
    payload = _dump(
        [{"identifier": "rapid.chat.compose"}],
        complete=False,
        reasons=["the record cap of 12000 was reached"],
    )
    result = _assert_absent(tmp_path, payload)
    assert result.returncode == 9, result.stdout + result.stderr
    assert "cannot rule out" in result.stderr
    assert "record cap of 12000" in result.stderr
    # The retries ran against a driver that could not produce anything. The
    # dump the caller captured is the artifact a human debugs from, and it is
    # also where that reason was read from, so it must survive them.
    assert json.loads((tmp_path / "dump.json").read_text()) == payload
    assert not (tmp_path / "dump.json.retry").exists()


# ---------------------------------------------------------------------------
# Lint: the raw idiom must not come back. It had already been copied to a third
# assertion (#1673) between the issue being filed and being fixed.
# ---------------------------------------------------------------------------


def test_no_flow_proves_absence_by_counting_elements_itself():
    source = _FLOWS.read_text()
    # A jq program that reaches into `ui_elements` and then asserts a zero
    # count — the two are within a few hundred characters of each other even
    # when the filter is wrapped across lines.
    offenders = [
        m.group(0)[:120]
        for m in re.finditer(r"ui_elements[\s\S]{0,400}?length\s*==\s*0", source)
    ]
    assert not offenders, (
        "prove absence with assert_ax_absent, which refuses to answer from an "
        "incomplete walk; `length == 0` on its own is satisfied by never having "
        f"looked:\n{offenders}"
    )


def test_the_helper_gates_on_the_completeness_signal():
    """Pin the premise. If the driver stops publishing `walk.complete`, or the
    helper stops consulting it, every test above still passes while the flows go
    back to proving absence from a walk that may be clipped."""
    assert "data.walk.complete == true" in _extract("ax_elements_match")
    driver = (
        _REPO_ROOT / "apps" / "rapid-mac" / "scripts" / "rapid-ax.swift"
    ).read_text()
    assert '"walk": ["complete": walkComplete' in driver
