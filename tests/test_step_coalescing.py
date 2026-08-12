# SPDX-License-Identifier: Apache-2.0
"""Tests for EngineCore._step_coalesced (#1861 conc investigation).

Deep batches run up to 4 scheduler steps per executor dispatch; the
helper must break early on finishes / no-work / pending admissions and
must deliver partial outputs when a later step raises (a step that
already advanced scheduler state has produced tokens the collectors
must still receive — codex r1 on #1878).
"""

from __future__ import annotations

from types import SimpleNamespace

from vllm_mlx.engine_core import EngineCore


def _out(finished=(), has_work=True):
    return SimpleNamespace(
        finished_request_ids=list(finished), has_work=has_work, outputs=[]
    )


class _ScriptedScheduler:
    """Yields scripted step results; an Exception instance raises."""

    def __init__(self, script, waiting=0):
        self.script = list(script)
        self.calls = 0
        self._waiting = waiting

    def step(self):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def get_num_waiting(self):
        return self._waiting


def _engine(scheduler):
    eng = EngineCore.__new__(EngineCore)
    eng.scheduler = scheduler
    return eng


def test_runs_up_to_max_steps():
    sched = _ScriptedScheduler([_out(), _out(), _out(), _out()])
    outs, err = _engine(sched)._step_coalesced(4)
    assert err is None
    assert len(outs) == 4
    assert sched.calls == 4


def test_breaks_on_finish():
    sched = _ScriptedScheduler([_out(), _out(finished=["r1"]), _out()])
    outs, err = _engine(sched)._step_coalesced(4)
    assert err is None
    assert len(outs) == 2  # stops the step after a finish
    assert sched.calls == 2


def test_breaks_on_no_work():
    sched = _ScriptedScheduler([_out(has_work=False), _out()])
    outs, err = _engine(sched)._step_coalesced(4)
    assert err is None
    assert len(outs) == 1


def test_breaks_on_pending_admissions():
    sched = _ScriptedScheduler([_out(), _out()], waiting=1)
    outs, err = _engine(sched)._step_coalesced(4)
    assert err is None
    assert len(outs) == 1  # work was already waiting at dispatch time


def test_partial_outputs_preserved_on_error():
    """codex r1 BLOCKING: a later step raising must not discard outputs
    from earlier steps that already advanced scheduler state — their
    tokens are produced and their finish events must still fire."""
    boom = RuntimeError("step 3 exploded")
    sched = _ScriptedScheduler([_out(), _out(), boom])
    outs, err = _engine(sched)._step_coalesced(4)
    assert err is boom
    assert len(outs) == 2  # both successful steps' outputs survive
