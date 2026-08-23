# SPDX-License-Identifier: Apache-2.0
"""Contracts for lane-scoped full-CI promotion.

These tests intentionally inspect the workflows: a future cleanup must not
restore the expensive behavior where applying ``full-ci`` changed an
engine-only or Desktop-only PR into an all-product run.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
ENGINE_WORKFLOW = ROOT / ".github/workflows/ci.yml"
DESKTOP_WORKFLOW = ROOT / ".github/workflows/rapid-mac-ci.yml"


def _step_run(workflow: Path, job: str, step_name: str) -> str:
    steps = yaml.safe_load(workflow.read_text())["jobs"][job]["steps"]
    (step,) = [candidate for candidate in steps if candidate.get("name") == step_name]
    return str(step["run"])


def _job(workflow: Path, job: str) -> dict[str, object]:
    return yaml.safe_load(workflow.read_text())["jobs"][job]


def test_engine_full_ci_still_classifies_the_pr_diff():
    run = _step_run(ENGINE_WORKFLOW, "changes", "Classify validation lanes")
    assert 'git diff --no-renames --name-only "$PR_BASE_SHA" "$GITHUB_SHA"' in run
    assert 'full_gate="$FULL_CI"' in run
    assert 'if [ "$FULL_CI" = true ]' not in run


def test_desktop_full_ci_still_classifies_the_pr_diff():
    run = _step_run(DESKTOP_WORKFLOW, "changes", "Classify desktop lane")
    assert 'git diff --no-renames --name-only "$PR_BASE_SHA" "$GITHUB_SHA"' in run
    assert 'echo "full_gate=$FULL_CI"' in run
    assert '|| [ "$FULL_CI" = true ]' not in run


def test_non_engine_change_exits_before_full_ci_requirement():
    run = _step_run(ENGINE_WORKFLOW, "tests", "Check test results")
    classifier_gate = run.index("needs.changes.result")
    common_gate = run.index("needs.lint.result")
    no_lane = run.index('if [ "$expected" != "true" ]')
    engine_gate = run.index("needs.engine-contracts.result")
    promotion = run.index("needs.changes.outputs.full_gate")
    assert classifier_gate < common_gate < no_lane < engine_gate < promotion


def test_non_desktop_change_exits_before_desktop_results():
    run = _step_run(DESKTOP_WORKFLOW, "desktop-tests", "Check desktop results")
    classifier_gate = run.index("needs.changes.result")
    no_lane = run.index('if [ "$DESKTOP_EXPECTED" != true ]')
    build_gate = run.index('for result in "$IDENTIFIERS"')
    gui_gate = run.index('if [ "$GUI_REQUIRED" = true ]')
    assert classifier_gate < no_lane < build_gate < gui_gate


def test_gui_golden_job_requires_desktop_lane_and_routed_gui_work():
    condition = str(_job(DESKTOP_WORKFLOW, "gui-golden-flows")["if"])
    assert "needs.changes.outputs.desktop == 'true'" in condition
    assert "needs.changes.outputs.gui_required == 'true'" in condition
    assert "needs.changes.outputs.full_gate" not in condition


def test_full_ci_is_an_all_gui_override_not_a_merge_prerequisite():
    route_run = _step_run(DESKTOP_WORKFLOW, "changes", "Route GUI journeys")
    assert '[ "$FULL_CI" = true ]' in route_run
    assert "--force-all" in route_run

    aggregate = _step_run(DESKTOP_WORKFLOW, "desktop-tests", "Check desktop results")
    assert "apply the full-ci label" not in aggregate
    assert 'if [ "$GUI_REQUIRED" = true ]' in aggregate


def test_gui_router_dependencies_are_desktop_scoped():
    changes = _job(DESKTOP_WORKFLOW, "changes")
    scoped_steps = {
        step["name"]: step
        for step in changes["steps"]
        if step.get("name")
        in {
            "Set up Python for GUI routing",
            "Install routing dependency",
            "Route GUI journeys",
        }
    }
    assert set(scoped_steps) == {
        "Set up Python for GUI routing",
        "Install routing dependency",
        "Route GUI journeys",
    }
    assert all(
        step.get("if") == "steps.policy.outputs.desktop == 'true'"
        for step in scoped_steps.values()
    )


def test_engine_only_contracts_are_not_universal_pr_guards():
    universal_steps = {
        step.get("name") for step in _job(ENGINE_WORKFLOW, "lint")["steps"]
    }
    engine_steps = {
        step.get("name") for step in _job(ENGINE_WORKFLOW, "engine-contracts")["steps"]
    }
    assert {
        "GitHub Actions SHA pinning",
        "Workflow expression sanity",
        "Model-management architecture SSOT",
        "Run ruff lint",
        "Run ruff format check",
        "Engine ↔ desktop app version sync",
    } <= universal_steps
    assert {
        "CLI ↔ Config fidelity audit",
        "Release-script offline tests",
        "Installer offline tests",
        "Parser microbench",
    } <= engine_steps
    assert not universal_steps & {
        "CLI ↔ Config fidelity audit",
        "Release-script offline tests",
        "Installer offline tests",
        "Parser microbench",
    }


def test_engine_jobs_follow_fail_closed_engine_classification():
    for job_name in ("engine-contracts", "type-check"):
        job = _job(ENGINE_WORKFLOW, job_name)
        assert job["needs"] == "changes"
        assert str(job["if"]) == "needs.changes.outputs.engine == 'true'"

    bound_guard = _job(ENGINE_WORKFLOW, "mlx-bound-guard")
    assert bound_guard["needs"] == "changes"
    condition = str(bound_guard["if"])
    assert "github.event_name == 'pull_request'" in condition
    assert "needs.changes.outputs.engine == 'true'" in condition
