# SPDX-License-Identifier: Apache-2.0
"""The macOS golden gate must not silently omit named GUI journeys."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "apps/rapid-mac/scripts/gui-golden-flows.sh"
WORKFLOW = ROOT / ".github/workflows/rapid-mac-ci.yml"
MANIFEST = ROOT / "apps/rapid-mac/Tests/GUIGoldenFlows/journeys.yaml"

# `chat-depth` requires all five turns to be simultaneously realised in AX.
# The hosted runner's 1024x681 app window virtualises the oldest messages, so
# that assertion is valid on larger local displays but false by construction in
# CI. Keep this exception exact: any additional omission is accidental.
CI_EXCLUSIONS = {"chat-depth"}


def harness_flows() -> set[str]:
    source = HARNESS.read_text()
    dispatcher = source.rsplit('case "$FLOW" in', 1)[1].split("esac", 1)[0]
    return set(re.findall(r"^    ([a-z][a-z0-9-]+)\)", dispatcher, re.MULTILINE)) - {
        "all"
    }


def workflow_flows() -> set[str]:
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["gui-golden-flows"]["steps"]
    return {
        match.group(1)
        for step in steps
        if (
            match := re.search(
                r"gui-golden-flows\.sh --flow ([a-z0-9-]+)", step.get("run", "")
            )
        )
    }


def workflow_steps() -> list[dict[str, object]]:
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]["gui-golden-flows"]["steps"]


def manifest_journeys() -> list[dict[str, object]]:
    payload = yaml.safe_load(MANIFEST.read_text())
    assert payload["version"] == 1
    return payload["journeys"]


def diagnostic_flows() -> set[str]:
    workflow = WORKFLOW.read_text()
    loop = workflow.split("for flow in ", 1)[1].split("; do", 1)[0]
    return set(loop.split())


def baseline_flows() -> set[str]:
    source = HARNESS.read_text()
    owners: set[str] = set()
    for match in re.finditer(
        r"^flow_([a-z0-9_]+)\(\) \{\n(.*?)(?=^\})", source, re.MULTILINE | re.DOTALL
    ):
        if re.search(r"\bbaseline\s", match.group(2)):
            owners.add(match.group(1).replace("_", "-"))
    return owners


def test_every_named_flow_is_gated_or_explicitly_excluded():
    named = harness_flows()
    gated = workflow_flows()
    assert named - gated == CI_EXCLUSIONS
    assert not gated - named


def test_manifest_is_the_complete_unique_flow_inventory():
    journeys = manifest_journeys()
    names = [str(journey["name"]) for journey in journeys]
    assert len(names) == len(set(names))
    assert set(names) == harness_flows()


def test_manifest_fields_are_valid_and_fail_closed():
    allowed_groups = {
        "chat",
        "audio",
        "models",
        "onboarding-settings",
        "images",
        "app-lifecycle",
    }
    allowed_risks = {"low", "medium", "high"}
    allowed_drivers = {"ax", "xcuitest", "hybrid"}
    allowed_tiers = {"pr", "local"}

    for journey in manifest_journeys():
        assert journey["group"] in allowed_groups
        assert journey["risk"] in allowed_risks
        assert journey["driver"] in allowed_drivers
        assert journey["ci_tier"] in allowed_tiers
        assert journey["fixtures"]
        assert journey["source_paths"]
        assert all(
            str(path).startswith("apps/rapid-mac/") for path in journey["source_paths"]
        )
        assert all((ROOT / str(path)).exists() for path in journey["source_paths"])
        assert isinstance(journey["owns_baseline"], bool)


def test_manifest_ci_tiers_match_the_workflow_contract():
    pr_flows = {
        str(journey["name"])
        for journey in manifest_journeys()
        if journey["ci_tier"] == "pr"
    }
    local_flows = {
        str(journey["name"])
        for journey in manifest_journeys()
        if journey["ci_tier"] == "local"
    }
    assert pr_flows == workflow_flows()
    assert local_flows == CI_EXCLUSIONS


def test_manifest_baseline_ownership_matches_harness_usage():
    declared = {
        str(journey["name"])
        for journey in manifest_journeys()
        if journey["owns_baseline"]
    }
    assert declared == baseline_flows()


def test_result_evidence_records_timing_and_artifact_location():
    source = HARNESS.read_text()
    assert source.count("duration_seconds: $duration_seconds") == 2
    assert source.count("artifact_path: $artifact_path") == 2
    assert source.count("started_at: $started_at") == 2


def test_failure_diagnostic_regenerates_every_ci_baseline_and_nothing_else():
    assert diagnostic_flows() == workflow_flows() & baseline_flows()


def test_failure_diagnostic_skips_regeneration_for_semantic_failures():
    steps = workflow_steps()
    (diagnostic,) = [
        step
        for step in steps
        if step.get("name") == "Regenerate baselines on this runner (diagnostic)"
    ]
    run = str(diagnostic.get("run", ""))
    assert 'for result in "$GOLDEN_ROOT"/*/result.json' in run
    assert 'find "$(dirname "$result")" -name \'*.observed.txt\'' in run
    assert "No structural baseline mismatch" in run


def test_all_named_flows_run_before_one_blocking_verdict():
    steps = workflow_steps()
    flow_steps = [
        step for step in steps if str(step.get("name", "")).startswith("Golden flow:")
    ]
    assert len(flow_steps) == len(workflow_flows())
    assert all(step.get("continue-on-error") is True for step in flow_steps)

    verdicts = [
        step for step in steps if step.get("name") == "Require every named golden flow"
    ]
    assert len(verdicts) == 1
    verdict = verdicts[0]
    assert verdict.get("if") == "always()"
    assert f"expected = {len(workflow_flows())}" in str(verdict.get("run", ""))


def test_golden_job_builds_the_release_ui_surface():
    """Release baselines cannot be compared against Debug-only controls."""
    build_steps = [
        step
        for step in workflow_steps()
        if step.get("name") == "Build Rapid-MLX Desktop.app"
    ]
    assert len(build_steps) == 1
    assert build_steps[0].get("env", {}).get("RAPID_BUILD_CONFIG") == "release"
