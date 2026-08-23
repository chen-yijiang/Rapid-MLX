from __future__ import annotations

import pytest

from scripts.route_gui_journeys import ROOT, load_manifest, route

MANIFEST = load_manifest()
ALL_GROUPS = {
    "app-lifecycle",
    "audio",
    "chat",
    "images",
    "models",
    "onboarding-settings",
}


def groups(*paths: str, force_all: bool = False) -> set[str]:
    return set(route(paths, MANIFEST, force_all=force_all)["gui_groups"])


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("apps/rapid-mac/Sources/Rapid/UI/ChatView.swift", {"chat"}),
        ("apps/rapid-mac/Sources/Rapid/UI/Markdown/MathView.swift", {"chat"}),
        ("apps/rapid-mac/Sources/Rapid/UI/AudioView.swift", {"audio"}),
        ("apps/rapid-mac/Sources/Rapid/UI/DictationView.swift", {"audio"}),
        ("apps/rapid-mac/Sources/Rapid/UI/ImagesView.swift", {"images"}),
        (
            "apps/rapid-mac/Sources/Rapid/UI/SettingsModelManagementPanel.swift",
            {"models"},
        ),
        (
            "apps/rapid-mac/Sources/Rapid/UI/OnboardingDirectionD.swift",
            {"onboarding-settings"},
        ),
        (
            "apps/rapid-mac/Sources/Rapid/UI/CampaignBanner.swift",
            {"app-lifecycle"},
        ),
    ],
)
def test_domain_ui_routes_to_one_group(path: str, expected: set[str]):
    assert groups(path) == expected


@pytest.mark.parametrize(
    "path",
    [
        "apps/rapid-mac/Sources/Rapid/Chat/ChatAttachmentDraft.swift",
        "apps/rapid-mac/Sources/Rapid/Audio/AudioClient.swift",
        "apps/rapid-mac/Sources/Rapid/Dictation/DictationEnablePolicy.swift",
        "apps/rapid-mac/Sources/Rapid/Server/ModelCatalog.swift",
        "apps/rapid-mac/Tests/RapidTests/ChatAttachmentDraftTests.swift",
        "README.md",
    ],
)
def test_logic_and_non_gui_paths_do_not_allocate_gui(path: str):
    result = route([path], MANIFEST)
    assert result["gui_required"] is False
    assert result["gui_flows"] == []
    assert result["gui_reason"] == "logic-only"


@pytest.mark.parametrize(
    "path",
    [
        "apps/rapid-mac/Sources/Rapid/UI/BrandNewSurface.swift",
        "apps/rapid-mac/Sources/Rapid/UI/ContentView.swift",
        "apps/rapid-mac/scripts/gui-golden-flows.sh",
        "apps/rapid-mac/Tests/GUIGoldenFlows/journeys.yaml",
        "apps/rapid-mac/Tests/GUIGoldenFlows/__Snapshots__/chat.txt",
        ".github/workflows/rapid-mac-ci.yml",
        "scripts/route_gui_journeys.py",
    ],
)
def test_unknown_or_shared_gui_paths_fail_closed(path: str):
    result = route([path], MANIFEST)
    assert set(result["gui_groups"]) == ALL_GROUPS
    assert result["gui_all"] is True
    assert len(result["gui_flows"]) == 29


def test_mixed_domain_paths_union_groups():
    assert groups(
        "apps/rapid-mac/Sources/Rapid/UI/ChatView.swift",
        "apps/rapid-mac/Sources/Rapid/UI/AudioView.swift",
    ) == {"chat", "audio"}


def test_full_promotion_and_empty_diff_fail_closed():
    assert groups(force_all=True) == ALL_GROUPS
    assert groups() == ALL_GROUPS


def test_selected_flows_are_pr_tier_members_of_selected_groups():
    result = route(["apps/rapid-mac/Sources/Rapid/UI/ChatView.swift"], MANIFEST)
    expected = sorted(
        journey["name"]
        for journey in MANIFEST["journeys"]
        if journey["group"] == "chat" and journey["ci_tier"] == "pr"
    )
    assert result["gui_flows"] == expected
    assert "chat-depth" not in result["gui_flows"]


def test_routing_schema_covers_every_group_and_real_repository_paths():
    routing = MANIFEST["routing"]
    assert set(routing) == {"shared_paths", "gui_roots", "group_paths"}
    assert set(routing["group_paths"]) == ALL_GROUPS

    repository_files = [str(path.relative_to(ROOT)) for path in ROOT.rglob("*")]
    prefixes = [
        *routing["shared_paths"],
        *routing["gui_roots"],
        *(
            prefix
            for group_prefixes in routing["group_paths"].values()
            for prefix in group_prefixes
        ),
    ]
    assert all(isinstance(prefix, str) and prefix for prefix in prefixes)
    assert all(
        any(path == prefix or path.startswith(prefix) for path in repository_files)
        for prefix in prefixes
    )


def test_group_prefixes_are_not_ambiguously_owned():
    owners: dict[str, str] = {}
    for group, prefixes in MANIFEST["routing"]["group_paths"].items():
        for prefix in prefixes:
            assert prefix not in owners, f"{prefix} is owned by two GUI groups"
            owners[prefix] = group
