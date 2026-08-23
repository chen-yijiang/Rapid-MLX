#!/usr/bin/env python3
"""Route changed paths to the smallest safe GUI journey set.

Non-rendering Desktop logic intentionally selects no GUI group: Swift unit
tests and the app build own that layer. Unknown GUI paths, shared navigation,
test infrastructure, and explicit full promotion fail closed to every group.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "apps/rapid-mac/Tests/GUIGoldenFlows/journeys.yaml"


def _matches(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix)


def route(
    paths: Iterable[str], manifest: Mapping[str, Any], *, force_all: bool = False
) -> dict[str, object]:
    journeys = manifest["journeys"]
    routing = manifest["routing"]
    all_groups = sorted({journey["group"] for journey in journeys})
    normalized = sorted(
        {path.strip().removeprefix("./") for path in paths if path.strip()}
    )

    selected: set[str] = set()
    all_reason: str | None = "explicit-full" if force_all else None
    if not normalized and not force_all:
        all_reason = "empty-diff"

    for path in normalized:
        if any(_matches(path, prefix) for prefix in routing["shared_paths"]):
            all_reason = f"shared:{path}"
            break

        matched = {
            group
            for group, prefixes in routing["group_paths"].items()
            if any(_matches(path, prefix) for prefix in prefixes)
        }
        if matched:
            selected.update(matched)
        elif any(_matches(path, root) for root in routing["gui_roots"]):
            all_reason = f"unclassified-gui:{path}"
            break

    if all_reason:
        selected = set(all_groups)

    flows = sorted(
        journey["name"]
        for journey in journeys
        if journey["ci_tier"] == "pr" and journey["group"] in selected
    )
    return {
        "gui_required": bool(selected),
        "gui_groups": sorted(selected),
        "gui_flows": flows,
        "gui_all": selected == set(all_groups),
        "gui_reason": all_reason or ("matched" if selected else "logic-only"),
    }


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--paths-file", type=argparse.FileType("r"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--force-all", action="store_true")
    parser.add_argument("--github-output", type=argparse.FileType("a"))
    args = parser.parse_args()

    paths = list(args.paths)
    if args.paths_file:
        paths.extend(args.paths_file.read().splitlines())
    result = route(paths, load_manifest(args.manifest), force_all=args.force_all)

    if args.github_output:
        for key, value in result.items():
            encoded = json.dumps(value, separators=(",", ":"))
            print(f"{key}={encoded}", file=args.github_output)
    else:
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
