#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Emit the all-alias/all-method speculative capability report as JSON."""

from __future__ import annotations

import json

from vllm_mlx.model_aliases import list_profiles
from vllm_mlx.spec_decode.capability import REGISTERED_METHODS, assess_method


def build_report() -> dict[str, dict[str, dict]]:
    return {
        alias: {
            method: assess_method(profile, method).to_dict()
            for method in REGISTERED_METHODS
        }
        for alias, profile in sorted(list_profiles().items())
    }


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2, sort_keys=True))
