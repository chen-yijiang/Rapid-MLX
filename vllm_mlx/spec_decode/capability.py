# SPDX-License-Identifier: Apache-2.0
"""Capability and recommendation tiers for speculative decoding.

Registry flags are recommendation evidence. They are not permission checks for
an operator-explicit target/drafter pair; runtime preflight owns compatibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from vllm_mlx.model_profile import ModelProfile


@dataclass(frozen=True)
class SpecCapability:
    method: str
    capable: bool | None
    recommendation: str
    explicit_opt_in: bool
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


def assess_method(profile: ModelProfile, method: str) -> SpecCapability:
    """Statically assess one alias; runtime-only facts remain unknown."""
    if profile.modality != "text":
        return SpecCapability(
            method, False, "incompatible", False, ("non-text serving lane",)
        )
    if method == "suffix":
        if profile.is_hybrid or not profile.supports_spec_decode:
            return SpecCapability(
                method,
                False,
                "incompatible",
                False,
                ("target verifier does not support lossless rollback",),
            )
        tier = profile.suffix_decoding_tier
        recommendation = "verified" if tier == "verified" else "experimental"
        return SpecCapability(method, True, recommendation, True)
    if method == "dflash":
        if profile.is_moe:
            return SpecCapability(
                method, False, "incompatible", False, ("MoE verifier unsupported",)
            )
        return SpecCapability(
            method,
            None,
            "verified" if profile.supports_dflash else "experimental",
            True,
            warnings=("requires a structurally compatible drafter",),
        )
    if method == "ddtree":
        if profile.is_moe:
            return SpecCapability(
                method, False, "incompatible", False, ("MoE tree verifier unsupported",)
            )
        return SpecCapability(
            method,
            None,
            "verified" if profile.supports_ddtree else "experimental",
            True,
            warnings=("requires drafter, speculative-token, and tree-budget metadata",),
        )
    if method == "mtp":
        return SpecCapability(
            method,
            None,
            "verified" if profile.mtp_draft_model else "experimental",
            True,
            warnings=("runtime validates architecture and sidecar metadata",),
        )
    if method == "dspark":
        return SpecCapability(
            method,
            None,
            "experimental",
            True,
            warnings=("runtime validates embedded DSpark metadata",),
        )
    raise ValueError(f"unknown speculative method: {method}")


REGISTERED_METHODS = ("suffix", "mtp", "dflash", "ddtree", "dspark")
