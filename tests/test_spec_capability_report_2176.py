from scripts.spec_capability_report import build_report
from vllm_mlx.model_aliases import AliasProfile, list_profiles
from vllm_mlx.spec_decode.capability import REGISTERED_METHODS, assess_method


def test_report_covers_every_alias_and_method():
    report = build_report()
    assert set(report) == set(list_profiles())
    assert all(
        set(entry["methods"]) == set(REGISTERED_METHODS) for entry in report.values()
    )
    assert all(
        {"hf_path", "checkpoint_format", "quantization", "modality"}
        <= set(entry["model"])
        for entry in report.values()
    )


def test_4bit_is_recommendation_evidence_not_capability_ban():
    profile = AliasProfile(hf_path="user/Qwen3.5-9B-abliterated-4bit")
    for method in ("mtp", "dflash", "ddtree", "dspark"):
        assessment = assess_method(profile, method)
        assert assessment.recommendation == "experimental"
        assert assessment.explicit_opt_in is True
        assert assessment.capable is not False


def test_true_verifier_incompatibility_remains_hard_failure():
    hybrid = AliasProfile(hf_path="user/hybrid", is_hybrid=True)
    suffix = assess_method(hybrid, "suffix")
    assert suffix.capable is False
    assert suffix.recommendation == "incompatible"
