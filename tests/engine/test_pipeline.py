from pathlib import Path

import pytest

from engine.pipeline import analyze_policy_documents
from shared.contracts.policy_analysis import AnalysisResult


FIXTURES = Path("tests/engine/fixtures")


def test_pipeline_returns_analysis_result():
    result = analyze_policy_documents(
        [FIXTURES / "access_policy.txt"]
    )

    assert isinstance(result, AnalysisResult)
    assert len(result.policies) == 1
    assert len(result.obligations) == 6
    assert result.processing_time_ms >= 0.0


def test_pipeline_accepts_single_document_path():
    result = analyze_policy_documents(
        FIXTURES / "access_policy.txt"
    )

    assert len(result.policies) == 1
    assert len(result.obligations) == 6


def test_pipeline_accepts_single_string_path():
    result = analyze_policy_documents(
        str(FIXTURES / "access_policy.txt")
    )

    assert len(result.policies) == 1
    assert len(result.obligations) == 6


def test_pipeline_empty_input_returns_empty_result():
    result = analyze_policy_documents([])

    assert result.policies == []
    assert result.obligations == []
    assert result.findings == []
    assert result.statistics["documents_received"] == 0
    assert result.statistics["policies_processed"] == 0
    assert result.statistics["candidate_pairs"] == 0
    assert result.statistics["findings_total"] == 0
    assert result.warnings == []


def test_pipeline_detects_expected_cross_policy_findings():
    result = analyze_policy_documents(
        [
            FIXTURES / "access_policy.txt",
            FIXTURES / "conflicting_policy.txt",
            FIXTURES / "network_policy.md",
            FIXTURES / "legacy_policy.txt",
        ]
    )

    finding_types = {
        finding.finding_type
        for finding in result.findings
    }

    assert "CONTRADICTION" in finding_types
    assert "FREQUENCY_MISMATCH" in finding_types
    assert "REDUNDANCY" in finding_types
    assert "MODALITY_INCONSISTENCY" in finding_types

    assert result.statistics["documents_received"] == 4
    assert result.statistics["policies_processed"] == 4
    assert result.statistics["documents_failed"] == 0
    assert result.statistics["obligations_extracted"] == 20
    assert result.statistics["candidate_pairs"] == 11
    assert result.statistics["findings_total"] == 8


def test_pipeline_returns_embedded_obligations():
    result = analyze_policy_documents(
        [FIXTURES / "access_policy.txt"]
    )

    assert result.obligations

    for obligation in result.obligations:
        assert len(obligation.embedding) == 384


def test_pipeline_preserves_policy_input_order():
    result = analyze_policy_documents(
        [
            FIXTURES / "legacy_policy.txt",
            FIXTURES / "access_policy.txt",
        ]
    )

    assert len(result.policies) == 2

    assert result.policies[0].source_file.endswith(
        "legacy_policy.txt"
    )

    assert result.policies[1].source_file.endswith(
        "access_policy.txt"
    )


def test_pipeline_isolates_missing_document_failure():
    result = analyze_policy_documents(
        [
            FIXTURES / "missing_policy.txt",
            FIXTURES / "access_policy.txt",
        ]
    )

    assert len(result.policies) == 1
    assert len(result.obligations) == 6

    assert result.statistics["documents_received"] == 2
    assert result.statistics["policies_processed"] == 1
    assert result.statistics["documents_failed"] == 1

    assert len(result.warnings) == 1
    assert "[ingestion]" in result.warnings[0]
    assert "missing_policy.txt" in result.warnings[0]


def test_pipeline_invalid_document_element_raises_type_error():
    with pytest.raises(TypeError):
        analyze_policy_documents(
            [
                FIXTURES / "access_policy.txt",
                123,
            ]
        )


def test_pipeline_invalid_documents_argument_raises_type_error():
    with pytest.raises(TypeError):
        analyze_policy_documents(123)


def test_pipeline_statistics_match_findings():
    result = analyze_policy_documents(
        [
            FIXTURES / "access_policy.txt",
            FIXTURES / "conflicting_policy.txt",
            FIXTURES / "network_policy.md",
            FIXTURES / "legacy_policy.txt",
        ]
    )

    assert (
        sum(result.statistics["findings_by_type"].values())
        == len(result.findings)
    )

    assert (
        sum(result.statistics["findings_by_severity"].values())
        == len(result.findings)
    )


def test_pipeline_result_is_contract_serializable():
    result = analyze_policy_documents(
        [
            FIXTURES / "access_policy.txt",
            FIXTURES / "conflicting_policy.txt",
        ]
    )

    dumped = result.model_dump(mode="json")

    assert isinstance(dumped, dict)
    assert isinstance(dumped["policies"], list)
    assert isinstance(dumped["obligations"], list)
    assert isinstance(dumped["findings"], list)
    assert isinstance(dumped["statistics"], dict)
    assert isinstance(dumped["warnings"], list)
    assert dumped["processing_time_ms"] >= 0.0


def test_pipeline_is_deterministic_except_processing_time():
    documents = [
        FIXTURES / "access_policy.txt",
        FIXTURES / "conflicting_policy.txt",
    ]

    first = analyze_policy_documents(documents)
    second = analyze_policy_documents(documents)

    first_dump = first.model_dump()
    second_dump = second.model_dump()

    first_dump.pop("processing_time_ms")
    second_dump.pop("processing_time_ms")

    assert first_dump == second_dump

def test_frozen_public_api_import_and_return_contract():
    from engine.pipeline import analyze_policy_documents
    from shared.contracts.policy_analysis import AnalysisResult

    result = analyze_policy_documents(
        [FIXTURES / "access_policy.txt"]
    )

    assert isinstance(result, AnalysisResult)

import json


class PipelineFakeLLMProvider:
    def __init__(self, response: dict):
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps(self.response)


class PipelineFailingLLMProvider:
    def generate(self, prompt: str) -> str:
        raise RuntimeError("pipeline provider unavailable")


def test_pipeline_default_behavior_unchanged_without_llm():
    documents = [
        FIXTURES / "access_policy.txt",
        FIXTURES / "conflicting_policy.txt",
        FIXTURES / "network_policy.md",
        FIXTURES / "legacy_policy.txt",
    ]

    result = analyze_policy_documents(documents)

    assert len(result.findings) == 8
    assert all(
        finding.llm_verified is False
        for finding in result.findings
    )


def test_pipeline_llm_verifies_ambiguous_findings():
    provider = PipelineFakeLLMProvider(
        {
            "verified": True,
            "confidence": 0.93,
            "explanation": "The deterministic finding is valid.",
        }
    )

    result = analyze_policy_documents(
        [
            FIXTURES / "access_policy.txt",
            FIXTURES / "conflicting_policy.txt",
            FIXTURES / "network_policy.md",
            FIXTURES / "legacy_policy.txt",
        ],
        llm_provider=provider,
    )

    assert provider.prompts

    assert any(
        finding.llm_verified
        for finding in result.findings
    )

    assert result.statistics["findings_total"] == len(result.findings)


def test_pipeline_llm_can_reject_ambiguous_findings():
    provider = PipelineFakeLLMProvider(
        {
            "verified": False,
            "confidence": 0.90,
            "explanation": "The finding is not supported by context.",
        }
    )

    result = analyze_policy_documents(
        [
            FIXTURES / "access_policy.txt",
            FIXTURES / "conflicting_policy.txt",
            FIXTURES / "network_policy.md",
            FIXTURES / "legacy_policy.txt",
        ],
        llm_provider=provider,
    )

    assert provider.prompts
    assert len(result.findings) < 8
    assert result.statistics["findings_total"] == len(result.findings)


def test_pipeline_llm_provider_failure_is_fail_safe():
    result = analyze_policy_documents(
        [
            FIXTURES / "access_policy.txt",
            FIXTURES / "conflicting_policy.txt",
            FIXTURES / "network_policy.md",
            FIXTURES / "legacy_policy.txt",
        ],
        llm_provider=PipelineFailingLLMProvider(),
    )

    assert len(result.findings) == 8
    assert any(
        "[llm]" in warning
        and "pipeline provider unavailable" in warning
        for warning in result.warnings
    )


def test_pipeline_single_policy_still_works_with_llm_provider():
    provider = PipelineFakeLLMProvider(
        {
            "verified": True,
            "confidence": 0.95,
            "explanation": "Confirmed.",
        }
    )

    result = analyze_policy_documents(
        [FIXTURES / "access_policy.txt"],
        llm_provider=provider,
    )

    assert len(result.policies) == 1
    assert len(result.obligations) == 6
    assert result.findings == []
    assert provider.prompts == []

def test_pipeline_reports_disabled_llm_observability():
    result = analyze_policy_documents(
        [
            FIXTURES / "access_policy.txt",
            FIXTURES / "conflicting_policy.txt",
            FIXTURES / "network_policy.md",
            FIXTURES / "legacy_policy.txt",
        ]
    )

    assert result.statistics["llm_enabled"] is False
    assert result.statistics["llm_provider"] is None
    assert result.statistics["llm_model"] is None
    assert result.statistics["llm_eligible_findings"] == 0
    assert result.statistics["llm_verified_findings"] == 0
    assert result.statistics["llm_rejected_findings"] == 0
    assert result.statistics["llm_bypassed_findings"] == 0
    assert result.statistics["llm_failed_findings"] == 0


def test_pipeline_reports_llm_observability():
    provider = PipelineFakeLLMProvider(
        {
            "verified": True,
            "confidence": 0.93,
            "explanation": "Confirmed.",
        }
    )

    result = analyze_policy_documents(
        [
            FIXTURES / "access_policy.txt",
            FIXTURES / "conflicting_policy.txt",
            FIXTURES / "network_policy.md",
            FIXTURES / "legacy_policy.txt",
        ],
        llm_provider=provider,
    )

    assert result.statistics["llm_enabled"] is True
    assert result.statistics["llm_provider"] == "PipelineFakeLLMProvider"
    assert result.statistics["llm_model"] is None
    assert result.statistics["llm_eligible_findings"] == 3
    assert result.statistics["llm_verified_findings"] == 3
    assert result.statistics["llm_rejected_findings"] == 0
    assert result.statistics["llm_bypassed_findings"] == 5
    assert result.statistics["llm_failed_findings"] == 0