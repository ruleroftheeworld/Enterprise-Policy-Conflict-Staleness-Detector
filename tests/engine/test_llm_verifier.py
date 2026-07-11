import json

import pytest

from engine.llm import (
    VerificationResult,
    build_verification_prompt,
    parse_verification_response,
    verify_findings,
)
from shared.contracts.policy_analysis import Finding, NormalizedObligation


class FakeProvider:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class FailingProvider:
    def generate(self, prompt: str) -> str:
        raise RuntimeError("provider unavailable")


def make_obligation(
    obligation_id: str,
    sentence: str,
) -> NormalizedObligation:
    return NormalizedObligation(
        obligation_id=obligation_id,
        policy_id="policy_1",
        section_id="section_1",
        sentence_text=sentence,
        subject="Employees",
        action="use",
        object="multi-factor authentication",
        strength=1.0,
        modality="REQUIRED",
        negated=False,
        confidence=1.0,
    )


def make_finding(
    *,
    score: float = 0.80,
    source_id: str = "obligation_1",
    target_id: str = "obligation_2",
) -> Finding:
    return Finding(
        finding_id="finding_1",
        finding_type="CONTRADICTION",
        source_obligation_id=source_id,
        target_obligation_id=target_id,
        severity="HIGH",
        confidence=score,
        deterministic_score=score,
        explanation="Deterministic contradiction detected.",
    )


@pytest.fixture
def obligations():
    return [
        make_obligation(
            "obligation_1",
            "Employees must use multi-factor authentication.",
        ),
        make_obligation(
            "obligation_2",
            "Employees must not use multi-factor authentication.",
        ),
    ]


def test_parse_valid_verification_response():
    result = parse_verification_response(
        json.dumps(
            {
                "verified": True,
                "confidence": 0.91,
                "explanation": "The requirements conflict.",
            }
        )
    )

    assert result == VerificationResult(
        verified=True,
        confidence=0.91,
        explanation="The requirements conflict.",
    )


def test_parse_clamps_confidence():
    result = parse_verification_response(
        json.dumps(
            {
                "verified": True,
                "confidence": 1.7,
                "explanation": "Confirmed.",
            }
        )
    )

    assert result.confidence == 1.0


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        "[]",
        '{"verified": "yes", "confidence": 0.8, "explanation": "x"}',
        '{"verified": true, "confidence": "high", "explanation": "x"}',
        '{"verified": true, "confidence": 0.8, "explanation": ""}',
    ],
)
def test_parse_rejects_invalid_response(response):
    with pytest.raises(ValueError):
        parse_verification_response(response)


def test_prompt_contains_finding_and_obligation_evidence(obligations):
    finding = make_finding()

    prompt = build_verification_prompt(
        finding,
        obligations[0],
        obligations[1],
    )

    assert "CONTRADICTION" in prompt
    assert obligations[0].sentence_text in prompt
    assert obligations[1].sentence_text in prompt
    assert finding.explanation in prompt


def test_no_provider_preserves_findings(obligations):
    finding = make_finding()

    result, warnings = verify_findings(
        [finding],
        obligations,
        None,
    )

    assert result == [finding]
    assert warnings == []


def test_high_confidence_finding_bypasses_provider(obligations):
    finding = make_finding(score=0.97)
    provider = FakeProvider("{}")

    result, warnings = verify_findings(
        [finding],
        obligations,
        provider,
    )

    assert result == [finding]
    assert provider.prompts == []
    assert warnings == []


def test_low_score_finding_bypasses_provider(obligations):
    finding = make_finding(score=0.60)
    provider = FakeProvider("{}")

    result, warnings = verify_findings(
        [finding],
        obligations,
        provider,
    )

    assert result == [finding]
    assert provider.prompts == []
    assert warnings == []


def test_verified_finding_is_updated(obligations):
    finding = make_finding(score=0.80)

    provider = FakeProvider(
        json.dumps(
            {
                "verified": True,
                "confidence": 0.94,
                "explanation": "The obligations directly conflict.",
            }
        )
    )

    result, warnings = verify_findings(
        [finding],
        obligations,
        provider,
    )

    assert len(result) == 1
    assert result[0].llm_verified is True
    assert result[0].confidence == 0.94
    assert "LLM verification" in result[0].explanation
    assert "llm_verification" in result[0].evidence
    assert warnings == []


def test_rejected_finding_is_removed(obligations):
    finding = make_finding(score=0.80)

    provider = FakeProvider(
        json.dumps(
            {
                "verified": False,
                "confidence": 0.88,
                "explanation": "The context does not establish conflict.",
            }
        )
    )

    result, warnings = verify_findings(
        [finding],
        obligations,
        provider,
    )

    assert result == []
    assert warnings == []


def test_provider_failure_preserves_finding(obligations):
    finding = make_finding(score=0.80)

    result, warnings = verify_findings(
        [finding],
        obligations,
        FailingProvider(),
    )

    assert result == [finding]
    assert len(warnings) == 1
    assert "[llm]" in warnings[0]
    assert "provider unavailable" in warnings[0]


def test_malformed_provider_response_preserves_finding(obligations):
    finding = make_finding(score=0.80)
    provider = FakeProvider("invalid response")

    result, warnings = verify_findings(
        [finding],
        obligations,
        provider,
    )

    assert result == [finding]
    assert len(warnings) == 1
    assert "valid JSON" in warnings[0]


def test_missing_obligation_preserves_finding(obligations):
    finding = make_finding(
        score=0.80,
        target_id="missing_obligation",
    )

    provider = FakeProvider("{}")

    result, warnings = verify_findings(
        [finding],
        obligations,
        provider,
    )

    assert result == [finding]
    assert provider.prompts == []
    assert len(warnings) == 1
    assert "not found" in warnings[0]


@pytest.mark.parametrize(
    ("minimum_score", "maximum_score"),
    [
        (-0.1, 0.9),
        (0.7, 1.1),
        (0.9, 0.9),
        (0.95, 0.70),
    ],
)
def test_invalid_thresholds_raise(
    obligations,
    minimum_score,
    maximum_score,
):
    with pytest.raises(ValueError):
        verify_findings(
            [make_finding()],
            obligations,
            FakeProvider("{}"),
            minimum_score=minimum_score,
            maximum_score=maximum_score,
        )


def test_original_finding_is_not_mutated(obligations):
    finding = make_finding(score=0.80)

    provider = FakeProvider(
        json.dumps(
            {
                "verified": True,
                "confidence": 0.93,
                "explanation": "Confirmed.",
            }
        )
    )

    result, warnings = verify_findings(
        [finding],
        obligations,
        provider,
    )

    assert result[0] is not finding
    assert finding.llm_verified is False
    assert finding.confidence == 0.80
    assert "llm_verification" not in finding.evidence
    assert warnings == []