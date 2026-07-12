import pytest

from engine.candidates import generate_candidate_pairs
from engine.detection import (
    CONTRADICTION,
    FREQUENCY_MISMATCH,
    MODALITY_INCONSISTENCY,
    REDUNDANCY,
    detect_findings,
)
from engine.embeddings import embed_obligations
from engine.extraction import extract_obligation
from engine.normalization import normalize_obligation


def _obligation(
    sentence: str,
    *,
    policy_id: str,
):
    obligation = extract_obligation(
        policy_id=policy_id,
        section_id="section_test",
        sentence=sentence,
    )

    assert obligation is not None

    return normalize_obligation(obligation)


def _detect(
    sentences: list[tuple[str, str]],
    *,
    similarity_threshold: float = 0.40,
):
    obligations = [
        _obligation(sentence, policy_id=policy_id)
        for sentence, policy_id in sentences
    ]

    embedded, warnings = embed_obligations(obligations)

    assert warnings == []

    candidates = generate_candidate_pairs(
        embedded,
        similarity_threshold=similarity_threshold,
        top_k_per_obligation=None,
    )

    findings = detect_findings(
        embedded,
        candidates,
    )

    return embedded, candidates, findings


def test_required_vs_prohibited_is_critical_contradiction():
    _, candidates, findings = _detect(
        [
            (
                "Employees must use multi-factor authentication for remote access.",
                "policy_a",
            ),
            (
                "Employees must not use multi-factor authentication for remote access.",
                "policy_b",
            ),
        ]
    )

    assert len(candidates) == 1
    assert len(findings) == 1

    finding = findings[0]

    assert finding.finding_type == CONTRADICTION
    assert finding.severity == "CRITICAL"
    assert finding.llm_verified is False
    assert finding.confidence > 0.90
    assert finding.deterministic_score > 0.90


def test_required_vs_prohibited_has_evidence():
    _, _, findings = _detect(
        [
            (
                "Employees must use multi-factor authentication for remote access.",
                "policy_a",
            ),
            (
                "Employees must not use multi-factor authentication for remote access.",
                "policy_b",
            ),
        ]
    )

    evidence = findings[0].evidence

    assert evidence["source_sentence"]
    assert evidence["target_sentence"]
    assert evidence["source_policy_id"] == "policy_a"
    assert evidence["target_policy_id"] == "policy_b"
    assert evidence["source_action"] == "use"
    assert evidence["target_action"] == "use"


def test_frequency_mismatch_is_detected():
    _, _, findings = _detect(
        [
            (
                "Administrators must review privileged user accounts every 30 days.",
                "policy_a",
            ),
            (
                "Administrators must review privileged user accounts every 90 days.",
                "policy_b",
            ),
        ]
    )

    assert len(findings) == 1

    finding = findings[0]

    assert finding.finding_type == FREQUENCY_MISMATCH
    assert finding.severity == "MEDIUM"
    assert "every 30 days" in finding.explanation
    assert "every 90 days" in finding.explanation


def test_large_frequency_mismatch_is_high_severity():
    _, _, findings = _detect(
        [
            (
                "Administrators must review privileged accounts every 7 days.",
                "policy_a",
            ),
            (
                "Administrators must review privileged accounts every 90 days.",
                "policy_b",
            ),
        ]
    )

    assert len(findings) == 1
    assert findings[0].finding_type == FREQUENCY_MISMATCH
    assert findings[0].severity == "HIGH"


def test_identical_requirements_are_redundant():
    _, _, findings = _detect(
        [
            (
                "Security teams should review firewall rules quarterly.",
                "policy_a",
            ),
            (
                "Security teams should review firewall rules quarterly.",
                "policy_b",
            ),
        ]
    )

    assert len(findings) == 1

    finding = findings[0]

    assert finding.finding_type == REDUNDANCY
    assert finding.severity == "LOW"


def test_required_vs_recommended_is_modality_inconsistency():
    _, _, findings = _detect(
        [
            (
                "Employees must review access controls every 30 days.",
                "policy_a",
            ),
            (
                "Employees should review access controls every 30 days.",
                "policy_b",
            ),
        ]
    )

    assert len(findings) == 1

    finding = findings[0]

    assert finding.finding_type == MODALITY_INCONSISTENCY
    assert finding.severity == "MEDIUM"


def test_frequency_mismatch_has_precedence_over_modality_inconsistency():
    _, _, findings = _detect(
        [
            (
                "Employees must review access controls every 30 days.",
                "policy_a",
            ),
            (
                "Employees should review access controls every 90 days.",
                "policy_b",
            ),
        ]
    )

    assert len(findings) == 1
    assert findings[0].finding_type == FREQUENCY_MISMATCH


def test_contradiction_has_precedence_over_frequency_mismatch():
    _, _, findings = _detect(
        [
            (
                "Employees must review access controls every 30 days.",
                "policy_a",
            ),
            (
                "Employees must not review access controls every 90 days.",
                "policy_b",
            ),
        ]
    )

    assert len(findings) == 1
    assert findings[0].finding_type == CONTRADICTION


def test_different_subjects_do_not_create_deterministic_finding():
    _, candidates, findings = _detect(
        [
            (
                "Employees must use multi-factor authentication.",
                "policy_a",
            ),
            (
                "Contractors must not use multi-factor authentication.",
                "policy_b",
            ),
        ],
        similarity_threshold=0.0,
    )

    assert len(candidates) == 1
    assert findings == []


def test_different_scopes_do_not_create_contradiction():
    _, candidates, findings = _detect(
        [
            (
                "Employees must use multi-factor authentication for remote access.",
                "policy_a",
            ),
            (
                "Employees must not use multi-factor authentication for local access.",
                "policy_b",
            ),
        ],
        similarity_threshold=0.0,
    )

    assert len(candidates) == 1
    assert findings == []


def test_different_technologies_do_not_create_contradiction():
    _, candidates, findings = _detect(
        [
            (
                "Administrators must use TLS 1.2.",
                "policy_a",
            ),
            (
                "Administrators must not use TLS 1.0.",
                "policy_b",
            ),
        ],
        similarity_threshold=0.0,
    )

    assert len(candidates) == 1
    assert findings == []


def test_unrelated_candidates_do_not_create_findings():
    obligations = [
        _obligation(
            "Employees must review access controls.",
            policy_id="policy_a",
        ),
        _obligation(
            "Employees must review access logs.",
            policy_id="policy_b",
        ),
    ]

    embedded, warnings = embed_obligations(obligations)

    assert warnings == []

    candidates = generate_candidate_pairs(
        embedded,
        similarity_threshold=0.0,
        top_k_per_obligation=None,
    )

    assert len(candidates) == 1

    findings = detect_findings(
        embedded,
        candidates,
    )

    assert findings == []


def test_finding_ids_are_deterministic():
    _, _, first = _detect(
        [
            (
                "Employees must use multi-factor authentication.",
                "policy_a",
            ),
            (
                "Employees must not use multi-factor authentication.",
                "policy_b",
            ),
        ]
    )

    _, _, second = _detect(
        [
            (
                "Employees must use multi-factor authentication.",
                "policy_a",
            ),
            (
                "Employees must not use multi-factor authentication.",
                "policy_b",
            ),
        ]
    )

    assert first[0].finding_id == second[0].finding_id


def test_findings_are_contract_compliant():
    _, _, findings = _detect(
        [
            (
                "Employees must use multi-factor authentication.",
                "policy_a",
            ),
            (
                "Employees must not use multi-factor authentication.",
                "policy_b",
            ),
        ]
    )

    finding = findings[0]

    dumped = finding.model_dump()

    assert dumped["finding_id"].startswith("finding_")
    assert dumped["finding_type"] == CONTRADICTION
    assert 0.0 <= dumped["confidence"] <= 1.0
    assert 0.0 <= dumped["deterministic_score"] <= 1.0
    assert dumped["llm_verified"] is False
    assert dumped["explanation"]
    assert dumped["evidence"]


def test_detection_is_deterministic():
    sentences = [
        (
            "Employees must use multi-factor authentication.",
            "policy_a",
        ),
        (
            "Employees must not use multi-factor authentication.",
            "policy_b",
        ),
        (
            "Security teams should review firewall rules quarterly.",
            "policy_c",
        ),
        (
            "Security teams should review firewall rules quarterly.",
            "policy_d",
        ),
    ]

    _, _, first = _detect(sentences)
    _, _, second = _detect(sentences)

    assert first == second