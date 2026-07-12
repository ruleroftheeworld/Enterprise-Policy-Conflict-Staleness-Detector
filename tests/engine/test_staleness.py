"""Tests for engine/staleness/staleness_detector.py.

Four categories are covered:
1. A policy reviewed more than 18 months ago → STALE_POLICY finding.
2. A policy reviewed less than 18 months ago → no STALE_POLICY finding.
3. A policy whose obligations reference deprecated technology → STALE_REFERENCE finding.
4. A policy with neither stale date nor deprecated references → no findings.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from engine.detection import STALE_POLICY, STALE_REFERENCE
from engine.staleness import detect_staleness
from engine.staleness.staleness_detector import (
    DEPRECATED_TECHNOLOGIES,
    _STALE_THRESHOLD_DAYS,
)
from shared.contracts.policy_analysis import (
    NormalizedObligation,
    NormalizedPolicy,
    PolicySection,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_TODAY = date(2026, 7, 12)  # Pin today for reproducible assertions


def _make_policy(
    *,
    policy_id: str = "policy_test_001",
    title: str = "Test Policy",
    review_date: date | None = None,
) -> NormalizedPolicy:
    """Construct a minimal NormalizedPolicy for testing."""
    return NormalizedPolicy(
        policy_id=policy_id,
        title=title,
        source_file=f"{policy_id}.txt",
        sections=[
            PolicySection(
                section_id="section_001",
                title="Test Section",
                text="Test content.",
                paragraphs=["Test content."],
                sentences=["Test content."],
            )
        ],
        review_date=review_date,
    )


def _make_obligation(
    *,
    obligation_id: str = "ob_001",
    policy_id: str = "policy_test_001",
    sentence_text: str = "All systems must use approved encryption.",
    technology: list[str] | None = None,
) -> NormalizedObligation:
    """Construct a minimal NormalizedObligation for testing."""
    return NormalizedObligation(
        obligation_id=obligation_id,
        policy_id=policy_id,
        section_id="section_001",
        sentence_text=sentence_text,
        technology=technology or [],
        strength=1.0,
        modality="REQUIRED",
        negated=False,
        confidence=0.95,
    )


# ---------------------------------------------------------------------------
# Test 1: Policy reviewed >18 months ago → STALE_POLICY
# ---------------------------------------------------------------------------


def test_stale_policy_by_review_date_produces_finding():
    """A policy whose review_date exceeds the 18-month threshold must yield STALE_POLICY."""
    # Use a date clearly older than the threshold.
    old_review = _TODAY - timedelta(days=_STALE_THRESHOLD_DAYS + 90)
    policy = _make_policy(review_date=old_review)
    obligations = [_make_obligation()]

    findings = detect_staleness(policy, obligations, today=_TODAY)

    stale_findings = [f for f in findings if f.finding_type == STALE_POLICY]
    assert len(stale_findings) == 1, (
        f"Expected exactly 1 STALE_POLICY finding, got {len(stale_findings)}"
    )

    finding = stale_findings[0]
    assert finding.severity == "MEDIUM"
    assert finding.confidence == 1.0
    assert finding.deterministic_score == 1.0
    assert finding.llm_verified is False
    assert finding.source_obligation_id is None
    assert finding.target_obligation_id is None
    assert str(old_review) in finding.explanation
    assert finding.evidence["review_date"] == str(old_review)
    assert finding.evidence["policy_id"] == policy.policy_id
    assert finding.finding_id.startswith("finding_")


def test_stale_policy_finding_id_is_stable():
    """The finding_id must be deterministic for the same policy + review_date."""
    old_review = date(2020, 1, 1)
    policy = _make_policy(review_date=old_review)
    obligations = [_make_obligation()]

    result_a = detect_staleness(policy, obligations, today=_TODAY)
    result_b = detect_staleness(policy, obligations, today=_TODAY)

    ids_a = {f.finding_id for f in result_a if f.finding_type == STALE_POLICY}
    ids_b = {f.finding_id for f in result_b if f.finding_type == STALE_POLICY}

    assert ids_a == ids_b, "STALE_POLICY finding_id must be stable across calls"


# ---------------------------------------------------------------------------
# Test 2: Policy reviewed recently → no STALE_POLICY
# ---------------------------------------------------------------------------


def test_recent_review_date_produces_no_stale_policy_finding():
    """A policy reviewed within the threshold window must not yield STALE_POLICY."""
    recent_review = _TODAY - timedelta(days=_STALE_THRESHOLD_DAYS - 30)
    policy = _make_policy(review_date=recent_review)
    obligations = [_make_obligation()]

    findings = detect_staleness(policy, obligations, today=_TODAY)

    stale_policy_findings = [f for f in findings if f.finding_type == STALE_POLICY]
    assert stale_policy_findings == [], (
        f"Expected no STALE_POLICY findings for a recently-reviewed policy, "
        f"got {stale_policy_findings}"
    )


def test_future_review_date_produces_no_stale_policy_finding():
    """A policy with a future review_date must not yield STALE_POLICY."""
    future_review = _TODAY + timedelta(days=365)
    policy = _make_policy(review_date=future_review)
    obligations = [_make_obligation()]

    findings = detect_staleness(policy, obligations, today=_TODAY)

    stale_policy_findings = [f for f in findings if f.finding_type == STALE_POLICY]
    assert stale_policy_findings == []


def test_none_review_date_produces_no_stale_policy_finding():
    """A policy with no review_date must not yield STALE_POLICY (not enough info)."""
    policy = _make_policy(review_date=None)
    obligations = [_make_obligation()]

    findings = detect_staleness(policy, obligations, today=_TODAY)

    stale_policy_findings = [f for f in findings if f.finding_type == STALE_POLICY]
    assert stale_policy_findings == []


# ---------------------------------------------------------------------------
# Test 3: Obligation references deprecated technology → STALE_REFERENCE
# ---------------------------------------------------------------------------


def test_deprecated_technology_in_technology_list_produces_finding():
    """An obligation with 'TLS 1.0' in its technology list must yield STALE_REFERENCE."""
    policy = _make_policy()
    obligation = _make_obligation(technology=["TLS 1.0"])

    findings = detect_staleness(policy, [obligation], today=_TODAY)

    ref_findings = [f for f in findings if f.finding_type == STALE_REFERENCE]
    assert len(ref_findings) == 1

    finding = ref_findings[0]
    assert finding.severity == "HIGH"
    assert finding.confidence == 1.0
    assert finding.deterministic_score == 1.0
    assert finding.source_obligation_id == obligation.obligation_id
    assert finding.target_obligation_id is None
    assert "TLS 1.0" in finding.explanation
    assert finding.evidence["deprecated_term"] == "TLS 1.0"
    assert finding.evidence["obligation_id"] == obligation.obligation_id


def test_deprecated_technology_in_sentence_text_produces_finding():
    """An obligation with a deprecated term in its structured technology list must yield
    STALE_REFERENCE.

    Note: prior to task-5b, staleness detection scanned sentence_text directly.
    It now relies exclusively on the structured technology field, which is populated
    by the extractor (obligation_extractor.py) from sentence patterns including
    '(Reference: X)' citations. This test verifies the structured-field path.
    """
    policy = _make_policy()
    # SHA-1 is in the structured technology list (as the extractor would populate it).
    obligation = _make_obligation(
        sentence_text="All applications must use SHA-1 for code signing.",
        technology=["SHA-1"],
    )

    findings = detect_staleness(policy, [obligation], today=_TODAY)

    ref_findings = [f for f in findings if f.finding_type == STALE_REFERENCE]
    assert len(ref_findings) == 1
    assert ref_findings[0].evidence["deprecated_term"] == "SHA-1"


def test_deprecated_technology_case_insensitive_sentence_match():
    """Structured technology list matching is case-insensitive.

    The normalizer lowercases technology names for lookup, so 'tls 1.0' in the
    structured list should match the canonical 'TLS 1.0' deprecated entry.
    """
    policy = _make_policy()
    obligation = _make_obligation(
        sentence_text="Systems may use tls 1.0 for legacy integrations.",
        technology=["TLS 1.0"],  # extractor canonicalises to 'TLS 1.0'
    )

    findings = detect_staleness(policy, [obligation], today=_TODAY)

    ref_findings = [f for f in findings if f.finding_type == STALE_REFERENCE]
    assert len(ref_findings) == 1


def test_deprecated_technology_no_false_positive_on_superstring():
    """'SHA-128' must NOT match 'SHA-1' (word-boundary guard)."""
    policy = _make_policy()
    obligation = _make_obligation(
        sentence_text="Systems must use SHA-128 for hashing.",
        technology=[],
    )

    findings = detect_staleness(policy, [obligation], today=_TODAY)

    ref_findings = [f for f in findings if f.finding_type == STALE_REFERENCE]
    sha1_hits = [f for f in ref_findings if f.evidence.get("deprecated_term") == "SHA-1"]
    assert sha1_hits == [], "SHA-128 must not trigger a SHA-1 STALE_REFERENCE finding"


def test_multiple_deprecated_terms_produce_separate_findings():
    """Multiple deprecated terms in different obligations -> one finding each."""
    policy = _make_policy()
    ob1 = _make_obligation(
        obligation_id="ob_001",
        sentence_text="Use TLS 1.0 for external connections.",
        technology=["TLS 1.0"],  # as populated by the extractor
    )
    ob2 = _make_obligation(
        obligation_id="ob_002",
        sentence_text="Sign packages with SHA-1.",
        technology=["SHA-1"],  # as populated by the extractor
    )

    findings = detect_staleness(policy, [ob1, ob2], today=_TODAY)

    ref_findings = [f for f in findings if f.finding_type == STALE_REFERENCE]
    assert len(ref_findings) == 2

    terms = {f.evidence["deprecated_term"] for f in ref_findings}
    assert "TLS 1.0" in terms
    assert "SHA-1" in terms


def test_stale_reference_finding_id_is_stable():
    """STALE_REFERENCE finding_id must be deterministic."""
    policy = _make_policy()
    obligation = _make_obligation(technology=["MD5"])

    result_a = detect_staleness(policy, [obligation], today=_TODAY)
    result_b = detect_staleness(policy, [obligation], today=_TODAY)

    ids_a = {f.finding_id for f in result_a if f.finding_type == STALE_REFERENCE}
    ids_b = {f.finding_id for f in result_b if f.finding_type == STALE_REFERENCE}

    assert ids_a == ids_b


def test_obligation_from_different_policy_is_not_checked():
    """Obligations belonging to a different policy must not trigger findings on the tested policy."""
    policy = _make_policy(policy_id="policy_a")
    other_policy_obligation = _make_obligation(
        policy_id="policy_b",  # different policy
        technology=["TLS 1.0"],
    )

    findings = detect_staleness(policy, [other_policy_obligation], today=_TODAY)

    assert findings == [], (
        "Obligations from other policies must not contribute to the tested policy's findings"
    )


# ---------------------------------------------------------------------------
# Test 4: Clean policy → no findings at all
# ---------------------------------------------------------------------------


def test_clean_policy_produces_no_findings():
    """A recently-reviewed policy with no deprecated references must produce zero findings."""
    recent_review = _TODAY - timedelta(days=30)
    policy = _make_policy(review_date=recent_review)
    obligation = _make_obligation(
        sentence_text="All systems must use TLS 1.3 for network communications.",
        technology=["TLS 1.3"],
    )

    findings = detect_staleness(policy, [obligation], today=_TODAY)

    assert findings == [], (
        f"Expected zero findings for a clean policy, got {findings}"
    )


def test_both_stale_date_and_deprecated_reference_produce_combined_findings():
    """A policy that is both stale by date AND references deprecated tech gets both finding types."""
    old_review = date(2020, 6, 1)
    policy = _make_policy(review_date=old_review)
    obligation = _make_obligation(technology=["SSLv3"])

    findings = detect_staleness(policy, [obligation], today=_TODAY)

    finding_types = {f.finding_type for f in findings}
    assert STALE_POLICY in finding_types
    assert STALE_REFERENCE in finding_types


# ---------------------------------------------------------------------------
# Test 5: Constant integrity checks
# ---------------------------------------------------------------------------


def test_deprecated_technologies_is_a_frozenset():
    """DEPRECATED_TECHNOLOGIES must be a frozenset so it is immutable."""
    assert isinstance(DEPRECATED_TECHNOLOGIES, frozenset)


def test_deprecated_technologies_contains_expected_items():
    """Spot-check that the required deprecated items are present."""
    required = {
        "TLS 1.0",
        "TLS 1.1",
        "SHA-1",
        "MD5",
        "DES",
        "SSLv3",
        "Windows Server 2012",
        "Windows Server 2008",
        "NIST SP 800-53 Rev 4",
        "NIST SP 800-63A",
        "NIST SP 800-63B",
        "PCI DSS v3.2",
    }
    missing = required - DEPRECATED_TECHNOLOGIES
    assert not missing, f"Missing deprecated technology entries: {missing}"
