from pathlib import Path

from engine.extraction import (
    extract_obligation,
    extract_policy_obligations,
)
from engine.ingestion import load_policy_document


FIXTURES = Path(__file__).parent / "fixtures"


def test_positive_required_obligation():
    obligation = extract_obligation(
        policy_id="policy_test",
        section_id="section_test",
        sentence="Administrators must review privileged user accounts every 30 days.",
    )

    assert obligation is not None
    assert obligation.modality == "REQUIRED"
    assert obligation.negated is False
    assert obligation.subject == "Administrators"
    assert obligation.action == "review"
    assert obligation.object == "privileged user accounts"
    assert obligation.frequency == "every 30 days"


def test_negative_prohibited_obligation():
    obligation = extract_obligation(
        policy_id="policy_test",
        section_id="section_test",
        sentence="Service accounts must not use interactive login.",
    )

    assert obligation is not None
    assert obligation.modality == "PROHIBITED"
    assert obligation.negated is True
    assert obligation.subject == "Service accounts"
    assert obligation.action == "use"
    assert obligation.object == "interactive login"


def test_shall_not_is_prohibited():
    obligation = extract_obligation(
        policy_id="policy_test",
        section_id="section_test",
        sentence="Database administrators shall not encrypt database backups using AES-256.",
    )

    assert obligation is not None
    assert obligation.modality == "PROHIBITED"
    assert obligation.negated is True
    assert obligation.action == "encrypt"
    assert "AES-256" in obligation.technology


def test_recommended_obligation():
    obligation = extract_obligation(
        policy_id="policy_test",
        section_id="section_test",
        sentence="Security teams should review firewall rules quarterly.",
    )

    assert obligation is not None
    assert obligation.modality == "RECOMMENDED"
    assert obligation.negated is False
    assert obligation.frequency == "quarterly"


def test_optional_obligation():
    obligation = extract_obligation(
        policy_id="policy_test",
        section_id="section_test",
        sentence="Production users may access reporting dashboards during business hours.",
    )

    assert obligation is not None
    assert obligation.modality == "OPTIONAL"
    assert obligation.scope == "during business hours"


def test_required_to_obligation():
    obligation = extract_obligation(
        policy_id="policy_test",
        section_id="section_test",
        sentence="Database administrators are required to encrypt database backups using AES-256.",
    )

    assert obligation is not None
    assert obligation.modality == "REQUIRED"
    assert obligation.subject == "Database administrators"
    assert obligation.action == "encrypt"
    assert "AES-256" in obligation.technology


def test_scope_extraction():
    obligation = extract_obligation(
        policy_id="policy_test",
        section_id="section_test",
        sentence="Employees must use multi-factor authentication for remote access.",
    )

    assert obligation is not None
    assert obligation.scope == "for remote access"
    assert "multi-factor authentication" in obligation.technology


def test_condition_extraction():
    obligation = extract_obligation(
        policy_id="policy_test",
        section_id="section_test",
        sentence="Employees must change passwords when a security incident occurs.",
    )

    assert obligation is not None
    assert obligation.condition == "a security incident occurs"


def test_exception_extraction():
    obligation = extract_obligation(
        policy_id="policy_test",
        section_id="section_test",
        sentence="Employees must use multi-factor authentication unless emergency access is approved.",
    )

    assert obligation is not None
    assert obligation.exception == "emergency access is approved"


def test_non_obligation_returns_none():
    obligation = extract_obligation(
        policy_id="policy_test",
        section_id="section_test",
        sentence="The security policy was published in January.",
    )

    assert obligation is None


def test_obligation_ids_are_deterministic():
    first = extract_obligation(
        policy_id="policy_test",
        section_id="section_test",
        sentence="Employees must use multi-factor authentication for remote access.",
    )

    second = extract_obligation(
        policy_id="policy_test",
        section_id="section_test",
        sentence="Employees must use multi-factor authentication for remote access.",
    )

    assert first is not None
    assert second is not None
    assert first.obligation_id == second.obligation_id


def test_extract_access_policy_obligations():
    policy = load_policy_document(FIXTURES / "access_policy.txt")

    obligations = extract_policy_obligations(policy)

    assert len(obligations) == 6

    modalities = {obligation.modality for obligation in obligations}

    assert "REQUIRED" in modalities
    assert "RECOMMENDED" in modalities
    assert "OPTIONAL" in modalities
    assert "PROHIBITED" in modalities   