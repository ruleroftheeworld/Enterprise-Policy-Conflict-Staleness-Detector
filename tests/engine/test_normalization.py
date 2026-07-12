from engine.extraction import extract_obligation
from engine.normalization import (
    normalize_obligation,
    normalize_obligations,
)


def _extract(sentence: str):
    obligation = extract_obligation(
        policy_id="policy_test",
        section_id="section_test",
        sentence=sentence,
    )

    assert obligation is not None

    return obligation


def test_normalizes_subject_case():
    obligation = _extract(
        "Administrators must review privileged user accounts."
    )

    normalized = normalize_obligation(obligation)

    assert normalized.subject == "administrators"


def test_normalizes_action_synonym():
    obligation = _extract(
        "Employees must utilize approved authentication controls."
    )

    normalized = normalize_obligation(obligation)

    assert normalized.action == "use"


def test_normalizes_scope():
    obligation = _extract(
        "Employees must use multi-factor authentication for remote access."
    )

    normalized = normalize_obligation(obligation)

    assert normalized.scope == "remote access"


def test_normalizes_frequency():
    obligation = _extract(
        "Administrators must review privileged accounts every 30 days."
    )

    normalized = normalize_obligation(obligation)

    assert normalized.frequency == "every 30 days"


def test_normalizes_quarterly_frequency():
    obligation = _extract(
        "Security teams should review firewall rules quarterly."
    )

    normalized = normalize_obligation(obligation)

    assert normalized.frequency == "every 3 months"


def test_removes_technology_from_object():
    obligation = _extract(
        "Database administrators must encrypt database backups using AES-256."
    )

    normalized = normalize_obligation(obligation)

    assert normalized.object == "database backups"
    assert normalized.technology == ["AES-256"]


def test_normalizes_prohibited_modality():
    obligation = _extract(
        "Employees must not use multi-factor authentication."
    )

    normalized = normalize_obligation(obligation)

    assert normalized.modality == "PROHIBITED"
    assert normalized.negated is True
    assert normalized.strength == 1.0


def test_preserves_sentence_evidence():
    sentence = "Employees must use multi-factor authentication for remote access."

    obligation = _extract(sentence)
    normalized = normalize_obligation(obligation)

    assert normalized.sentence_text == sentence


def test_preserves_obligation_id():
    obligation = _extract(
        "Employees must use multi-factor authentication."
    )

    normalized = normalize_obligation(obligation)

    assert normalized.obligation_id == obligation.obligation_id


def test_does_not_mutate_original():
    obligation = _extract(
        "Administrators must review privileged accounts every 30 days."
    )

    original_subject = obligation.subject
    original_frequency = obligation.frequency

    normalize_obligation(obligation)

    assert obligation.subject == original_subject
    assert obligation.frequency == original_frequency


def test_batch_normalization():
    obligations = [
        _extract("Employees must use multi-factor authentication."),
        _extract("Security teams should review firewall rules quarterly."),
    ]

    normalized = normalize_obligations(obligations)

    assert len(normalized) == 2
    assert normalized[0].subject == "employees"
    assert normalized[1].frequency == "every 3 months"