from pathlib import Path

from engine.extraction import extract_policy_obligations
from engine.ingestion import load_policy_document
from engine.normalization import normalize_obligations


FIXTURES = Path(__file__).parent / "fixtures"


def _load_normalized(filename: str):
    policy = load_policy_document(FIXTURES / filename)

    obligations = extract_policy_obligations(policy)

    return policy, normalize_obligations(obligations)


def _find_obligation(
    obligations,
    *,
    subject=None,
    action=None,
    object_text=None,
    scope=None,
    frequency=None,
    modality=None,
):
    matches = []

    for obligation in obligations:
        if subject is not None and obligation.subject != subject:
            continue

        if action is not None and obligation.action != action:
            continue

        if object_text is not None and obligation.object != object_text:
            continue

        if scope is not None and obligation.scope != scope:
            continue

        if frequency is not None and obligation.frequency != frequency:
            continue

        if modality is not None and obligation.modality != modality:
            continue

        matches.append(obligation)

    assert len(matches) == 1, (
        f"Expected exactly one obligation but found {len(matches)}. "
        f"Filters: subject={subject}, action={action}, "
        f"object={object_text}, scope={scope}, "
        f"frequency={frequency}, modality={modality}"
    )

    return matches[0]


def test_mfa_required_and_prohibited_align_semantically():
    _, access_obligations = _load_normalized("access_policy.txt")
    _, conflicting_obligations = _load_normalized("conflicting_policy.txt")

    required = _find_obligation(
        access_obligations,
        subject="employees",
        action="use",
        object_text="multi-factor authentication",
        scope="remote access",
        modality="REQUIRED",
    )

    prohibited = _find_obligation(
        conflicting_obligations,
        subject="employees",
        action="use",
        object_text="multi-factor authentication",
        scope="remote access",
        modality="PROHIBITED",
    )

    assert required.action == prohibited.action
    assert required.object == prohibited.object
    assert required.scope == prohibited.scope
    assert required.modality != prohibited.modality
    assert required.negated is False
    assert prohibited.negated is True


def test_repeated_privileged_review_obligations_align():
    _, access_obligations = _load_normalized("access_policy.txt")
    _, network_obligations = _load_normalized("network_policy.md")

    first = _find_obligation(
        access_obligations,
        subject="administrators",
        action="review",
        object_text="privileged user accounts",
        frequency="every 30 days",
        modality="REQUIRED",
    )

    second = _find_obligation(
        network_obligations,
        subject="administrators",
        action="review",
        object_text="privileged user accounts",
        frequency="every 30 days",
        modality="REQUIRED",
    )

    assert first.action == second.action
    assert first.object == second.object
    assert first.frequency == second.frequency
    assert first.modality == second.modality

    # IDs must remain different because evidence comes from different policies.
    assert first.obligation_id != second.obligation_id


def test_frequency_mismatch_remains_visible_after_normalization():
    _, access_obligations = _load_normalized("access_policy.txt")
    _, conflicting_obligations = _load_normalized("conflicting_policy.txt")

    every_30_days = _find_obligation(
        access_obligations,
        subject="administrators",
        action="review",
        object_text="privileged user accounts",
        frequency="every 30 days",
    )

    every_90_days = _find_obligation(
        conflicting_obligations,
        subject="administrators",
        action="review",
        object_text="privileged user accounts",
        frequency="every 90 days",
    )

    assert every_30_days.action == every_90_days.action
    assert every_30_days.object == every_90_days.object
    assert every_30_days.frequency != every_90_days.frequency


def test_database_encryption_conflict_aligns_after_normalization():
    _, access_obligations = _load_normalized("access_policy.txt")
    _, conflicting_obligations = _load_normalized("conflicting_policy.txt")

    required = _find_obligation(
        access_obligations,
        subject="database administrators",
        action="encrypt",
        object_text="database backups",
        modality="REQUIRED",
    )

    prohibited = _find_obligation(
        conflicting_obligations,
        subject="database administrators",
        action="encrypt",
        object_text="database backups",
        modality="PROHIBITED",
    )

    assert required.object == prohibited.object
    assert required.technology == prohibited.technology == ["AES-256"]
    assert required.modality != prohibited.modality


def test_firewall_review_obligations_align_for_redundancy():
    _, network_obligations = _load_normalized("network_policy.md")
    _, conflicting_obligations = _load_normalized("conflicting_policy.txt")

    first = _find_obligation(
        network_obligations,
        subject="security teams",
        action="review",
        object_text="firewall rules",
        frequency="every 3 months",
    )

    second = _find_obligation(
        conflicting_obligations,
        subject="security teams",
        action="review",
        object_text="firewall rules",
        frequency="every 3 months",
    )

    assert first.action == second.action
    assert first.object == second.object
    assert first.frequency == second.frequency


def test_legacy_policy_retains_deprecated_technologies():
    _, obligations = _load_normalized("legacy_policy.txt")

    technologies = {
        technology
        for obligation in obligations
        for technology in obligation.technology
    }

    assert "TLS 1.0" in technologies
    assert "SHA-1" in technologies
    assert "Windows Server 2012" in technologies


def test_policy_metadata_survives_ingestion():
    policy, _ = _load_normalized("legacy_policy.txt")

    assert policy.title == "Legacy Infrastructure Security Policy"
    assert policy.version == "1.0"
    assert policy.owner is None
    assert str(policy.review_date) == "2021-01-01"


def test_unrelated_obligations_remain_distinct():
    _, access_obligations = _load_normalized("access_policy.txt")

    mfa = _find_obligation(
        access_obligations,
        subject="employees",
        action="use",
        object_text="multi-factor authentication",
    )

    database_encryption = _find_obligation(
        access_obligations,
        subject="database administrators",
        action="encrypt",
        object_text="database backups",
    )

    assert mfa.action != database_encryption.action
    assert mfa.object != database_encryption.object


def test_original_sentence_evidence_survives_full_pre_embedding_pipeline():
    _, obligations = _load_normalized("conflicting_policy.txt")

    obligation = _find_obligation(
        obligations,
        subject="employees",
        action="use",
        object_text="multi-factor authentication",
        modality="PROHIBITED",
    )

    assert obligation.sentence_text == (
        "Employees must not use multi-factor authentication for remote access."
    )


def test_obligation_ids_are_stable_across_pipeline_runs():
    _, first_run = _load_normalized("access_policy.txt")
    _, second_run = _load_normalized("access_policy.txt")

    first_ids = [obligation.obligation_id for obligation in first_run]
    second_ids = [obligation.obligation_id for obligation in second_run]

    assert first_ids == second_ids