from pathlib import Path

import pytest

from engine.ingestion import (
    UnsupportedDocumentTypeError,
    load_policy_document,
    load_policy_documents,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_load_txt_policy():
    policy = load_policy_document(FIXTURES / "access_policy.txt")

    assert policy.title == "Enterprise Access Control Policy"
    assert policy.version == "2.1"
    assert policy.owner == "Information Security"
    assert policy.department == "Cyber Security"
    assert str(policy.effective_date) == "2025-01-01"
    assert str(policy.review_date) == "2027-01-01"
    assert policy.status == "Active"

    assert len(policy.sections) == 2
    assert policy.sections[0].title == "User Access Management"
    assert policy.sections[1].title == "Database Access"


def test_load_markdown_policy():
    policy = load_policy_document(FIXTURES / "network_policy.md")

    assert policy.title == "Enterprise Network Security Policy"
    assert policy.version == "3.0"

    titles = [section.title for section in policy.sections]

    assert "Remote Access" in titles
    assert "Encryption Requirements" in titles


def test_sentences_are_extracted():
    policy = load_policy_document(FIXTURES / "access_policy.txt")

    sentences = [
        sentence
        for section in policy.sections
        for sentence in section.sentences
    ]

    assert (
        "Administrators must review privileged user accounts every 30 days."
        in sentences
    )

    assert (
        "Service accounts must not use interactive login."
        in sentences
    )


def test_missing_metadata_is_preserved_as_none():
    policy = load_policy_document(FIXTURES / "legacy_policy.txt")

    assert policy.owner is None
    assert policy.version == "1.0"


def test_unsupported_extension(tmp_path):
    document = tmp_path / "policy.csv"
    document.write_text("test", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentTypeError):
        load_policy_document(document)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_policy_document(FIXTURES / "does_not_exist.txt")


def test_batch_loading_returns_warnings():
    policies, warnings = load_policy_documents(
        [
            FIXTURES / "access_policy.txt",
            FIXTURES / "does_not_exist.txt",
        ]
    )

    assert len(policies) == 1
    assert len(warnings) == 1


def test_policy_ids_are_deterministic():
    document = FIXTURES / "access_policy.txt"

    first = load_policy_document(document)
    second = load_policy_document(document)

    assert first.policy_id == second.policy_id

    first_section_ids = [section.section_id for section in first.sections]
    second_section_ids = [section.section_id for section in second.sections]

    assert first_section_ids == second_section_ids


def test_bold_markdown_metadata_review_date():
    """Regression test: real policy files use '**Last Reviewed:** YYYY-MM-DD'.

    Ensures the ingestion parser strips markdown bold markers and matches the
    'last reviewed' alias so review_date is a real date object, not None.
    """
    from datetime import date

    policy = load_policy_document(FIXTURES / "bold_metadata_policy.md")

    # Core assertion: review_date must be parsed as a date, never None.
    assert policy.review_date is not None, (
        "review_date was None — '**Last Reviewed:**' format not recognised"
    )
    assert isinstance(policy.review_date, date)
    assert policy.review_date == date(2021, 8, 15)

    # Bonus assertions: other bold-formatted fields should also parse.
    assert policy.version == "v1.0"
    assert policy.department == "Security"
    assert policy.status == "active"