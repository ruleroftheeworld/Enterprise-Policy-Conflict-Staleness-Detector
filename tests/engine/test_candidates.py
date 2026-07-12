import numpy as np
import pytest

from engine.candidates import (
    cosine_similarity,
    generate_candidate_pairs,
)
from engine.embeddings import embed_obligations
from engine.extraction import extract_obligation
from engine.normalization import normalize_obligation


def _obligation(
    sentence: str,
    *,
    policy_id: str,
    section_id: str = "section_test",
):
    obligation = extract_obligation(
        policy_id=policy_id,
        section_id=section_id,
        sentence=sentence,
    )

    assert obligation is not None

    return normalize_obligation(obligation)


def _embed(obligations):
    embedded, warnings = embed_obligations(obligations)

    assert warnings == []

    return embedded


def test_cosine_similarity_identical_vectors():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_similarity_dimension_mismatch():
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 0.0])


def test_required_and_prohibited_same_target_become_candidate():
    obligations = _embed(
        [
            _obligation(
                "Employees must use multi-factor authentication for remote access.",
                policy_id="policy_a",
            ),
            _obligation(
                "Employees must not use multi-factor authentication for remote access.",
                policy_id="policy_b",
            ),
        ]
    )

    candidates = generate_candidate_pairs(obligations)

    assert len(candidates) == 1

    candidate = candidates[0]

    assert candidate.same_action is True
    assert candidate.same_object is True
    assert candidate.same_scope is True
    assert candidate.modality_mismatch is True
    assert candidate.negation_mismatch is True


def test_frequency_mismatch_becomes_candidate():
    obligations = _embed(
        [
            _obligation(
                "Administrators must review privileged user accounts every 30 days.",
                policy_id="policy_a",
            ),
            _obligation(
                "Administrators must review privileged user accounts every 90 days.",
                policy_id="policy_b",
            ),
        ]
    )

    candidates = generate_candidate_pairs(
        obligations,
        similarity_threshold=0.40,
    )

    assert len(candidates) == 1
    assert candidates[0].same_action is True
    assert candidates[0].same_object is True
    assert candidates[0].same_frequency is False


def test_redundant_obligations_become_candidate():
    obligations = _embed(
        [
            _obligation(
                "Security teams should review firewall rules quarterly.",
                policy_id="policy_a",
            ),
            _obligation(
                "Security teams should review firewall rules quarterly.",
                policy_id="policy_b",
            ),
        ]
    )

    candidates = generate_candidate_pairs(obligations)

    assert len(candidates) == 1
    assert candidates[0].cosine_similarity == pytest.approx(1.0, abs=1e-5)
    assert candidates[0].modality_mismatch is False


def test_unrelated_obligations_do_not_become_candidates():
    obligations = _embed(
        [
            _obligation(
                "Employees must use multi-factor authentication.",
                policy_id="policy_a",
            ),
            _obligation(
                "Database administrators must encrypt database backups using AES-256.",
                policy_id="policy_b",
            ),
        ]
    )

    candidates = generate_candidate_pairs(
        obligations,
        similarity_threshold=0.0,
    )

    assert candidates == []


def test_same_policy_pairs_can_be_excluded():
    obligations = _embed(
        [
            _obligation(
                "Employees must use multi-factor authentication.",
                policy_id="policy_a",
                section_id="section_a",
            ),
            _obligation(
                "Employees must not use multi-factor authentication.",
                policy_id="policy_a",
                section_id="section_b",
            ),
        ]
    )

    candidates = generate_candidate_pairs(
        obligations,
        include_same_policy=False,
    )

    assert candidates == []


def test_exact_duplicate_evidence_is_skipped():
    obligation = _obligation(
        "Employees must use multi-factor authentication.",
        policy_id="policy_a",
    )

    first = obligation.model_copy(deep=True)
    second = obligation.model_copy(deep=True)

    first.embedding = [1.0, 0.0]
    second.embedding = [1.0, 0.0]

    candidates = generate_candidate_pairs(
        [first, second],
        similarity_threshold=0.0,
    )

    assert candidates == []


def test_top_k_limits_candidates_per_obligation():
    obligations = [
        _obligation(
            "Employees must review access controls.",
            policy_id=f"policy_{index}",
        )
        for index in range(4)
    ]

    dimension = 8
    vector = np.ones(dimension, dtype=np.float32)
    vector /= np.linalg.norm(vector)

    for obligation in obligations:
        obligation.embedding = vector.tolist()

    candidates = generate_candidate_pairs(
        obligations,
        similarity_threshold=0.0,
        top_k_per_obligation=1,
    )

    counts = {index: 0 for index in range(len(obligations))}

    for candidate in candidates:
        counts[candidate.source_index] += 1
        counts[candidate.target_index] += 1

    assert all(count <= 1 for count in counts.values())


def test_candidate_generation_is_deterministic():
    obligations = [
        _obligation(
            "Employees must review access controls.",
            policy_id=f"policy_{index}",
        )
        for index in range(3)
    ]

    for obligation in obligations:
        obligation.embedding = [1.0, 0.0]

    first = generate_candidate_pairs(
        obligations,
        similarity_threshold=0.0,
    )

    second = generate_candidate_pairs(
        obligations,
        similarity_threshold=0.0,
    )

    assert first == second


def test_invalid_similarity_threshold():
    with pytest.raises(ValueError):
        generate_candidate_pairs([], similarity_threshold=1.1)


def test_invalid_top_k():
    with pytest.raises(ValueError):
        generate_candidate_pairs([], top_k_per_obligation=0)