from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from shared.contracts.policy_analysis import NormalizedObligation


DEFAULT_SIMILARITY_THRESHOLD = 0.55
DEFAULT_TOP_K_PER_OBLIGATION = 20


@dataclass(frozen=True)
class CandidatePair:
    source_index: int
    target_index: int
    source_obligation_id: str
    target_obligation_id: str
    cosine_similarity: float

    same_policy: bool
    same_subject: bool
    same_action: bool
    same_object: bool
    same_scope: bool
    same_frequency: bool

    modality_mismatch: bool
    negation_mismatch: bool

    @property
    def pair_key(self) -> tuple[str, str]:
        return tuple(
            sorted(
                (
                    self.source_obligation_id,
                    self.target_obligation_id,
                )
            )
        )


def cosine_similarity(
    first: list[float],
    second: list[float],
) -> float:
    if not first or not second:
        return 0.0

    if len(first) != len(second):
        raise ValueError(
            "Embedding dimensions must match: "
            f"{len(first)} != {len(second)}"
        )

    first_array = np.asarray(first, dtype=np.float32)
    second_array = np.asarray(second, dtype=np.float32)

    first_norm = float(np.linalg.norm(first_array))
    second_norm = float(np.linalg.norm(second_array))

    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0

    similarity = float(
        np.dot(first_array, second_array)
        / (first_norm * second_norm)
    )

    return max(-1.0, min(1.0, similarity))


def _same_optional_text(
    first: str | None,
    second: str | None,
) -> bool:
    return first == second


def _has_semantic_anchor(
    first: NormalizedObligation,
    second: NormalizedObligation,
) -> bool:
    """
    Cheap deterministic blocking.

    At least one strong structural field must align before cosine
    similarity is calculated.

    Subject equality alone is intentionally not enough.
    """
    same_action = (
        first.action is not None
        and first.action == second.action
    )

    same_object = (
        first.object is not None
        and first.object == second.object
    )

    shared_technology = bool(
        set(first.technology) & set(second.technology)
    )

    return same_action or same_object or shared_technology


def _is_exact_duplicate_evidence(
    first: NormalizedObligation,
    second: NormalizedObligation,
) -> bool:
    return (
        first.policy_id == second.policy_id
        and first.section_id == second.section_id
        and first.sentence_text == second.sentence_text
    )


def _candidate_priority(
    candidate: CandidatePair,
) -> tuple:
    """
    Stable ranking.

    Conflict-shaped pairs come first, followed by stronger semantic
    similarity and stronger structural alignment.
    """
    structural_matches = sum(
        (
            candidate.same_subject,
            candidate.same_action,
            candidate.same_object,
            candidate.same_scope,
            candidate.same_frequency,
        )
    )

    return (
        int(candidate.modality_mismatch),
        int(candidate.negation_mismatch),
        candidate.cosine_similarity,
        structural_matches,
        candidate.pair_key,
    )


def generate_candidate_pairs(
    obligations: list[NormalizedObligation],
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    top_k_per_obligation: int | None = DEFAULT_TOP_K_PER_OBLIGATION,
    include_same_policy: bool = True,
) -> list[CandidatePair]:
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError(
            "similarity_threshold must be between 0.0 and 1.0"
        )

    if top_k_per_obligation is not None and top_k_per_obligation <= 0:
        raise ValueError(
            "top_k_per_obligation must be positive or None"
        )

    if len(obligations) < 2:
        return []

    provisional: list[CandidatePair] = []

    for source_index, target_index in combinations(
        range(len(obligations)),
        2,
    ):
        source = obligations[source_index]
        target = obligations[target_index]

        if (
            not include_same_policy
            and source.policy_id == target.policy_id
        ):
            continue

        if _is_exact_duplicate_evidence(source, target):
            continue

        if getattr(source, "topic", None) != getattr(target, "topic", None):
            continue

        if not _has_semantic_anchor(source, target):
            continue

        similarity = cosine_similarity(
            source.embedding,
            target.embedding,
        )

        if similarity < similarity_threshold:
            continue

        provisional.append(
            CandidatePair(
                source_index=source_index,
                target_index=target_index,
                source_obligation_id=source.obligation_id,
                target_obligation_id=target.obligation_id,
                cosine_similarity=similarity,
                same_policy=source.policy_id == target.policy_id,
                same_subject=source.subject == target.subject,
                same_action=source.action == target.action,
                same_object=source.object == target.object,
                same_scope=_same_optional_text(
                    source.scope,
                    target.scope,
                ),
                same_frequency=_same_optional_text(
                    source.frequency,
                    target.frequency,
                ),
                modality_mismatch=source.modality != target.modality,
                negation_mismatch=source.negated != target.negated,
            )
        )

    provisional.sort(
        key=_candidate_priority,
        reverse=True,
    )

    if top_k_per_obligation is None:
        return provisional

    selected: list[CandidatePair] = []
    counts = [0] * len(obligations)

    for candidate in provisional:
        if (
            counts[candidate.source_index] >= top_k_per_obligation
            or counts[candidate.target_index] >= top_k_per_obligation
        ):
            continue

        selected.append(candidate)
        counts[candidate.source_index] += 1
        counts[candidate.target_index] += 1

    selected.sort(
        key=lambda candidate: (
            -candidate.cosine_similarity,
            candidate.pair_key,
        )
    )

    return selected