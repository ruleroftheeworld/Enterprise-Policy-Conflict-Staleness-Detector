from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from engine.candidates import CandidatePair
from shared.contracts.policy_analysis import (
    Finding,
    NormalizedObligation,
)


CONTRADICTION = "CONTRADICTION"
FREQUENCY_MISMATCH = "FREQUENCY_MISMATCH"
REDUNDANCY = "REDUNDANCY"
MODALITY_INCONSISTENCY = "MODALITY_INCONSISTENCY"
STALE_POLICY = "STALE_POLICY"
STALE_REFERENCE = "STALE_REFERENCE"


@dataclass(frozen=True)
class FrequencyValue:
    amount: int
    unit: str
    approximate_days: float


_FREQUENCY_TO_DAYS = {
    "minute": 1.0 / (24.0 * 60.0),
    "hour": 1.0 / 24.0,
    "day": 1.0,
    "week": 7.0,
    "month": 30.0,
    "year": 365.0,
}


def _stable_finding_id(
    finding_type: str,
    source_obligation_id: str,
    target_obligation_id: str,
) -> str:
    first, second = sorted(
        (
            source_obligation_id,
            target_obligation_id,
        )
    )

    value = f"{finding_type}:{first}:{second}"

    digest = hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()[:16]

    return f"finding_{digest}"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _rounded(value: float) -> float:
    return round(_clamp(value), 4)


def _same_target(candidate: CandidatePair) -> bool:
    """
    Strict deterministic target alignment.

    We intentionally require action and object equality for direct
    deterministic findings.

    Embedding similarity alone must not create a contradiction.
    """
    return candidate.same_action and candidate.same_object


def _scope_compatible(
    first: NormalizedObligation,
    second: NormalizedObligation,
) -> bool:
    """
    Return True when the two obligations may share the same scope.

    - Both None: genuinely unknown vs unknown — let the LLM decide, treat as
      compatible here so the finding is emitted and enters the LLM band.
    - One None, one populated: asymmetric information; still structurally
      compatible (the populated scope may apply to both), but a -0.10 score
      penalty is applied by _scope_penalty so the finding lands in the LLM
      verification band rather than skewing toward auto-accept.
    - Both populated and equal: compatible.
    - Both populated and different: incompatible (unchanged).
    """
    if first.scope is None and second.scope is None:
        return True

    if first.scope is None or second.scope is None:
        # asymmetric: one side has scope context, the other doesn't
        return True

    return first.scope == second.scope


def _scope_penalty(
    first: NormalizedObligation,
    second: NormalizedObligation,
) -> float:
    """
    Return a deterministic-score penalty when scope information is
    asymmetric (one obligation has a scope, the other doesn't).

    - Both None or both populated: 0.0 (no penalty)
    - One None, one populated: -0.10
    """
    one_none = (first.scope is None) != (second.scope is None)
    return -0.10 if one_none else 0.0


def _technology_compatible(
    first: NormalizedObligation,
    second: NormalizedObligation,
) -> bool:
    first_technologies = set(first.technology)
    second_technologies = set(second.technology)

    if not first_technologies or not second_technologies:
        return True

    return bool(first_technologies & second_technologies)


def _subjects_compatible(
    first: NormalizedObligation,
    second: NormalizedObligation,
) -> bool:
    if first.subject is None or second.subject is None:
        return True

    return first.subject == second.subject


def _parse_frequency(
    frequency: str | None,
) -> FrequencyValue | None:
    if frequency is None:
        return None

    match = re.fullmatch(
        r"every\s+(\d+)\s+"
        r"(minute|minutes|hour|hours|day|days|week|weeks|month|months|year|years)",
        frequency.strip().lower(),
    )

    if match is None:
        return None

    amount = int(match.group(1))
    unit = match.group(2)

    if unit.endswith("s"):
        unit = unit[:-1]

    approximate_days = (
        amount * _FREQUENCY_TO_DAYS[unit]
    )

    return FrequencyValue(
        amount=amount,
        unit=unit,
        approximate_days=approximate_days,
    )


def _frequency_ratio(
    first: FrequencyValue,
    second: FrequencyValue,
) -> float:
    lower = min(
        first.approximate_days,
        second.approximate_days,
    )

    higher = max(
        first.approximate_days,
        second.approximate_days,
    )

    if lower <= 0.0:
        return 1.0

    return higher / lower


def _is_contradiction(
    first: NormalizedObligation,
    second: NormalizedObligation,
    candidate: CandidatePair,
) -> bool:
    if not _same_target(candidate):
        return False

    if not _subjects_compatible(first, second):
        return False

    if not _scope_compatible(first, second):
        return False

    if not _technology_compatible(first, second):
        return False

    return (
        first.negated != second.negated
        and (
            first.modality == "PROHIBITED"
            or second.modality == "PROHIBITED"
        )
    )


def _is_frequency_mismatch(
    first: NormalizedObligation,
    second: NormalizedObligation,
    candidate: CandidatePair,
) -> bool:
    if not _same_target(candidate):
        return False

    if not _subjects_compatible(first, second):
        return False

    if not _scope_compatible(first, second):
        return False

    if not _technology_compatible(first, second):
        return False

    if first.negated or second.negated:
        return False

    first_frequency = _parse_frequency(first.frequency)
    second_frequency = _parse_frequency(second.frequency)

    if first_frequency is None or second_frequency is None:
        return False

    return (
        first_frequency.approximate_days
        != second_frequency.approximate_days
    )


def _is_modality_inconsistency(
    first: NormalizedObligation,
    second: NormalizedObligation,
    candidate: CandidatePair,
) -> bool:
    if not _same_target(candidate):
        return False

    if not _subjects_compatible(first, second):
        return False

    if not _scope_compatible(first, second):
        return False

    if not _technology_compatible(first, second):
        return False

    if first.negated or second.negated:
        return False

    if first.frequency != second.frequency:
        return False

    if first.modality == second.modality:
        return False

    return {
        first.modality,
        second.modality,
    }.issubset(
        {
            "REQUIRED",
            "RECOMMENDED",
            "OPTIONAL",
        }
    )


def _is_redundancy(
    first: NormalizedObligation,
    second: NormalizedObligation,
    candidate: CandidatePair,
) -> bool:
    if not _same_target(candidate):
        return False

    if not _subjects_compatible(first, second):
        return False

    if not _scope_compatible(first, second):
        return False

    if not _technology_compatible(first, second):
        return False

    if first.corpus_frequency is not None and first.corpus_frequency >= 5:
        return False

    if second.corpus_frequency is not None and second.corpus_frequency >= 5:
        return False

    return (
        first.modality == second.modality
        and first.negated == second.negated
        and first.frequency == second.frequency
        and candidate.cosine_similarity >= 0.95
    )


def _severity_for_contradiction(
    first: NormalizedObligation,
    second: NormalizedObligation,
) -> str:
    modalities = {
        first.modality,
        second.modality,
    }

    if "PROHIBITED" in modalities and "REQUIRED" in modalities:
        return "CRITICAL"

    return "HIGH"


def _severity_for_frequency_mismatch(
    first: NormalizedObligation,
    second: NormalizedObligation,
) -> str:
    first_frequency = _parse_frequency(first.frequency)
    second_frequency = _parse_frequency(second.frequency)

    if first_frequency is None or second_frequency is None:
        return "MEDIUM"

    ratio = _frequency_ratio(
        first_frequency,
        second_frequency,
    )

    if ratio >= 4.0:
        return "HIGH"

    if ratio >= 2.0:
        return "MEDIUM"

    return "LOW"


def _deterministic_score(
    *,
    base_score: float,
    candidate: CandidatePair,
    first: NormalizedObligation,
    second: NormalizedObligation,
) -> float:
    structural_score = sum(
        (
            candidate.same_subject,
            candidate.same_action,
            candidate.same_object,
            candidate.same_scope,
            candidate.same_frequency,
        )
    ) / 5.0

    confidence_score = (
        first.confidence + second.confidence
    ) / 2.0

    score = (
        base_score
        + 0.10 * candidate.cosine_similarity
        + 0.05 * structural_score
        + 0.05 * confidence_score
        + _scope_penalty(first, second)  # -0.10 when scope is asymmetric
    )

    return _rounded(score)


def _finding_confidence(
    candidate: CandidatePair,
    first: NormalizedObligation,
    second: NormalizedObligation,
) -> float:
    extraction_confidence = (
        first.confidence + second.confidence
    ) / 2.0

    confidence = (
        0.65 * candidate.cosine_similarity
        + 0.35 * extraction_confidence
    )

    return _rounded(confidence)


def _build_evidence(
    first: NormalizedObligation,
    second: NormalizedObligation,
    candidate: CandidatePair,
) -> dict:
    return {
        "source_policy_id": first.policy_id,
        "target_policy_id": second.policy_id,
        "source_section_id": first.section_id,
        "target_section_id": second.section_id,
        "source_sentence": first.sentence_text,
        "target_sentence": second.sentence_text,
        "source_subject": first.subject,
        "target_subject": second.subject,
        "source_action": first.action,
        "target_action": second.action,
        "source_object": first.object,
        "target_object": second.object,
        "source_scope": first.scope,
        "target_scope": second.scope,
        "source_frequency": first.frequency,
        "target_frequency": second.frequency,
        "source_modality": first.modality,
        "target_modality": second.modality,
        "source_negated": first.negated,
        "target_negated": second.negated,
        "source_technology": first.technology,
        "target_technology": second.technology,
        "cosine_similarity": round(
            candidate.cosine_similarity,
            4,
        ),
    }


def _build_finding(
    *,
    finding_type: str,
    severity: str,
    explanation: str,
    base_score: float,
    first: NormalizedObligation,
    second: NormalizedObligation,
    candidate: CandidatePair,
) -> Finding:
    return Finding(
        finding_id=_stable_finding_id(
            finding_type,
            first.obligation_id,
            second.obligation_id,
        ),
        finding_type=finding_type,
        source_obligation_id=first.obligation_id,
        target_obligation_id=second.obligation_id,
        severity=severity,
        confidence=_finding_confidence(
            candidate,
            first,
            second,
        ),
        deterministic_score=_deterministic_score(
            base_score=base_score,
            candidate=candidate,
            first=first,
            second=second,
        ),
        llm_verified=False,
        explanation=explanation,
        evidence=_build_evidence(
            first,
            second,
            candidate,
        ),
    )


def detect_candidate(
    obligations: list[NormalizedObligation],
    candidate: CandidatePair,
) -> Finding | None:
    first = obligations[candidate.source_index]
    second = obligations[candidate.target_index]

    if _is_contradiction(first, second, candidate):
        return _build_finding(
            finding_type=CONTRADICTION,
            severity=_severity_for_contradiction(
                first,
                second,
            ),
            explanation=(
                "The obligations apply to the same normalized target "
                "but impose opposing required/prohibited behavior."
            ),
            base_score=0.80,
            first=first,
            second=second,
            candidate=candidate,
        )

    if _is_frequency_mismatch(first, second, candidate):
        return _build_finding(
            finding_type=FREQUENCY_MISMATCH,
            severity=_severity_for_frequency_mismatch(
                first,
                second,
            ),
            explanation=(
                f"The obligations require the same normalized action "
                f"on the same target at different frequencies: "
                f"'{first.frequency}' versus '{second.frequency}'."
            ),
            base_score=0.65,
            first=first,
            second=second,
            candidate=candidate,
        )

    if _is_modality_inconsistency(first, second, candidate):
        return _build_finding(
            finding_type=MODALITY_INCONSISTENCY,
            severity="MEDIUM",
            explanation=(
                "The obligations describe the same normalized target "
                "but assign different requirement strengths."
            ),
            base_score=0.55,
            first=first,
            second=second,
            candidate=candidate,
        )

    if _is_redundancy(first, second, candidate):
        return _build_finding(
            finding_type=REDUNDANCY,
            severity="LOW",
            explanation=(
                "The obligations express the same normalized requirement "
                "with high semantic similarity."
            ),
            base_score=0.45,
            first=first,
            second=second,
            candidate=candidate,
        )

    return None


def detect_findings(
    obligations: list[NormalizedObligation],
    candidates: list[CandidatePair],
) -> list[Finding]:
    findings = []

    seen_finding_ids = set()

    for candidate in candidates:
        finding = detect_candidate(
            obligations,
            candidate,
        )

        if finding is None:
            continue

        if finding.finding_id in seen_finding_ids:
            continue

        seen_finding_ids.add(finding.finding_id)
        findings.append(finding)

    findings.sort(
        key=lambda finding: (
            -finding.deterministic_score,
            finding.finding_type,
            finding.finding_id,
        )
    )

    return findings


def compute_corpus_frequencies(
    obligations: list[NormalizedObligation],
) -> dict[str, int]:
    from engine.candidates.candidate_generator import cosine_similarity

    normalized_texts = {}
    for obl in obligations:
        text = obl.sentence_text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s]", "", text)
        normalized_texts[obl.obligation_id] = text

    obl_to_freq = {}
    for obl_a in obligations:
        matching_policies = {obl_a.policy_id}
        norm_a = normalized_texts[obl_a.obligation_id]
        emb_a = obl_a.embedding

        for obl_b in obligations:
            if obl_b.policy_id == obl_a.policy_id:
                continue
            if obl_b.policy_id in matching_policies:
                continue

            norm_b = normalized_texts[obl_b.obligation_id]
            if norm_a == norm_b:
                matching_policies.add(obl_b.policy_id)
                continue

            if emb_a and obl_b.embedding:
                sim = cosine_similarity(emb_a, obl_b.embedding)
                if sim >= 0.95:
                    matching_policies.add(obl_b.policy_id)

        obl_to_freq[obl_a.obligation_id] = len(matching_policies)

    return obl_to_freq