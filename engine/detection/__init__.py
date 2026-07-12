from .deterministic_detector import (
    CONTRADICTION,
    FREQUENCY_MISMATCH,
    MODALITY_INCONSISTENCY,
    REDUNDANCY,
    STALE_POLICY,
    STALE_REFERENCE,
    detect_candidate,
    detect_findings,
    compute_corpus_frequencies,
)

__all__ = [
    "CONTRADICTION",
    "FREQUENCY_MISMATCH",
    "MODALITY_INCONSISTENCY",
    "REDUNDANCY",
    "STALE_POLICY",
    "STALE_REFERENCE",
    "detect_candidate",
    "detect_findings",
    "compute_corpus_frequencies",
]