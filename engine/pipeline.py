from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from engine.candidates import generate_candidate_pairs
from engine.detection import detect_findings
from engine.embeddings import embed_obligations
from engine.extraction import extract_policy_obligations
from engine.ingestion import load_policy_document
from engine.normalization import normalize_obligations
from shared.contracts.policy_analysis import (
    AnalysisResult,
    NormalizedObligation,
    NormalizedPolicy,
)


DocumentInput = str | Path


def _warning(
    *,
    stage: str,
    document: str | None,
    message: str,
) -> str:
    location = f" document={document!r}" if document is not None else ""

    return f"[{stage}]{location}: {message}"


def _safe_error_message(exc: Exception) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _normalize_documents(
    documents: Iterable[DocumentInput],
) -> list[DocumentInput]:
    if isinstance(documents, (str, Path)):
        return [documents]

    try:
        normalized = list(documents)
    except TypeError as exc:
        raise TypeError(
            "documents must be a path or an iterable of paths"
        ) from exc

    for index, document in enumerate(normalized):
        if not isinstance(document, (str, Path)):
            raise TypeError(
                "each document must be a str or pathlib.Path; "
                f"received {type(document).__name__} at index {index}"
            )

    return normalized


def _process_document(
    document: DocumentInput,
) -> tuple[
    NormalizedPolicy | None,
    list[NormalizedObligation],
    list[str],
]:
    warnings: list[str] = []
    document_name = str(document)

    try:
        policy = load_policy_document(document)
    except Exception as exc:
        warnings.append(
            _warning(
                stage="ingestion",
                document=document_name,
                message=_safe_error_message(exc),
            )
        )
        return None, [], warnings

    try:
        extracted = extract_policy_obligations(policy)
    except Exception as exc:
        warnings.append(
            _warning(
                stage="extraction",
                document=document_name,
                message=_safe_error_message(exc),
            )
        )
        return policy, [], warnings

    try:
        obligations = normalize_obligations(extracted)
    except Exception as exc:
        warnings.append(
            _warning(
                stage="normalization",
                document=document_name,
                message=_safe_error_message(exc),
            )
        )
        return policy, [], warnings

    return policy, obligations, warnings


def _statistics(
    *,
    documents_received: int,
    policies: list[NormalizedPolicy],
    obligations: list[NormalizedObligation],
    candidate_count: int,
    findings: list,
) -> dict[str, Any]:
    finding_types: dict[str, int] = {}
    severities: dict[str, int] = {}

    for finding in findings:
        finding_types[finding.finding_type] = (
            finding_types.get(finding.finding_type, 0) + 1
        )

        severities[finding.severity] = (
            severities.get(finding.severity, 0) + 1
        )

    return {
        "documents_received": documents_received,
        "policies_processed": len(policies),
        "documents_failed": documents_received - len(policies),
        "obligations_extracted": len(obligations),
        "candidate_pairs": candidate_count,
        "findings_total": len(findings),
        "findings_by_type": dict(sorted(finding_types.items())),
        "findings_by_severity": dict(sorted(severities.items())),
    }


def analyze_policy_documents(
    documents: Iterable[DocumentInput] | DocumentInput,
) -> AnalysisResult:
    started = time.perf_counter()

    document_list = _normalize_documents(documents)

    policies: list[NormalizedPolicy] = []
    obligations: list[NormalizedObligation] = []
    warnings: list[str] = []

    for document in document_list:
        policy, document_obligations, document_warnings = (
            _process_document(document)
        )

        warnings.extend(document_warnings)

        if policy is not None:
            policies.append(policy)

        obligations.extend(document_obligations)

    embedded_obligations = obligations

    if obligations:
        try:
            embedded_obligations, embedding_warnings = (
                embed_obligations(obligations)
            )

            warnings.extend(
                _warning(
                    stage="embeddings",
                    document=None,
                    message=warning,
                )
                for warning in embedding_warnings
            )

        except Exception as exc:
            warnings.append(
                _warning(
                    stage="embeddings",
                    document=None,
                    message=_safe_error_message(exc),
                )
            )

            embedded_obligations = obligations

    candidates = []

    if embedded_obligations:
        try:
            candidates = generate_candidate_pairs(
                embedded_obligations,
            )
        except Exception as exc:
            warnings.append(
                _warning(
                    stage="candidates",
                    document=None,
                    message=_safe_error_message(exc),
                )
            )

    findings = []

    if candidates:
        try:
            findings = detect_findings(
                embedded_obligations,
                candidates,
            )
        except Exception as exc:
            warnings.append(
                _warning(
                    stage="detection",
                    document=None,
                    message=_safe_error_message(exc),
                )
            )

    statistics = _statistics(
        documents_received=len(document_list),
        policies=policies,
        obligations=embedded_obligations,
        candidate_count=len(candidates),
        findings=findings,
    )

    processing_time_ms = (
        time.perf_counter() - started
    ) * 1000.0

    return AnalysisResult(
        policies=policies,
        obligations=embedded_obligations,
        findings=findings,
        statistics=statistics,
        warnings=warnings,
        processing_time_ms=processing_time_ms,
    )


__all__ = [
    "analyze_policy_documents",
]