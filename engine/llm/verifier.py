from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from shared.contracts.policy_analysis import Finding, NormalizedObligation


class LLMProvider(Protocol):
    """Minimal provider interface used by the verifier."""

    def generate(self, prompt: str) -> str:
        ...


@dataclass(frozen=True)
class VerificationResult:
    verified: bool
    confidence: float
    explanation: str


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _safe_error_message(exc: Exception) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def build_verification_prompt(
    finding: Finding,
    source: NormalizedObligation,
    target: NormalizedObligation,
) -> str:
    return (
        "You are verifying a deterministic enterprise policy finding.\n"
        "Return JSON only. Do not return markdown.\n\n"
        "Allowed JSON schema:\n"
        '{'
        '"verified": true or false, '
        '"confidence": number from 0.0 to 1.0, '
        '"explanation": "short explanation"'
        '}\n\n'
        f"Finding type: {finding.finding_type}\n"
        f"Deterministic score: {finding.deterministic_score}\n"
        f"Source obligation: {source.sentence_text}\n"
        f"Target obligation: {target.sentence_text}\n"
        f"Deterministic explanation: {finding.explanation}\n"
    )


def parse_verification_response(response: str) -> VerificationResult:
    try:
        payload = json.loads(response)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("LLM response is not valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object")

    verified = payload.get("verified")
    confidence = payload.get("confidence")
    explanation = payload.get("explanation")

    if not isinstance(verified, bool):
        raise ValueError("'verified' must be a boolean")

    if isinstance(confidence, bool) or not isinstance(
        confidence,
        (int, float),
    ):
        raise ValueError("'confidence' must be a number")

    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("'explanation' must be a non-empty string")

    return VerificationResult(
        verified=verified,
        confidence=_clamp(float(confidence)),
        explanation=explanation.strip(),
    )


def _obligation_index(
    obligations: list[NormalizedObligation],
) -> dict[str, NormalizedObligation]:
    return {
        obligation.obligation_id: obligation
        for obligation in obligations
    }


def verify_findings(
    findings: list[Finding],
    obligations: list[NormalizedObligation],
    provider: LLMProvider | None,
    *,
    minimum_score: float = 0.70,
    maximum_score: float = 0.95,
) -> tuple[list[Finding], list[str]]:
    """
    Verify ambiguous deterministic findings with an optional LLM provider.

    Routing:
    - provider unavailable: preserve findings unchanged
    - score below minimum_score: preserve unchanged
    - score >= maximum_score: preserve unchanged
    - ambiguous score range: ask provider for verification

    A rejected finding is removed from the returned findings.

    Provider failures and malformed responses fail open:
    the original finding is preserved and a warning is returned.
    """

    if minimum_score < 0.0 or minimum_score > 1.0:
        raise ValueError("minimum_score must be between 0.0 and 1.0")

    if maximum_score < 0.0 or maximum_score > 1.0:
        raise ValueError("maximum_score must be between 0.0 and 1.0")

    if minimum_score >= maximum_score:
        raise ValueError(
            "minimum_score must be less than maximum_score"
        )

    if provider is None or not findings:
        return list(findings), []

    obligations_by_id = _obligation_index(obligations)

    verified_findings: list[Finding] = []
    warnings: list[str] = []

    for finding in findings:
        score = finding.deterministic_score

        if score < minimum_score or score >= maximum_score:
            verified_findings.append(finding)
            continue

        source_id = finding.source_obligation_id
        target_id = finding.target_obligation_id

        source = obligations_by_id.get(source_id) if source_id else None
        target = obligations_by_id.get(target_id) if target_id else None

        if source is None or target is None:
            warnings.append(
                f"[llm] finding={finding.finding_id!r}: "
                "source or target obligation not found"
            )
            verified_findings.append(finding)
            continue

        prompt = build_verification_prompt(
            finding,
            source,
            target,
        )

        try:
            response = provider.generate(prompt)
            verification = parse_verification_response(response)
        except Exception as exc:
            warnings.append(
                f"[llm] finding={finding.finding_id!r}: "
                f"{_safe_error_message(exc)}"
            )
            verified_findings.append(finding)
            continue

        if not verification.verified:
            continue

        updated = finding.model_copy(
            update={
                "llm_verified": True,
                "confidence": verification.confidence,
                "explanation": (
                    f"{finding.explanation} "
                    f"LLM verification: {verification.explanation}"
                ),
                "evidence": {
                    **finding.evidence,
                    "llm_verification": {
                        "verified": verification.verified,
                        "confidence": verification.confidence,
                        "explanation": verification.explanation,
                    },
                },
            }
        )

        verified_findings.append(updated)

    return verified_findings, warnings


__all__ = [
    "LLMProvider",
    "VerificationResult",
    "build_verification_prompt",
    "parse_verification_response",
    "verify_findings",
]