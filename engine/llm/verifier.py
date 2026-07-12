from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol
from concurrent.futures import ThreadPoolExecutor, as_completed

from shared.contracts.policy_analysis import Finding, NormalizedObligation
from dataclasses import dataclass, field

class LLMProvider(Protocol):
    """Minimal provider interface used by the verifier."""

    def generate(self, prompt: str) -> str:
        ...


@dataclass(frozen=True)
class VerificationResult:
    verified: bool
    confidence: float
    explanation: str

@dataclass
class VerificationStats:
    eligible_findings: int = 0
    verified_findings: int = 0
    rejected_findings: int = 0
    bypassed_findings: int = 0
    dropped_findings: int = 0
    failed_findings: int = 0
    failure_messages: list[str] = field(default_factory=list)

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
    minimum_score: float = 0.50,
    maximum_score: float = 0.95,
    stats: VerificationStats | None = None,
) -> tuple[list[Finding], list[str]]:
    """
    Verify ambiguous deterministic findings with an optional LLM provider.

    Routing:
    - provider unavailable: preserve findings unchanged
    - score >= maximum_score (default 0.95): auto-accept, bypass LLM
      (reserved for STALE_POLICY / STALE_REFERENCE which always emit 1.0)
    - minimum_score <= score < maximum_score (default 0.50–0.95):
      send to LLM verifier; keep only if verified=true
    - score < minimum_score (default 0.50): drop the candidate entirely;
      do NOT emit a finding and do NOT call the LLM

    A rejected finding (LLM says verified=false) is removed from the
    returned findings. A dropped finding (score too low) is silently
    discarded.

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
    active_stats = stats if stats is not None else VerificationStats()

    if not findings:
        return [], []

    if provider is None:
        active_stats.bypassed_findings += len(findings)
        return list(findings), []

    obligations_by_id = _obligation_index(obligations)

    bypassed_findings: list[Finding] = []
    dropped_count = 0
    to_verify: list[tuple[Finding, NormalizedObligation, NormalizedObligation]] = []
    failed_lookup_findings: list[Finding] = []
    warnings: list[str] = []

    for finding in findings:
        score = finding.deterministic_score
        if score >= maximum_score:
            bypassed_findings.append(finding)
        elif score < minimum_score:
            dropped_count += 1
        else:
            source_id = finding.source_obligation_id
            target_id = finding.target_obligation_id
            source = obligations_by_id.get(source_id) if source_id else None
            target = obligations_by_id.get(target_id) if target_id else None

            if source is None or target is None:
                failed_lookup_findings.append(finding)
                warning = (
                    f"[llm] finding={finding.finding_id!r}: "
                    "source or target obligation not found"
                )
                warnings.append(warning)
                active_stats.failure_messages.append(warning)
            else:
                to_verify.append((finding, source, target))

    active_stats.bypassed_findings += len(bypassed_findings)
    active_stats.dropped_findings += dropped_count
    active_stats.failed_findings += len(failed_lookup_findings)

    verified_findings: list[Finding] = list(bypassed_findings) + list(failed_lookup_findings)

    if not to_verify:
        verified_findings.sort(
            key=lambda f: (
                -f.deterministic_score,
                f.finding_type,
                f.finding_id,
            )
        )
        return verified_findings, warnings

    total_to_verify = len(to_verify)
    print(f"[llm verifier] Starting concurrent verification of {total_to_verify} findings...", flush=True)

    def verify_one(item):
        finding, source, target = item
        prompt = build_verification_prompt(finding, source, target)
        try:
            response = provider.generate(prompt)
            verification = parse_verification_response(response)
            return (finding, verification, None)
        except Exception as exc:
            return (finding, None, exc)

    # Use 8 workers to process in parallel
    completed_count = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(verify_one, item): item for item in to_verify}
        for future in as_completed(futures):
            completed_count += 1
            finding, verification, exc = future.result()

            if completed_count % 10 == 0 or completed_count == total_to_verify:
                print(f"[llm verifier] Verified {completed_count}/{total_to_verify} findings...", flush=True)

            active_stats.eligible_findings += 1

            if exc is not None:
                active_stats.failed_findings += 1
                warning = (
                    f"[llm] finding={finding.finding_id!r}: "
                    f"{_safe_error_message(exc)}"
                )
                warnings.append(warning)
                active_stats.failure_messages.append(warning)
                verified_findings.append(finding)
            else:
                if not verification.verified:
                    active_stats.rejected_findings += 1
                else:
                    active_stats.verified_findings += 1
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

    verified_findings.sort(
        key=lambda f: (
            -f.deterministic_score,
            f.finding_type,
            f.finding_id,
        )
    )

    return verified_findings, warnings


__all__ = [
    "LLMProvider",
    "VerificationResult",
    "build_verification_prompt",
    "parse_verification_response",
    "verify_findings",
    "VerificationStats",
]