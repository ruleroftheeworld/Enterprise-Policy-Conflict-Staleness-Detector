from __future__ import annotations

import hashlib
import re
from datetime import date, timedelta

from engine.detection.deterministic_detector import (
    STALE_POLICY,
    STALE_REFERENCE,
    _clamp,
    _rounded,
)
from shared.contracts.policy_analysis import (
    Finding,
    NormalizedObligation,
    NormalizedPolicy,
)

# ---------------------------------------------------------------------------
# Configuration: deprecated technology identifiers
# Add to this set to extend staleness reference detection without touching
# logic elsewhere.
# ---------------------------------------------------------------------------

DEPRECATED_TECHNOLOGIES: frozenset[str] = frozenset(
    {
        "TLS 1.0",
        "TLS 1.1",
        "SHA-1",
        "MD5",
        "DES",
        "SSLv3",
        "Windows Server 2012",
        "Windows Server 2008",
        "NIST SP 800-53 Rev 4",
        "NIST SP 800-63A",
        "NIST SP 800-63B",
        "PCI DSS v3.2",
        # Added: present in real policy (Reference: X) citations in the label set.
        "WEP",
        "GDPR 2016",
        "SOX 2002",
    }
)

# 18 months expressed as days (≈ 548 days); using 548 to match the intent of
# "18 calendar months" without pulling in dateutil.
_STALE_THRESHOLD_DAYS: int = 548


# ---------------------------------------------------------------------------
# Internal helpers (mirror the style in deterministic_detector.py)
# ---------------------------------------------------------------------------


def _stable_finding_id(
    finding_type: str,
    policy_id: str,
    discriminator: str = "",
) -> str:
    """Produce a stable, collision-resistant finding ID for a policy-level finding.

    Unlike the pair-based version in deterministic_detector.py, staleness
    findings are anchored to a single policy (and optionally a discriminator
    such as a deprecated term), so we hash those two/three values together.
    """
    value = f"{finding_type}:{policy_id}:{discriminator}"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"finding_{digest}"


def _is_stale_by_date(policy: NormalizedPolicy, today: date) -> bool:
    """Return True when the policy's review_date is older than the threshold."""
    if policy.review_date is None:
        return False
    return (today - policy.review_date) > timedelta(days=_STALE_THRESHOLD_DAYS)


def _find_deprecated_references(
    policy: NormalizedPolicy,
    obligations: list[NormalizedObligation],
) -> list[tuple[str, str]]:
    """Return a list of (deprecated_term, obligation_id) pairs found in this policy.

    Checks ONLY the structured ``technology`` list field of each obligation.
    Free-text sentence scanning is intentionally excluded: it generates false
    positives when a deprecated term appears in a '(Reference: X)' citation
    that is not the mechanism being mandated. The extractor populates the
    ``technology`` field from these citation patterns, so structured-only
    detection is both precise and sufficient.
    """
    policy_obligations = [
        ob for ob in obligations if ob.policy_id == policy.policy_id
    ]

    hits: list[tuple[str, str]] = []

    # Build a mapping from lowercase term → canonical casing for O(1) lookup.
    _lower_to_canonical: dict[str, str] = {
        term.lower(): term for term in DEPRECATED_TECHNOLOGIES
    }

    for ob in policy_obligations:
        # Check the structured technology list (exact match, case-insensitive).
        for tech in ob.technology:
            canonical = _lower_to_canonical.get(tech.lower())
            if canonical is not None:
                hits.append((canonical, ob.obligation_id))

    return hits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_staleness(
    policy: NormalizedPolicy,
    obligations: list[NormalizedObligation],
    *,
    today: date | None = None,
) -> list[Finding]:
    """Detect staleness findings for a single policy.

    Runs two independent checks:

    1. **STALE_POLICY** – the policy's ``review_date`` is more than 18 months
       before *today*.
    2. **STALE_REFERENCE** – at least one obligation references a deprecated
       technology identifier (see ``DEPRECATED_TECHNOLOGIES``).

    Returns a (possibly empty) list of :class:`~shared.contracts.policy_analysis.Finding`
    objects using the same schema as the deterministic detector.

    Args:
        policy: The normalised policy to evaluate.
        obligations: All normalised obligations from the full corpus; only
            obligations belonging to *policy* are examined.
        today: Override the current date (useful for deterministic tests).
            Defaults to :func:`datetime.date.today`.
    """
    _today = today or date.today()
    findings: list[Finding] = []

    # ------------------------------------------------------------------
    # Check 1: review_date staleness
    # ------------------------------------------------------------------
    if _is_stale_by_date(policy, _today):
        age_days = (_today - policy.review_date).days  # type: ignore[operator]
        months_overdue = round(age_days / 30.4)

        findings.append(
            Finding(
                finding_id=_stable_finding_id(
                    STALE_POLICY,
                    policy.policy_id,
                    str(policy.review_date),
                ),
                finding_type=STALE_POLICY,
                source_obligation_id=None,
                target_obligation_id=None,
                severity="MEDIUM",
                confidence=_rounded(1.0),
                deterministic_score=_rounded(1.0),
                llm_verified=False,
                explanation=(
                    f"Policy '{policy.title}' was last scheduled for review on "
                    f"{policy.review_date} ({months_overdue} months ago), which "
                    f"exceeds the 18-month staleness threshold."
                ),
                evidence={
                    "policy_id": policy.policy_id,
                    "policy_title": policy.title,
                    "review_date": str(policy.review_date),
                    "age_days": age_days,
                    "threshold_days": _STALE_THRESHOLD_DAYS,
                },
            )
        )

    # ------------------------------------------------------------------
    # Check 2: deprecated technology references
    # ------------------------------------------------------------------
    deprecated_hits = _find_deprecated_references(policy, obligations)

    # De-duplicate by (term, obligation_id) — one finding per unique pair so
    # that the same deprecated term in multiple obligations each gets its own
    # finding with a stable ID, while the same term in the same obligation is
    # only reported once.
    seen: set[tuple[str, str]] = set()

    for deprecated_term, obligation_id in deprecated_hits:
        key = (deprecated_term, obligation_id)
        if key in seen:
            continue
        seen.add(key)

        findings.append(
            Finding(
                finding_id=_stable_finding_id(
                    STALE_REFERENCE,
                    policy.policy_id,
                    f"{deprecated_term}:{obligation_id}",
                ),
                finding_type=STALE_REFERENCE,
                source_obligation_id=obligation_id,
                target_obligation_id=None,
                severity="HIGH",
                confidence=_rounded(1.0),
                deterministic_score=_rounded(1.0),
                llm_verified=False,
                explanation=(
                    f"Obligation references deprecated technology '{deprecated_term}', "
                    f"which is no longer considered secure or supported."
                ),
                evidence={
                    "policy_id": policy.policy_id,
                    "policy_title": policy.title,
                    "obligation_id": obligation_id,
                    "deprecated_term": deprecated_term,
                },
            )
        )

    return findings


__all__ = [
    "DEPRECATED_TECHNOLOGIES",
    "detect_staleness",
]
