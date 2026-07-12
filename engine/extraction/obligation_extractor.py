from __future__ import annotations

import hashlib
import re

from shared.contracts.policy_analysis import (
    NormalizedObligation,
    NormalizedPolicy,
)


# Order matters: negative multi-word forms must be matched before positive forms.
MODALITY_RULES = (
    (re.compile(r"\bmust\s+not\b", re.IGNORECASE), "PROHIBITED", True, 1.00),
    (re.compile(r"\bshall\s+not\b", re.IGNORECASE), "PROHIBITED", True, 1.00),
    (re.compile(r"\bmay\s+not\b", re.IGNORECASE), "PROHIBITED", True, 0.90),
    (re.compile(r"\bshould\s+not\b", re.IGNORECASE), "PROHIBITED", True, 0.80),
    (re.compile(r"\bprohibited\s+from\b", re.IGNORECASE), "PROHIBITED", True, 1.00),
    (re.compile(r"\bcannot\b", re.IGNORECASE), "PROHIBITED", True, 1.00),

    (re.compile(r"\bis\s+required\s+to\b", re.IGNORECASE), "REQUIRED", False, 1.00),
    (re.compile(r"\bare\s+required\s+to\b", re.IGNORECASE), "REQUIRED", False, 1.00),
    (re.compile(r"\brequired\s+to\b", re.IGNORECASE), "REQUIRED", False, 1.00),
    (re.compile(r"\bis\s+responsible\s+for\b", re.IGNORECASE), "REQUIRED", False, 0.90),
    (re.compile(r"\bare\s+responsible\s+for\b", re.IGNORECASE), "REQUIRED", False, 0.90),
    (re.compile(r"\bmandatory\b", re.IGNORECASE), "REQUIRED", False, 0.95),
    (re.compile(r"\bneeds\s+to\b", re.IGNORECASE), "REQUIRED", False, 0.90),
    (re.compile(r"\bneed\s+to\b", re.IGNORECASE), "REQUIRED", False, 0.90),
    (re.compile(r"\bmust\b", re.IGNORECASE), "REQUIRED", False, 1.00),
    (re.compile(r"\bshall\b", re.IGNORECASE), "REQUIRED", False, 1.00),

    (re.compile(r"\brecommended\s+to\b", re.IGNORECASE), "RECOMMENDED", False, 0.80),
    (re.compile(r"\bis\s+expected\s+to\b", re.IGNORECASE), "RECOMMENDED", False, 0.80),
    (re.compile(r"\bare\s+expected\s+to\b", re.IGNORECASE), "RECOMMENDED", False, 0.80),
    (re.compile(r"\bshould\b", re.IGNORECASE), "RECOMMENDED", False, 0.80),

    (re.compile(r"\boptional\b", re.IGNORECASE), "OPTIONAL", False, 0.65),
    (re.compile(r"\bmay\b", re.IGNORECASE), "OPTIONAL", False, 0.70),
)


TECHNOLOGY_RULES = (
    ("multi-factor authentication", re.compile(r"\bmulti-factor authentication\b", re.I)),
    ("AES-256", re.compile(r"\bAES[- ]?256\b", re.I)),
    ("TLS 1.0", re.compile(r"\bTLS\s*1\.0\b", re.I)),
    ("TLS 1.1", re.compile(r"\bTLS\s*1\.1\b", re.I)),
    ("TLS 1.2", re.compile(r"\bTLS\s*1\.2\b", re.I)),
    ("TLS 1.3", re.compile(r"\bTLS\s*1\.3\b", re.I)),
    ("SHA-1", re.compile(r"\bSHA[- ]?1\b", re.I)),
    ("Windows Server 2012", re.compile(r"\bWindows Server 2012(?: R2)?\b", re.I)),
    ("SSL", re.compile(r"\bSSL(?:v?2|v?3|\s*2\.0|\s*3\.0)?\b", re.I)),
    ("VPN", re.compile(r"\bVPN\b", re.I)),
    # Added: present in real policy (Reference: X) citations in the label set.
    # Staleness detection relies exclusively on the structured technology field,
    # so these must be extractable by the extractor.
    ("WEP", re.compile(r"\bWEP\b", re.I)),
    ("GDPR 2016", re.compile(r"\bGDPR\s*2016\b", re.I)),
    ("SOX 2002", re.compile(r"\bSOX\s*2002\b", re.I)),
    ("NIST SP 800-53 Rev 4", re.compile(r"\bNIST\s+SP\s+800-53\s+Rev\s*4\b", re.I)),
)


FREQUENCY_RULES = (
    re.compile(
        r"\bevery\s+\d+\s+"
        r"(?:minute|minutes|hour|hours|day|days|week|weeks|month|months|year|years)\b",
        re.I,
    ),
    re.compile(r"\b(?:daily|weekly|monthly|quarterly|annually|yearly)\b", re.I),
    re.compile(
        r"\bonce\s+(?:per|a)\s+"
        r"(?:day|week|month|quarter|year)\b",
        re.I,
    ),
    re.compile(
        r"\b\d+\s+times?\s+(?:per|a)\s+"
        r"(?:day|week|month|quarter|year)\b",
        re.I,
    ),
)


SCOPE_PATTERNS = (
    # ── Situational scope ────────────────────────────────────────────────────
    re.compile(r"\bfor\s+remote\s+access\b", re.I),
    re.compile(r"\bduring\s+business\s+hours\b", re.I),
    re.compile(r"\bduring\s+approved\s+maintenance\s+windows\b", re.I),
    re.compile(r"\bfor\s+external\s+communications\b", re.I),
    re.compile(r"\bfor\s+internal\s+communications\b", re.I),
    re.compile(r"\bon\s+production\s+systems\b", re.I),
    re.compile(r"\bon\s+development\s+systems\b", re.I),

    # ── Actor-type scope — matches ground-truth label schema ─────────────────
    # Canonical labels: contractors, developers, admins, employees,
    #                   service_accounts, all_employees
    re.compile(r"\ball\s+employees\b", re.I),          # all_employees (before employees)
    re.compile(r"\bcontractors?\b", re.I),              # contractors
    re.compile(r"\bdevelopers?\b", re.I),               # developers
    re.compile(r"\badmins?\b", re.I),                   # admins
    re.compile(r"\badministrators?\b", re.I),           # admins (long form)
    re.compile(r"\bservice\s+accounts?\b", re.I),       # service_accounts
    re.compile(r"\bemployees?\b", re.I),                # employees (general)
)


CONDITION_PATTERN = re.compile(
    r"\b(?:if|when|whenever|provided that|as long as|in the event that)\s+"
    r"(.+?)(?=,\s*|\bunless\b|\bexcept\b|$)",
    re.I,
)


EXCEPTION_PATTERN = re.compile(
    r"\b(?:unless|except(?:\s+when|\s+where|\s+for)?|with the exception of)\s+(.+?)(?=[.;]|$)",
    re.I,
)


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _clean_fragment(value: str | None) -> str | None:
    if value is None:
        return None

    value = re.sub(r"\s+", " ", value)
    value = value.strip(" \t\r\n,;:.-")

    return value or None


def _detect_modality(sentence: str):
    for pattern, modality, negated, strength in MODALITY_RULES:
        match = pattern.search(sentence)

        if match:
            return match, modality, negated, strength

    return None


def _extract_subject(sentence: str, modality_match: re.Match) -> str | None:
    return _clean_fragment(sentence[:modality_match.start()])


def _extract_frequency(sentence: str) -> str | None:
    for pattern in FREQUENCY_RULES:
        match = pattern.search(sentence)

        if match:
            return _clean_fragment(match.group(0).lower())

    return None


def _extract_technologies(sentence: str) -> list[str]:
    technologies = []

    for canonical_name, pattern in TECHNOLOGY_RULES:
        if pattern.search(sentence):
            technologies.append(canonical_name)

    return technologies


def _extract_scope(sentence: str) -> str | None:
    for pattern in SCOPE_PATTERNS:
        match = pattern.search(sentence)

        if match:
            return _clean_fragment(match.group(0).lower())

    return None


def _extract_condition(sentence: str) -> str | None:
    match = CONDITION_PATTERN.search(sentence)

    if not match:
        return None

    return _clean_fragment(match.group(1))


def _extract_exception(sentence: str) -> str | None:
    match = EXCEPTION_PATTERN.search(sentence)

    if not match:
        return None

    return _clean_fragment(match.group(1))


def _remove_trailing_qualifiers(text: str) -> str:
    boundaries = []

    patterns = list(FREQUENCY_RULES) + list(SCOPE_PATTERNS)

    for pattern in patterns:
        match = pattern.search(text)

        if match:
            boundaries.append(match.start())

    condition_match = CONDITION_PATTERN.search(text)
    exception_match = EXCEPTION_PATTERN.search(text)

    if condition_match:
        boundaries.append(condition_match.start())

    if exception_match:
        boundaries.append(exception_match.start())

    if boundaries:
        text = text[:min(boundaries)]

    return text.strip()


def _extract_action_object(
    sentence: str,
    modality_match: re.Match,
) -> tuple[str | None, str | None]:
    remainder = sentence[modality_match.end():].strip()

    # Handles "prohibited from using..."
    remainder = re.sub(r"^from\s+", "", remainder, flags=re.I)

    remainder = _remove_trailing_qualifiers(remainder)
    remainder = remainder.strip(" \t\r\n,;:.-")

    if not remainder:
        return None, None

    tokens = remainder.split()

    action = tokens[0].lower()

    object_text = " ".join(tokens[1:]) if len(tokens) > 1 else None

    return _clean_fragment(action), _clean_fragment(object_text)


def _calculate_confidence(
    subject: str | None,
    action: str | None,
    object_text: str | None,
    modality_strength: float,
) -> float:
    confidence = 0.50

    confidence += 0.15 if subject else 0.0
    confidence += 0.15 if action else 0.0
    confidence += 0.10 if object_text else 0.0
    confidence += 0.10 * modality_strength

    return round(min(confidence, 1.0), 3)


_TOPIC_KEYWORDS = {
    "password": ["password", "credential", "passphrase"],
    "encryption": ["encryption", "encrypt", "crypt", "cipher", "tls", "ssl", "wep", "sha-1", "md5", "des"],
    "access": ["access", "authorize", "authentication", "permission", "privilege", "mfa"],
    "data_retention": ["data retention", "retention", "retain", "delete", "destroy", "purge", "archive"],
    "logging": ["logging", "log", "audit", "monitoring", "monitor", "siem", "event"],
    "network": ["network", "firewall", "vpn", "ip", "port", "dns", "router", "switch", "wi-fi"],
    "patch": ["patch", "update", "vulnerability", "upgrade"],
    "backup": ["backup", "restore", "disaster", "dr", "replica"],
    "provisioning": ["provisioning", "provision", "deprovision", "hire", "terminate", "onboard", "offboard"],
    "vendor": ["vendor", "third-party", "supplier", "contractor", "partner"],
    "asset": ["asset", "inventory", "device", "hardware", "laptop", "desktop", "server"],
    "mobile": ["mobile", "phone", "tablet", "byod"],
    "physical": ["physical", "badge", "visitor", "lock", "facility", "building", "guard"],
    "api": ["api", "web service", "integration"],
    "endpoint": ["endpoint", "antivirus", "malware", "edr"],
    "cloud": ["cloud", "aws", "azure", "gcp", "saas", "paas", "iaas"],
    "change": ["change", "approve", "deploy", "release"],
    "hr": ["hr", "employee", "personnel", "staff"],
}

def _classify_topic(sentence: str) -> str | None:
    text = sentence.lower()
    for topic, keywords in _TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return topic
    return None


def extract_obligation(
    policy_id: str,
    section_id: str,
    sentence: str,
) -> NormalizedObligation | None:
    detected = _detect_modality(sentence)

    if detected is None:
        return None

    modality_match, modality, negated, strength = detected

    subject = _extract_subject(sentence, modality_match)
    action, object_text = _extract_action_object(sentence, modality_match)

    technology = _extract_technologies(sentence)
    scope = _extract_scope(sentence)
    frequency = _extract_frequency(sentence)
    condition = _extract_condition(sentence)
    exception = _extract_exception(sentence)

    identity = f"{policy_id}:{section_id}:{sentence}"

    confidence = _calculate_confidence(
        subject=subject,
        action=action,
        object_text=object_text,
        modality_strength=strength,
    )

    topic_val = _classify_topic(sentence)

    return NormalizedObligation(
        obligation_id=_stable_id("obligation", identity),
        policy_id=policy_id,
        section_id=section_id,
        sentence_text=sentence,
        subject=subject,
        action=action,
        object=object_text,
        technology=technology,
        scope=scope,
        frequency=frequency,
        condition=condition,
        exception=exception,
        topic=topic_val,
        strength=strength,
        modality=modality,
        negated=negated,
        confidence=confidence,
        embedding=[],
    )


def extract_policy_obligations(
    policy: NormalizedPolicy,
) -> list[NormalizedObligation]:
    obligations = []

    for section in policy.sections:
        for sentence in section.sentences:
            obligation = extract_obligation(
                policy_id=policy.policy_id,
                section_id=section.section_id,
                sentence=sentence,
            )

            if obligation is not None:
                obligations.append(obligation)

    return obligations


def extract_policies_obligations(
    policies: list[NormalizedPolicy],
) -> list[NormalizedObligation]:
    obligations = []

    for policy in policies:
        obligations.extend(extract_policy_obligations(policy))

    return obligations