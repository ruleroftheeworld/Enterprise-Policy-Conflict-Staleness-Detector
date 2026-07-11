from __future__ import annotations

import re

from shared.contracts.policy_analysis import NormalizedObligation


ACTION_SYNONYMS = {
    "utilize": "use",
    "employ": "use",
    "authenticate": "use",
    "permit": "allow",
    "authorize": "allow",
    "forbid": "deny",
    "disallow": "deny",
    "prohibit": "deny",
    "block": "deny",
    "validate": "verify",
    "inspect": "review",
    "check": "review",
}


SUBJECT_SYNONYMS = {
    "staff members": "employees",
    "staff": "employees",
    "workers": "employees",
    "system admins": "system administrators",
    "sysadmins": "system administrators",
    "db administrators": "database administrators",
    "db admins": "database administrators",
    "network admins": "network administrators",
}


TECHNOLOGY_ALIASES = {
    "mfa": "multi-factor authentication",
    "multifactor authentication": "multi-factor authentication",
    "multi factor authentication": "multi-factor authentication",

    "aes256": "AES-256",
    "aes 256": "AES-256",
    "aes-256": "AES-256",

    "sha1": "SHA-1",
    "sha 1": "SHA-1",
    "sha-1": "SHA-1",

    "tls1.0": "TLS 1.0",
    "tls 1.0": "TLS 1.0",

    "tls1.1": "TLS 1.1",
    "tls 1.1": "TLS 1.1",

    "tls1.2": "TLS 1.2",
    "tls 1.2": "TLS 1.2",

    "tls1.3": "TLS 1.3",
    "tls 1.3": "TLS 1.3",

    "windows server 2012 r2": "Windows Server 2012",
    "windows server 2012": "Windows Server 2012",

    "virtual private network": "VPN",
    "vpn": "VPN",

    "secure sockets layer": "SSL",
    "ssl": "SSL",
}


FREQUENCY_ALIASES = {
    "daily": "every 1 day",
    "once per day": "every 1 day",
    "once a day": "every 1 day",

    "weekly": "every 1 week",
    "once per week": "every 1 week",
    "once a week": "every 1 week",

    "monthly": "every 1 month",
    "once per month": "every 1 month",
    "once a month": "every 1 month",

    "quarterly": "every 3 months",
    "once per quarter": "every 3 months",
    "once a quarter": "every 3 months",

    "annually": "every 1 year",
    "yearly": "every 1 year",
    "once per year": "every 1 year",
    "once a year": "every 1 year",
}


SCOPE_PREFIXES = (
    "for ",
    "during ",
    "on ",
    "within ",
)


TECHNOLOGY_PHRASE_PATTERNS = (
    re.compile(r"\s+using\s+AES[- ]?256\b", re.I),
    re.compile(r"\s+using\s+SHA[- ]?1\b", re.I),
    re.compile(r"\s+using\s+TLS\s*1\.[0-3]\b", re.I),
    re.compile(r"\s+using\s+SSL(?:v?2|v?3|\s*2\.0|\s*3\.0)?\b", re.I),
)


def _normalize_whitespace(value: str | None) -> str | None:
    if value is None:
        return None

    value = re.sub(r"\s+", " ", value).strip()

    return value or None


def _normalize_lower(value: str | None) -> str | None:
    value = _normalize_whitespace(value)

    return value.lower() if value else None


def _normalize_subject(subject: str | None) -> str | None:
    subject = _normalize_lower(subject)

    if not subject:
        return None

    return SUBJECT_SYNONYMS.get(subject, subject)


def _normalize_action(action: str | None) -> str | None:
    action = _normalize_lower(action)

    if not action:
        return None

    return ACTION_SYNONYMS.get(action, action)


def _normalize_technology_name(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value).strip()
    lookup = normalized.lower()

    return TECHNOLOGY_ALIASES.get(lookup, normalized)


def _normalize_technologies(technologies: list[str]) -> list[str]:
    normalized = []

    for technology in technologies:
        canonical = _normalize_technology_name(technology)

        if canonical not in normalized:
            normalized.append(canonical)

    return sorted(normalized, key=str.lower)


def _remove_technology_phrases(
    object_text: str,
    technologies: list[str],
) -> str:
    result = object_text

    for pattern in TECHNOLOGY_PHRASE_PATTERNS:
        result = pattern.sub("", result)

    for technology in technologies:
        escaped = re.escape(technology)

        result = re.sub(
            rf"\s+(?:using|with|via)\s+{escaped}\b",
            "",
            result,
            flags=re.I,
        )

    return result


def _normalize_object(
    object_text: str | None,
    technologies: list[str],
) -> str | None:
    object_text = _normalize_lower(object_text)

    if not object_text:
        return None

    object_text = _remove_technology_phrases(
        object_text,
        technologies,
    )

    object_text = re.sub(r"\s+", " ", object_text)
    object_text = object_text.strip(" ,;:.-")

    return object_text or None


def _normalize_scope(scope: str | None) -> str | None:
    scope = _normalize_lower(scope)

    if not scope:
        return None

    for prefix in SCOPE_PREFIXES:
        if scope.startswith(prefix):
            scope = scope[len(prefix):]
            break

    return scope.strip() or None


def _singularize_frequency_unit(unit: str) -> str:
    unit = unit.lower()

    mapping = {
        "minutes": "minute",
        "hours": "hour",
        "days": "day",
        "weeks": "week",
        "months": "month",
        "years": "year",
    }

    return mapping.get(unit, unit)

def _format_every_frequency(amount: int, unit: str) -> str:
    suffix = "" if amount == 1 else "s"
    return f"every {amount} {unit}{suffix}"

def _normalize_frequency(frequency: str | None) -> str | None:
    frequency = _normalize_lower(frequency)

    if not frequency:
        return None

    if frequency in FREQUENCY_ALIASES:
        return FREQUENCY_ALIASES[frequency]

    every_match = re.fullmatch(
        r"every\s+(\d+)\s+"
        r"(minute|minutes|hour|hours|day|days|week|weeks|month|months|year|years)",
        frequency,
    )

    if every_match:
        amount = int(every_match.group(1))
        unit = _singularize_frequency_unit(every_match.group(2))

        return _format_every_frequency(amount, unit)

    times_match = re.fullmatch(
        r"(\d+)\s+times?\s+(?:per|a)\s+"
        r"(day|week|month|quarter|year)",
        frequency,
    )

    if times_match:
        amount = int(times_match.group(1))
        unit = times_match.group(2)

        return f"{amount} times per {unit}"

    return frequency


def _normalize_condition(value: str | None) -> str | None:
    return _normalize_lower(value)


def _normalize_exception(value: str | None) -> str | None:
    return _normalize_lower(value)


def _normalize_modality(
    modality: str,
    negated: bool,
) -> tuple[str, bool, float]:
    modality = modality.upper().strip()

    if negated or modality == "PROHIBITED":
        return "PROHIBITED", True, 1.0

    strength_by_modality = {
        "REQUIRED": 1.0,
        "RECOMMENDED": 0.8,
        "OPTIONAL": 0.7,
    }

    if modality not in strength_by_modality:
        raise ValueError(f"Unsupported modality: {modality}")

    return modality, False, strength_by_modality[modality]


def normalize_obligation(
    obligation: NormalizedObligation,
) -> NormalizedObligation:
    normalized = obligation.model_copy(deep=True)

    normalized.subject = _normalize_subject(normalized.subject)
    normalized.action = _normalize_action(normalized.action)

    normalized.technology = _normalize_technologies(
        normalized.technology
    )

    normalized.object = _normalize_object(
        normalized.object,
        normalized.technology,
    )

    normalized.scope = _normalize_scope(normalized.scope)
    normalized.frequency = _normalize_frequency(normalized.frequency)
    normalized.condition = _normalize_condition(normalized.condition)
    normalized.exception = _normalize_exception(normalized.exception)

    (
        normalized.modality,
        normalized.negated,
        normalized.strength,
    ) = _normalize_modality(
        normalized.modality,
        normalized.negated,
    )

    return normalized


def normalize_obligations(
    obligations: list[NormalizedObligation],
) -> list[NormalizedObligation]:
    return [
        normalize_obligation(obligation)
        for obligation in obligations
    ]