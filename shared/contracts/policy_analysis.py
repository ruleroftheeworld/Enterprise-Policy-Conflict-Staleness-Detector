from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class PolicySection(BaseModel):
    section_id: str
    title: str | None = None
    text: str
    paragraphs: list[str] = Field(default_factory=list)
    sentences: list[str] = Field(default_factory=list)


class NormalizedPolicy(BaseModel):
    policy_id: str
    title: str
    version: str | None = None
    owner: str | None = None
    department: str | None = None
    effective_date: date | None = None
    review_date: date | None = None
    status: str | None = None
    source_file: str
    sections: list[PolicySection] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NormalizedObligation(BaseModel):
    obligation_id: str
    policy_id: str
    section_id: str
    sentence_text: str

    subject: str | None = None
    action: str | None = None
    object: str | None = None

    technology: list[str] = Field(default_factory=list)

    scope: str | None = None
    frequency: str | None = None
    condition: str | None = None
    exception: str | None = None
    topic: str | None = None

    strength: float
    modality: str
    negated: bool
    confidence: float

    embedding: list[float] = Field(default_factory=list)

    corpus_frequency: int | None = None


class Finding(BaseModel):
    finding_id: str
    finding_type: str

    source_obligation_id: str | None = None
    target_obligation_id: str | None = None

    severity: str
    confidence: float
    deterministic_score: float

    llm_verified: bool = False

    explanation: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    policies: list[NormalizedPolicy]
    obligations: list[NormalizedObligation]
    findings: list[Finding]

    statistics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    processing_time_ms: float