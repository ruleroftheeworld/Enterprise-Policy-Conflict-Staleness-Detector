from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from shared.contracts.policy_analysis import AnalysisResult


class PolicyAnalysisUnavailableError(RuntimeError):
    """Raised when the policy engine is unavailable to the backend."""


class PolicyAnalyzer(Protocol):
    def __call__(self, documents: Sequence[str | Path]) -> AnalysisResult:
        """Analyze policy documents and return the shared result contract."""


def load_engine_analyzer() -> PolicyAnalyzer:
    try:
        from engine.pipeline import analyze_policy_documents
    except ImportError as exc:
        raise PolicyAnalysisUnavailableError(
            "Policy analysis engine is unavailable."
        ) from exc

    return analyze_policy_documents


def analyze_documents(
    documents: Sequence[str | Path],
    *,
    analyzer: Callable[[Sequence[str | Path]], AnalysisResult] | None = None,
) -> dict[str, Any]:
    selected_analyzer = analyzer or load_engine_analyzer()
    result = selected_analyzer(documents)

    if not isinstance(result, AnalysisResult):
        raise TypeError("Policy analyzer returned an invalid result type.")

    return result.model_dump(mode="json")