from pathlib import Path
from typing import Any

import pytest

from backend.app.services import policy_analysis
from shared.contracts.policy_analysis import AnalysisResult


def build_analysis_result() -> AnalysisResult:
    return AnalysisResult(
        policies=[],
        obligations=[],
        findings=[],
        statistics={"finding_count": 0},
        warnings=[],
        processing_time_ms=12.5,
    )


def test_analyze_documents_delegates_and_serializes_result() -> None:
    received_documents: list[Path] = []

    def fake_analyzer(documents: Any) -> AnalysisResult:
        received_documents.extend(documents)
        return build_analysis_result()

    document_paths = [Path("first.pdf"), Path("second.docx")]

    result = policy_analysis.analyze_documents(
        document_paths,
        analyzer=fake_analyzer,
    )

    assert received_documents == document_paths
    assert result["statistics"] == {"finding_count": 0}
    assert result["processing_time_ms"] == 12.5


def test_analyze_documents_rejects_invalid_result_type() -> None:
    def invalid_analyzer(documents: Any) -> dict[str, Any]:
        return {"documents": list(documents)}

    with pytest.raises(
        TypeError,
        match="Policy analyzer returned an invalid result type.",
    ):
        policy_analysis.analyze_documents(
            [Path("policy.pdf")],
            analyzer=invalid_analyzer,
        )


def test_load_engine_analyzer_reports_unavailable_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_engine_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "engine.pipeline":
            raise ImportError("engine branch is not present")

        return original_import(name, globals, locals, fromlist, level)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", reject_engine_import)

    with pytest.raises(
        policy_analysis.PolicyAnalysisUnavailableError,
        match="Policy analysis engine is unavailable.",
    ):
        policy_analysis.load_engine_analyzer()