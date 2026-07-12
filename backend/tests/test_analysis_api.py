from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from backend.app import main


client = TestClient(main.app)


def test_analyze_policy_files_passes_temporary_paths_to_service(
    monkeypatch: Any,
) -> None:
    captured_paths: list[Path] = []

    def fake_analyze_documents(document_paths: list[Path]) -> dict[str, object]:
        captured_paths.extend(document_paths)

        assert len(document_paths) == 2
        assert all(path.exists() for path in document_paths)
        assert document_paths[0].name == "0-first.pdf"
        assert document_paths[1].name == "1-second.docx"
        assert document_paths[0].read_bytes() == b"first policy"
        assert document_paths[1].read_bytes() == b"second policy"

        return {
            "policies": [],
            "obligations": [],
            "findings": [],
            "statistics": {"document_count": 2},
            "warnings": [],
            "processing_time_ms": 1.0,
        }

    monkeypatch.setattr(main, "analyze_documents", fake_analyze_documents)

    response = client.post(
        "/api/v1/analyses",
        files=[
            ("files", ("first.pdf", b"first policy", "application/pdf")),
            (
                "files",
                (
                    "second.docx",
                    b"second policy",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ),
            ),
        ],
    )

    assert response.status_code == 200
    assert response.json()["statistics"] == {"document_count": 2}

    assert captured_paths
    assert all(not path.exists() for path in captured_paths)


def test_analyze_policy_files_strips_directory_components_from_filename(
    monkeypatch: Any,
) -> None:
    captured_names: list[str] = []

    def fake_analyze_documents(document_paths: list[Path]) -> dict[str, object]:
        captured_names.extend(path.name for path in document_paths)

        return {
            "policies": [],
            "obligations": [],
            "findings": [],
            "statistics": {},
            "warnings": [],
            "processing_time_ms": 1.0,
        }

    monkeypatch.setattr(main, "analyze_documents", fake_analyze_documents)

    response = client.post(
        "/api/v1/analyses",
        files=[
            ("files", ("../unsafe.pdf", b"policy", "application/pdf")),
        ],
    )

    assert response.status_code == 200
    assert captured_names == ["0-unsafe.pdf"]

def test_analyze_policy_files_rejects_empty_file_list() -> None:
    response = client.post(
        "/api/v1/analyses",
        files=[],
    )

    assert response.status_code == 422

def test_analyze_policy_files_accepts_markdown_document(
    monkeypatch: Any,
) -> None:
    captured_paths: list[Path] = []

    def fake_analyze_documents(
        document_paths: list[Path],
    ) -> dict[str, object]:
        captured_paths.extend(document_paths)

        assert len(document_paths) == 1
        assert document_paths[0].exists()
        assert document_paths[0].name == "0-policy.md"
        assert document_paths[0].read_bytes() == b"# Password Policy"

        return {
            "policies": [],
            "obligations": [],
            "findings": [],
            "statistics": {"document_count": 1},
            "warnings": [],
            "processing_time_ms": 1.0,
        }

    monkeypatch.setattr(main, "analyze_documents", fake_analyze_documents)

    response = client.post(
        "/api/v1/analyses",
        files=[
            (
                "files",
                ("policy.md", b"# Password Policy", "text/markdown"),
            ),
        ],
    )

    assert response.status_code == 200
    assert len(captured_paths) == 1


def test_analyze_policy_files_rejects_unsupported_extension() -> None:
    response = client.post(
        "/api/v1/analyses",
        files=[
            ("files", ("policy.exe", b"not a policy", "application/octet-stream")),
        ],
    )

    assert response.status_code == 415
    assert response.json() == {
        "detail": (
            "Unsupported file type for 'policy.exe'. "
            "Allowed types: .pdf, .docx, .txt, .md."
        )
    }


def test_analyze_policy_files_rejects_empty_file() -> None:
    response = client.post(
        "/api/v1/analyses",
        files=[
            ("files", ("empty.pdf", b"", "application/pdf")),
        ],
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Uploaded file 'empty.pdf' is empty."
    }


def test_analyze_policy_files_returns_503_when_engine_is_unavailable(
    monkeypatch: Any,
) -> None:
    def unavailable_analyzer(document_paths: list[Path]) -> dict[str, object]:
        raise main.PolicyAnalysisUnavailableError(
            "Policy analysis engine is unavailable."
        )

    monkeypatch.setattr(main, "analyze_documents", unavailable_analyzer)

    response = client.post(
        "/api/v1/analyses",
        files=[
            ("files", ("policy.pdf", b"policy", "application/pdf")),
        ],
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Policy analysis engine is unavailable."
    }


def test_analyze_policy_files_hides_unexpected_analysis_errors(
    monkeypatch: Any,
) -> None:
    def failing_analyzer(document_paths: list[Path]) -> dict[str, object]:
        raise RuntimeError("sensitive internal failure")

    monkeypatch.setattr(main, "analyze_documents", failing_analyzer)

    response = client.post(
        "/api/v1/analyses",
        files=[
            ("files", ("policy.pdf", b"policy", "application/pdf")),
        ],
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Policy analysis failed."}
    assert "sensitive internal failure" not in response.text