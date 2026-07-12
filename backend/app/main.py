from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from backend.app.services.policy_analysis import (
    PolicyAnalysisUnavailableError,
    analyze_documents,
)

ALLOWED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

app = FastAPI(
    title="Enterprise Policy Intelligence Platform API",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/analyses")
async def analyze_policy_files(
    files: Annotated[list[UploadFile], File(...)],
) -> dict[str, object]:
    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one policy document is required.",
        )

    with TemporaryDirectory(prefix="policy-analysis-") as temporary_directory:
        document_paths: list[Path] = []

        try:
            for index, uploaded_file in enumerate(files):
                safe_name = Path(
                    uploaded_file.filename or f"document-{index}"
                ).name
                extension = Path(safe_name).suffix.lower()

                if extension not in ALLOWED_DOCUMENT_EXTENSIONS:
                    raise HTTPException(
                        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                        detail=(
                            f"Unsupported file type for '{safe_name}'. "
                            "Allowed types: .pdf, .docx, .txt, .md."
                        ),
                    )

                destination = (
                    Path(temporary_directory) / f"{index}-{safe_name}"
                )
                bytes_written = 0

                with destination.open("wb") as output_file:
                    while chunk := await uploaded_file.read(1024 * 1024):
                        output_file.write(chunk)
                        bytes_written += len(chunk)

                if bytes_written == 0:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Uploaded file '{safe_name}' is empty.",
                    )

                document_paths.append(destination)

            try:
                return analyze_documents(document_paths)
            except PolicyAnalysisUnavailableError as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Policy analysis engine is unavailable.",
                ) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Policy analysis failed.",
                ) from exc
        finally:
            for uploaded_file in files:
                await uploaded_file.close()