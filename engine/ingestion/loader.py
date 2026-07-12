from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path

from docx import Document
from pypdf import PdfReader

from shared.contracts.policy_analysis import NormalizedPolicy, PolicySection


SUPPORTED_EXTENSIONS = {".txt", ".md", ".docx", ".pdf"}

METADATA_PATTERNS = {
    "title": re.compile(r"^\s*(?:title|#)\s*:?\s*(.+?)\s*$", re.IGNORECASE),
    "version": re.compile(r"^\s*version\s*:\s*(.+?)\s*$", re.IGNORECASE),
    # Author is treated as an alias for owner in real policy files.
    "owner": re.compile(
        r"^\s*(?:owner|author)\s*:\s*(.+?)\s*$", re.IGNORECASE
    ),
    "department": re.compile(r"^\s*department\s*:\s*(.+?)\s*$", re.IGNORECASE),
    "effective_date": re.compile(
        r"^\s*effective\s+date\s*:\s*(.+?)\s*$", re.IGNORECASE
    ),
    # Matches both synthetic 'Review Date:' and real '**Last Reviewed:**' formats.
    # Markdown bold markers (**) are stripped before matching (see _extract_metadata).
    "review_date": re.compile(
        r"^\s*(?:review\s+date|last\s+reviewed)\s*:\s*(.+?)\s*$", re.IGNORECASE
    ),
    "status": re.compile(r"^\s*status\s*:\s*(.+?)\s*$", re.IGNORECASE),
}

NUMBERED_SECTION_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)*)[.)]?\s+(.+?)\s*$")
MARKDOWN_SECTION_PATTERN = re.compile(r"^\s*#{2,6}\s+(.+?)\s*$")

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


class UnsupportedDocumentTypeError(ValueError):
    pass


class DocumentIngestionError(RuntimeError):
    pass


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _read_txt_or_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_docx(path: Path) -> str:
    document = Document(path)
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    page_text = []

    for page in reader.pages:
        text = page.extract_text() or ""
        page_text.append(text)

    return "\n".join(page_text)


def _read_document(path: Path) -> str:
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentTypeError(
            f"Unsupported document type: {extension or '<none>'}"
        )

    try:
        if extension in {".txt", ".md"}:
            return _read_txt_or_markdown(path)

        if extension == ".docx":
            return _read_docx(path)

        return _read_pdf(path)

    except UnsupportedDocumentTypeError:
        raise
    except Exception as exc:
        raise DocumentIngestionError(
            f"Failed to read document '{path.name}': {exc}"
        ) from exc


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")

    lines = [line.rstrip() for line in text.splitlines()]

    cleaned_lines = []
    previous_blank = False

    for line in lines:
        blank = not line.strip()

        if blank and previous_blank:
            continue

        cleaned_lines.append(line)
        previous_blank = blank

    return "\n".join(cleaned_lines).strip()


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None

    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _extract_metadata(text: str, path: Path) -> dict:
    metadata = {
        "title": None,
        "version": None,
        "owner": None,
        "department": None,
        "effective_date": None,
        "review_date": None,
        "status": None,
    }

    lines = text.splitlines()

    for line in lines[:40]:
        stripped = line.strip()

        if not stripped:
            continue

        # Strip markdown bold markers (**) so that lines like
        # '**Last Reviewed:** 2021-08-15' match pattern keys correctly.
        stripped = stripped.replace("**", "")

        for field, pattern in METADATA_PATTERNS.items():
            match = pattern.match(stripped)

            if match and metadata[field] is None:
                metadata[field] = match.group(1).strip()

    if path.suffix.lower() == ".md":
        for line in lines[:20]:
            stripped = line.strip()

            if stripped.startswith("# ") and not stripped.startswith("## "):
                metadata["title"] = stripped[2:].strip()
                break

    metadata["title"] = metadata["title"] or path.stem

    metadata["effective_date"] = _parse_date(metadata["effective_date"])
    metadata["review_date"] = _parse_date(metadata["review_date"])

    return metadata


def _split_paragraphs(text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]


def _split_sentence_text(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []
    parts = SENTENCE_SPLIT_PATTERN.split(compact)
    return [p.strip() for p in parts if p.strip()]


def _split_sentences(text: str) -> list[str]:
    lines = text.splitlines()
    sentences = []
    current_sentence = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        is_bullet = False
        if stripped.startswith(("-", "*", "+")):
            if len(stripped) == 1 or stripped[1].isspace():
                is_bullet = True
        else:
            match = re.match(r"^\d+\.\s+", stripped)
            if match:
                is_bullet = True

        if is_bullet:
            if current_sentence:
                sentences.extend(_split_sentence_text(" ".join(current_sentence)))
                current_sentence = []

            if stripped.startswith(("-", "*", "+")):
                content = stripped[1:].strip()
            else:
                content = re.sub(r"^\d+\.\s+", "", stripped).strip()

            current_sentence.append(content)
        else:
            if current_sentence:
                prev_line = current_sentence[-1]
                if prev_line and prev_line[-1] in ".!?":
                    sentences.extend(_split_sentence_text(" ".join(current_sentence)))
                    current_sentence = [stripped]
                else:
                    current_sentence.append(stripped)
            else:
                current_sentence.append(stripped)

    if current_sentence:
        sentences.extend(_split_sentence_text(" ".join(current_sentence)))

    return sentences


def _is_metadata_line(line: str) -> bool:
    stripped = line.strip()

    for pattern in METADATA_PATTERNS.values():
        if pattern.match(stripped):
            return True

    if stripped.startswith("# ") and not stripped.startswith("## "):
        return True

    return False


def _match_section_heading(line: str) -> str | None:
    markdown_match = MARKDOWN_SECTION_PATTERN.match(line)

    if markdown_match:
        return markdown_match.group(1).strip()

    numbered_match = NUMBERED_SECTION_PATTERN.match(line)

    if numbered_match:
        return numbered_match.group(2).strip()

    return None


def _build_section(
    policy_id: str,
    index: int,
    title: str | None,
    lines: list[str],
) -> PolicySection | None:
    content_lines = [
        line.strip()
        for line in lines
        if line.strip() and not _is_metadata_line(line)
    ]

    text = "\n".join(content_lines).strip()

    if not text:
        return None

    paragraphs = _split_paragraphs(text)
    sentences = _split_sentences(text)

    section_identity = f"{policy_id}:{index}:{title or 'document'}"

    return PolicySection(
        section_id=_stable_id("section", section_identity),
        title=title,
        text=text,
        paragraphs=paragraphs,
        sentences=sentences,
    )


def _extract_sections(text: str, policy_id: str) -> list[PolicySection]:
    lines = text.splitlines()

    sections: list[PolicySection] = []

    current_title: str | None = None
    current_lines: list[str] = []
    section_index = 0

    for line in lines:
        heading = _match_section_heading(line)

        if heading:
            section = _build_section(
                policy_id,
                section_index,
                current_title,
                current_lines,
            )

            if section:
                sections.append(section)
                section_index += 1

            current_title = heading
            current_lines = []
            continue

        current_lines.append(line)

    final_section = _build_section(
        policy_id,
        section_index,
        current_title,
        current_lines,
    )

    if final_section:
        sections.append(final_section)

    if not sections:
        sections.append(
            PolicySection(
                section_id=_stable_id("section", f"{policy_id}:document"),
                title=None,
                text="",
                paragraphs=[],
                sentences=[],
            )
        )

    return sections


def load_policy_document(document: str | Path) -> NormalizedPolicy:
    path = Path(document)

    if not path.exists():
        raise FileNotFoundError(f"Policy document not found: {path}")

    if not path.is_file():
        raise ValueError(f"Policy document is not a file: {path}")

    raw_text = _read_document(path)
    cleaned_text = _clean_text(raw_text)

    metadata = _extract_metadata(cleaned_text, path)

    policy_identity = f"{path.resolve()}:{cleaned_text}"
    policy_id = _stable_id("policy", policy_identity)

    sections = _extract_sections(cleaned_text, policy_id)

    return NormalizedPolicy(
        policy_id=policy_id,
        title=metadata["title"],
        version=metadata["version"],
        owner=metadata["owner"],
        department=metadata["department"],
        effective_date=metadata["effective_date"],
        review_date=metadata["review_date"],
        status=metadata["status"],
        source_file=str(path),
        sections=sections,
        metadata={
            "file_type": path.suffix.lower(),
            "character_count": len(cleaned_text),
            "section_count": len(sections),
        },
    )


def load_policy_documents(
    documents: list[str | Path],
) -> tuple[list[NormalizedPolicy], list[str]]:
    policies = []
    warnings = []

    for document in documents:
        try:
            policies.append(load_policy_document(document))
        except Exception as exc:
            warnings.append(f"{document}: {exc}")

    return policies, warnings