"""Document loading: extracts raw text from .txt files and text-layer PDFs.

No OCR / no image input (see docs/scope.md). A PDF with no extractable text
layer, or an image file, is rejected with a clear error rather than silently
falling back to OCR or vision.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pdfplumber
import pypdf

RawInput = Union[bytes, str, Path, io.IOBase]

TEXT_EXTENSIONS = {".txt", ".md"}
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}

# Below this many extracted characters, treat a PDF as having no text layer
# (scanned/image-only) rather than as a document with a short body.
MIN_EXTRACTED_CHARS = 20


class UnsupportedDocumentError(ValueError):
    """A document can't be loaded within the locked scope — see docs/scope.md."""


@dataclass
class LoadedDocument:
    text: str
    source_filename: str
    page_count: Optional[int] = None


def _read_bytes(file: RawInput) -> bytes:
    if isinstance(file, bytes):
        return file
    if isinstance(file, (str, Path)):
        return Path(file).read_bytes()
    if hasattr(file, "read"):
        return file.read()
    raise TypeError(f"Unsupported input type for document loading: {type(file)!r}")


def load_text(raw: RawInput, filename: str = "document.txt") -> LoadedDocument:
    data = _read_bytes(raw)
    text = data.decode("utf-8", errors="replace").strip()
    if not text:
        raise UnsupportedDocumentError(f"'{filename}' is empty.")
    return LoadedDocument(text=text, source_filename=filename)


def _extract_pdf_text(data: bytes) -> tuple[str, int]:
    """Try pdfplumber first (better layout handling); fall back to pypdf."""
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        page_texts = [page.extract_text() or "" for page in pdf.pages]
        page_count = len(pdf.pages)
    text = "\n\n".join(t.strip() for t in page_texts if t.strip())
    if len(text) >= MIN_EXTRACTED_CHARS:
        return text, page_count

    reader = pypdf.PdfReader(io.BytesIO(data))
    page_texts = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(t.strip() for t in page_texts if t.strip())
    return text, len(reader.pages)


def load_pdf(raw: RawInput, filename: str = "document.pdf") -> LoadedDocument:
    data = _read_bytes(raw)
    text, page_count = _extract_pdf_text(data)
    if len(text) < MIN_EXTRACTED_CHARS:
        raise UnsupportedDocumentError(
            f"'{filename}' has no extractable text layer. Scanned/image PDFs "
            "are out of scope for this prototype (no OCR) — see docs/scope.md."
        )
    return LoadedDocument(text=text, source_filename=filename, page_count=page_count)


def load_document(raw: RawInput, filename: str) -> LoadedDocument:
    """Dispatch to the right loader based on the file extension in `filename`."""
    suffix = Path(filename).suffix.lower()
    if suffix in PDF_EXTENSIONS:
        return load_pdf(raw, filename=filename)
    if suffix in TEXT_EXTENSIONS:
        return load_text(raw, filename=filename)
    if suffix in IMAGE_EXTENSIONS:
        raise UnsupportedDocumentError(
            f"Image input ('{suffix}') is out of scope for this prototype "
            "(no OCR) — see docs/scope.md."
        )
    raise UnsupportedDocumentError(
        f"Unsupported file type '{suffix}' for '{filename}'. "
        "Only .txt and text-layer .pdf are supported."
    )
