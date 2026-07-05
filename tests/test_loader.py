import io

import pytest
from reportlab.pdfgen import canvas

from src.loader import UnsupportedDocumentError, load_document


def _make_pdf_bytes(text: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, text)
    c.save()
    return buf.getvalue()


def test_load_text_extracts_content():
    doc = load_document(io.BytesIO(b"Patient: Jane Doe\nDOB: 01/01/1980"), filename="note.txt")
    assert "Jane Doe" in doc.text
    assert doc.source_filename == "note.txt"
    assert doc.page_count is None


def test_load_text_rejects_empty():
    with pytest.raises(UnsupportedDocumentError):
        load_document(io.BytesIO(b"   "), filename="empty.txt")


def test_load_pdf_with_text_layer():
    pdf_bytes = _make_pdf_bytes("Discharge Summary for Jane Doe, DOB 01/01/1980")
    doc = load_document(io.BytesIO(pdf_bytes), filename="discharge.pdf")
    assert "Jane Doe" in doc.text
    assert doc.page_count == 1


def test_load_pdf_rejects_blank_page_no_text_layer():
    buf = io.BytesIO()
    canvas.Canvas(buf).save()  # single blank page, nothing to extract
    with pytest.raises(UnsupportedDocumentError):
        load_document(io.BytesIO(buf.getvalue()), filename="scanned.pdf")


def test_rejects_image_input():
    with pytest.raises(UnsupportedDocumentError):
        load_document(io.BytesIO(b"\x89PNG\r\n\x1a\n"), filename="scan.png")


def test_rejects_unknown_extension():
    with pytest.raises(UnsupportedDocumentError):
        load_document(io.BytesIO(b"anything"), filename="notes.docx")


def test_load_document_accepts_bytes_directly():
    doc = load_document(b"Lab report body text here.", filename="lab.txt")
    assert "Lab report" in doc.text


def test_load_document_accepts_path(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text("Sample lab report text.")
    doc = load_document(p, filename="sample.txt")
    assert "Sample lab report" in doc.text
