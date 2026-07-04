# Scope Lock

This document is the source of truth for what this prototype does and does not do. Built for a 5-day proof-of-concept — scope is intentionally narrow to ship a working, demoable pipeline.

## In scope

- **Document types (2):**
  - Discharge summary
  - Lab report
- **Input formats:** plain text, and PDF containing a text layer (extracted via `pdfplumber`/`pypdf`)
- **Pipeline:** Document Loader → Schema-Constrained Extraction (LLM + Pydantic validation, with retry on invalid schema output) → Confidence & Source-Grounding (per-field source citation) → Summarization & Recommendation → Streamlit UI
- **Output:** color-coded summary card (high confidence / needs review / low-no-citation), PDF export, audit log (SQLite/JSON) of every run, multi-document comparison (current vs. prior visit)

## Explicitly out of scope

- **No OCR / no image input.** Scanned or photographed documents, and image file uploads (PNG/JPEG/etc.), are not supported. A PDF must already contain a real text layer — if `pdfplumber`/`pypdf` extract no text, the document is rejected with a clear error rather than falling back to OCR or vision.
- **No document types beyond discharge summaries and lab reports** — intake forms and physician notes (mentioned in the brief as examples) are not built, even though the schema/pipeline pattern would generalize to them.
- **No EHR/EMR integration** — no HL7/FHIR ingestion, no live system connections.
- **No authentication/authorization or multi-tenant access control** — single-user local/demo deployment only.
- **No proprietary or real patient data** — synthetic/generated sample documents only (`/data/synthetic`).

## Why this scope

**Why:** 5-day build window; the brief's evaluation rubric weights AI integration quality and clinical reasoning clarity over breadth of document-type coverage. Two document types is enough to demonstrate the extraction → confidence → summarization pipeline end-to-end and to exercise multi-document comparison (discharge summary vs. lab report, or same document type across visits).

**How to apply:** If asked to add a new document type, image/OCR input, or EHR integration, treat that as an explicit scope change requiring a decision, not a small addition — flag the tradeoff against the remaining time budget before implementing.
