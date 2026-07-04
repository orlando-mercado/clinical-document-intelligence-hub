# Pydantic schemas for extracted clinical fields

- `common.py` — `ConfidenceLevel`, `ConfidenceMixin`, `ExtractedField[T]`. Confidence bands (high/needs_review/low) are recomputed after validation from the numeric score + presence of a `source_citation`, never trusted verbatim from the LLM.
- `discharge_summary.py` — `DischargeSummary`, `MedicationItem`.
- `lab_report.py` — `LabReport`, `LabResultItem`, `LabResultFlag`.

Lab results and medications carry per-item confidence/citation (safety-critical, multi-part values); simpler fields (names, dates, diagnoses) use the generic `ExtractedField[T]` wrapper. All document-level fields are required — the extraction prompt must explicitly emit `value=null, confidence=0.0, source_citation=null` rather than omitting a field, and `extra="forbid"` means any hallucinated field trips validation and triggers the extraction retry.
