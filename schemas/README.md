# Pydantic schemas for extracted clinical fields

- `common.py` — `ConfidenceLevel`, `ConfidenceMixin`, `ExtractedField[T]`. Confidence bands (high/needs_review/low) are recomputed after validation from the numeric score + presence of a `source_citation`, never trusted verbatim from the LLM.
- `discharge_summary.py` — `DischargeSummary`, `MedicationItem`.
- `lab_report.py` — `LabReport`, `LabResultItem`, `LabResultFlag`.
- `summary_card.py` — `SummarizationOutput` (the summarization LLM's constrained output: plain-language summary, risk flag, recommended actions) and `SummaryCard` (the fully assembled object — `SummaryCard.from_summarization(...)` combines a `SummarizationOutput` with the already-extracted `DischargeSummary`/`LabReport`), plus `RiskLevel`/`RiskFlag`/`RecommendedAction`.

Lab results and medications carry per-item confidence/citation (safety-critical, multi-part values); simpler fields (names, dates, diagnoses) use the generic `ExtractedField[T]` wrapper. All document-level fields are required — the extraction prompt must explicitly emit `value=null, confidence=0.0, source_citation=null` rather than omitting a field, and `extra="forbid"` means any hallucinated field trips validation and triggers the extraction retry.

`SummaryCard`'s document-level fields (`overall_confidence`, `overall_confidence_level`, `fields_needing_review`, `document_type`, `patient_display_name`) are all `computed_field`s derived from `extracted_data` — never independent LLM output. This keeps the summarization call from re-stating (and possibly drifting from) data the extraction stage already produced and graded.
