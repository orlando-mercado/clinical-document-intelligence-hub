"""Summarization & Recommendation: the second pipeline LLM call.

Takes an already-extracted, already-confidence-graded DischargeSummary or
LabReport (not raw document text — see schemas.summary_card.SummarizationOutput's
docstring for why) and produces a plain-language summary, risk flag, and
recommended actions, then assembles the final SummaryCard.
"""

from __future__ import annotations

import anthropic

from schemas import SummarizationOutput, SummaryCard
from src.config import DEFAULT_MODEL
from src.extraction import DOC_LABEL, ExtractedDocumentModel
from src.llm import StructuredCallError, call_structured

MAX_SUMMARIZATION_ATTEMPTS = 3
MAX_TOKENS = 2048

_SYSTEM_PROMPT = """You are a clinical summarization assistant. You will be given already-\
extracted, already-confidence-graded structured data from a {doc_label} as JSON — not raw \
document text. Each field carries a `confidence`, `confidence_level` (high / needs_review / low), \
and `source_citation` from a prior extraction step. Do not re-extract or restate this data; use \
it to produce:

1. `plain_language_summary`: 3-5 sentences summarizing the clinically significant facts for a \
clinical/administrative reader, in plain language, avoiding unexplained jargon.
2. `risk_flag`: an overall risk assessment.
   - `level` must be one of: none, low, moderate, high, critical.
   - Escalate to "critical" for any life-threatening lab value (e.g. a result flagged CRITICAL) \
or a clear safety-critical gap (e.g. no follow-up scheduled after a serious diagnosis).
   - `rationale` must reference the specific data driving the assessment.
3. `recommended_actions`: concrete next steps a clinician or care coordinator should take, each \
with a `priority` of routine, urgent, or immediate.

Fields with `confidence_level: "low"` (including anything with no source_citation) were not \
reliably grounded in the source document — treat them as uncertain, not as established fact. If a \
safety-critical judgment would depend on a low-confidence field, say so explicitly rather than \
presenting it as settled.

Base your summary and recommendations only on the provided structured data. Do not invent \
clinical facts, medications, or values that are not present in it."""


class SummarizationError(RuntimeError):
    """Raised when summarization fails after all retry attempts."""


def _build_messages(extracted_data: ExtractedDocumentModel, source_filename: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": (
                f"Extracted data from '{source_filename}' "
                f"(document_type={extracted_data.document_type}):\n\n"
                f"{extracted_data.model_dump_json()}"
            ),
        }
    ]


def summarize(
    extracted_data: ExtractedDocumentModel,
    *,
    source_filename: str,
    client: anthropic.Anthropic | None = None,
    model: str = DEFAULT_MODEL,
    max_attempts: int = MAX_SUMMARIZATION_ATTEMPTS,
) -> SummaryCard:
    """Summarize an already-extracted document into a plain-language
    summary + risk flag + recommended actions, and assemble the SummaryCard."""
    doc_label = DOC_LABEL[extracted_data.document_type]
    client = client or anthropic.Anthropic()

    try:
        summarization_output = call_structured(
            client,
            model=model,
            system=_SYSTEM_PROMPT.format(doc_label=doc_label),
            messages=_build_messages(extracted_data, source_filename),
            output_format=SummarizationOutput,
            max_tokens=MAX_TOKENS,
            max_attempts=max_attempts,
            label=f"summarization of {doc_label} from '{source_filename}'",
        )
    except StructuredCallError as exc:
        raise SummarizationError(str(exc)) from exc

    return SummaryCard.from_summarization(
        source_filename=source_filename,
        extracted_data=extracted_data,
        summarization=summarization_output,
    )
