"""Multi-Doc Comparison: the third pipeline LLM call.

Compares two already-extracted, already-confidence-graded documents of the
SAME type — current vs. prior visit — and produces a plain-language
narrative, overall trend, structured key changes, and a risk flag for the
trajectory itself. The trajectory risk can differ from either visit's own
risk_flag — e.g. two individually-moderate lab values trending sharply
toward critical warrant escalation even if neither visit alone would.
"""

from __future__ import annotations

import anthropic

from schemas import ComparisonCard, ComparisonOutput
from src.config import DEFAULT_MODEL
from src.extraction import DOC_LABEL, ExtractedDocumentModel
from src.llm import StructuredCallError, call_structured

MAX_COMPARISON_ATTEMPTS = 3
MAX_TOKENS = 2048

_SYSTEM_PROMPT = """You are a clinical comparison assistant. You will be given two already-\
extracted, already-confidence-graded {doc_label}s as JSON — a CURRENT visit and a PRIOR visit for \
the same patient. Each field carries a `confidence`, `confidence_level`, and `source_citation` \
from a prior extraction step. Do not re-extract or restate the raw data; use it to produce:

1. `narrative`: 3-5 sentences in plain language describing what changed between the two visits \
and why it matters clinically.
2. `trend`: one of improving, stable, worsening, mixed, or not_comparable (use not_comparable if \
the two documents don't share enough comparable fields to judge a trend).
3. `key_changes`: the specific facts or values that changed, each with the prior value, current \
value, and a plain-language note on clinical significance. Only include fields present in both \
documents with a meaningful difference — do not list every field.
4. `risk_flag`: a risk assessment for the TRAJECTORY itself, not just the current visit in \
isolation. Escalate when a value is trending toward a critical range even if it wasn't flagged \
critical on either visit individually.

Fields with `confidence_level: "low"` (including anything with no source_citation) were not \
reliably grounded — treat them as uncertain, not fact, and do not build a key_changes entry or \
trend conclusion on a low-confidence field alone without noting the uncertainty.

Base your analysis only on the provided structured data. Do not invent clinical facts, values, or \
events that are not present in it."""


class ComparisonError(RuntimeError):
    """Raised when comparison fails, including mismatched document types
    or exhausting all retry attempts."""


def _build_messages(
    current_data: ExtractedDocumentModel,
    prior_data: ExtractedDocumentModel,
    current_filename: str,
    prior_filename: str,
) -> list[dict]:
    return [
        {
            "role": "user",
            "content": (
                f"CURRENT visit — '{current_filename}':\n\n{current_data.model_dump_json()}\n\n"
                f"PRIOR visit — '{prior_filename}':\n\n{prior_data.model_dump_json()}"
            ),
        }
    ]


def compare(
    current_data: ExtractedDocumentModel,
    prior_data: ExtractedDocumentModel,
    *,
    current_filename: str,
    prior_filename: str,
    client: anthropic.Anthropic | None = None,
    model: str = DEFAULT_MODEL,
    max_attempts: int = MAX_COMPARISON_ATTEMPTS,
) -> ComparisonCard:
    """Compare two already-extracted documents of the same type (current vs.
    prior visit) into a trend narrative + key changes + risk flag."""
    if current_data.document_type != prior_data.document_type:
        raise ComparisonError(
            f"Cannot compare a {current_data.document_type} against a "
            f"{prior_data.document_type} — both documents must be the same type."
        )

    doc_label = DOC_LABEL[current_data.document_type]
    client = client or anthropic.Anthropic()

    try:
        comparison_output = call_structured(
            client,
            model=model,
            system=_SYSTEM_PROMPT.format(doc_label=doc_label),
            messages=_build_messages(current_data, prior_data, current_filename, prior_filename),
            output_format=ComparisonOutput,
            max_tokens=MAX_TOKENS,
            max_attempts=max_attempts,
            label=f"comparison of {doc_label}s '{current_filename}' vs '{prior_filename}'",
        )
    except StructuredCallError as exc:
        raise ComparisonError(str(exc)) from exc

    return ComparisonCard.from_comparison(
        current_filename=current_filename,
        prior_filename=prior_filename,
        current_data=current_data,
        prior_data=prior_data,
        comparison=comparison_output,
    )
