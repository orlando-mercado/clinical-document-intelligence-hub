"""Schema-Constrained Extraction: turns loaded document text into a validated
DischargeSummary or LabReport.

Uses Claude's structured outputs (`output_format=`) so the API itself
enforces the JSON shape returned. The retry loop below exists for what
structured outputs does *not* enforce server-side — numeric bounds like
`confidence` being in [0, 1] are stripped from the schema sent to the model
and instead validated client-side by Pydantic (see the claude-api skill's
Structured Output notes), so an out-of-range value surfaces as a
ValidationError here, not a 400 from the API — plus transient API errors and
safety refusals.

After a successful parse, every claimed `source_citation` is checked against
the actual source text (schemas.common.iter_confidence_fields +
clear_ungrounded_citation) — the model's own claim of grounding is not
trusted any more than its self-reported confidence score.
"""

from __future__ import annotations

import logging
import re
from typing import Literal, Type, Union

import anthropic
from pydantic import ValidationError

from schemas import DischargeSummary, LabReport, iter_confidence_fields
from src.config import DEFAULT_MODEL
from src.loader import LoadedDocument

logger = logging.getLogger(__name__)

MAX_EXTRACTION_ATTEMPTS = 3
MAX_TOKENS = 4096

DocumentType = Literal["discharge_summary", "lab_report"]
ExtractedDocumentModel = Union[DischargeSummary, LabReport]

SCHEMA_BY_DOCUMENT_TYPE: dict[DocumentType, Type[ExtractedDocumentModel]] = {
    "discharge_summary": DischargeSummary,
    "lab_report": LabReport,
}

_DOC_LABEL: dict[DocumentType, str] = {
    "discharge_summary": "discharge summary",
    "lab_report": "lab report",
}

_SYSTEM_PROMPT = """You are a clinical document extraction assistant. You will be given the raw \
text of a {doc_label} and must extract structured fields exactly matching the provided schema.

For every extracted field:
- `confidence` is your own honest estimate (0.0-1.0) of how certain you are the value is correct.
- `source_citation` MUST be a verbatim quote (an exact substring) from the source document that \
supports the value. If you cannot find a supporting quote, set `source_citation` to null.
- If a field is not present in the document, set `value` to null, `confidence` to 0.0, and \
`source_citation` to null. Do not guess or fabricate values.
- Never invent a source_citation that does not appear in the source text.

Extract only what is explicitly stated in the document. Do not infer clinical conclusions beyond \
what is written."""


class ExtractionError(RuntimeError):
    """Raised when extraction fails after all retry attempts."""


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_verbatim_citation(citation: str, source_text: str) -> bool:
    """Whitespace-normalized substring check. Real documents often have
    column padding or line wraps that an honest, non-hallucinated quote would
    naturally collapse when phrased as prose — an exact byte-for-byte match
    would produce false negatives on those, not just on real hallucinations."""
    return _normalize_whitespace(citation) in _normalize_whitespace(source_text)


def _verify_source_grounding(document_model: ExtractedDocumentModel, source_text: str) -> list[str]:
    """Downgrade any field whose claimed source_citation isn't actually
    (whitespace-normalized) present in the source text. Returns the field
    paths that were downgraded, for logging."""
    downgraded: list[str] = []
    for path, node in iter_confidence_fields(document_model):
        if node.source_citation and not is_verbatim_citation(node.source_citation, source_text):
            node.clear_ungrounded_citation()
            downgraded.append(path)
    return downgraded


def _build_messages(document: LoadedDocument) -> list[dict]:
    return [
        {
            "role": "user",
            "content": f"Source document ({document.source_filename}):\n\n{document.text}",
        }
    ]


def extract(
    document: LoadedDocument,
    document_type: DocumentType,
    *,
    client: anthropic.Anthropic | None = None,
    model: str = DEFAULT_MODEL,
    max_attempts: int = MAX_EXTRACTION_ATTEMPTS,
) -> ExtractedDocumentModel:
    """Extract structured, confidence-graded, source-grounded fields from
    `document` into the schema for `document_type`."""
    schema_cls = SCHEMA_BY_DOCUMENT_TYPE[document_type]
    doc_label = _DOC_LABEL[document_type]
    client = client or anthropic.Anthropic()

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.messages.parse(
                model=model,
                max_tokens=MAX_TOKENS,
                system=_SYSTEM_PROMPT.format(doc_label=doc_label),
                messages=_build_messages(document),
                output_format=schema_cls,
            )
        except (anthropic.APIError, ValidationError) as exc:
            last_error = exc
            logger.warning(
                "Extraction attempt %d/%d for '%s' failed: %s",
                attempt, max_attempts, document.source_filename, exc,
            )
            continue

        if response.stop_reason == "refusal":
            last_error = ExtractionError(
                f"Model declined to extract from '{document.source_filename}'."
            )
            logger.warning(
                "Extraction attempt %d/%d for '%s' refused.",
                attempt, max_attempts, document.source_filename,
            )
            continue

        parsed = response.parsed_output
        if parsed is None:
            last_error = ExtractionError("Structured output parsing returned no result.")
            logger.warning(
                "Extraction attempt %d/%d for '%s': parsed_output is None.",
                attempt, max_attempts, document.source_filename,
            )
            continue

        try:
            # parsed should already be a schema_cls instance; re-validate to
            # guarantee our custom validators (confidence banding,
            # extra="forbid") ran, regardless of exactly what the SDK handed back.
            data = parsed if isinstance(parsed, dict) else parsed.model_dump()
            result = schema_cls.model_validate(data)
        except ValidationError as exc:
            last_error = exc
            logger.warning(
                "Extraction attempt %d/%d for '%s' failed re-validation: %s",
                attempt, max_attempts, document.source_filename, exc,
            )
            continue

        downgraded = _verify_source_grounding(result, document.text)
        if downgraded:
            logger.info(
                "Downgraded %d ungrounded citation(s) in '%s': %s",
                len(downgraded), document.source_filename, downgraded,
            )
        return result

    raise ExtractionError(
        f"Failed to extract {doc_label} from '{document.source_filename}' after "
        f"{max_attempts} attempts."
    ) from last_error
