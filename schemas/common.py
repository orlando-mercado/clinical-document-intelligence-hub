"""Shared building blocks for schema-constrained extraction output.

Confidence banding is intentionally *not* trusted from the LLM's own label —
`confidence_level` is recomputed deterministically after validation from the
numeric `confidence` score and whether a `source_citation` is present. This
keeps the color-coding in the UI consistent regardless of how the model
phrases its own certainty.
"""

from __future__ import annotations

from enum import Enum
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Confidence bands (green / amber / red in the UI). A field with no
# source_citation is always LOW, regardless of its numeric confidence.
HIGH_CONFIDENCE_THRESHOLD = 0.85
NEEDS_REVIEW_THRESHOLD = 0.6


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    NEEDS_REVIEW = "needs_review"
    LOW = "low"


def _confidence_level_for(confidence: float, source_citation: Optional[str]) -> ConfidenceLevel:
    if not source_citation or not source_citation.strip():
        return ConfidenceLevel.LOW
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return ConfidenceLevel.HIGH
    if confidence >= NEEDS_REVIEW_THRESHOLD:
        return ConfidenceLevel.NEEDS_REVIEW
    return ConfidenceLevel.LOW


class ConfidenceMixin(BaseModel):
    """Adds confidence + source-grounding to any extracted fact."""

    # Deliberately NOT validate_assignment=True: the "after" validator below
    # sets self.confidence_level directly, and under validate_assignment that
    # assignment re-triggers full model validation, which re-enters the same
    # validator — infinite recursion. Post-construction correction (see
    # clear_ungrounded_citation) is done as a plain, un-validated attribute
    # write instead.
    model_config = ConfigDict(extra="forbid")

    confidence: float = Field(
        ge=0.0, le=1.0, description="Model self-reported confidence, 0-1."
    )
    source_citation: Optional[str] = Field(
        default=None,
        description="Verbatim sentence/phrase from the source document "
        "supporting this value. Null if the value could not be grounded.",
    )
    confidence_level: ConfidenceLevel = Field(
        default=ConfidenceLevel.LOW,
        description="Derived band; recomputed on validation, do not set directly.",
    )

    @model_validator(mode="after")
    def _derive_confidence_level(self) -> "ConfidenceMixin":
        self.confidence_level = _confidence_level_for(self.confidence, self.source_citation)
        return self

    def clear_ungrounded_citation(self) -> None:
        """Call when a claimed source_citation turns out not to be a verbatim
        substring of the source document (see src/extraction.py). Clears the
        citation and recomputes confidence_level to match — always LOW, since
        a field with no citation is never trusted regardless of its
        confidence score."""
        self.source_citation = None
        self.confidence_level = _confidence_level_for(self.confidence, self.source_citation)


T = TypeVar("T")


class ExtractedField(ConfidenceMixin, Generic[T]):
    """A single extracted value plus its confidence and source grounding.

    Used for fields where the value itself doesn't need its own sub-schema
    (names, dates, free-text, or a list of strings). For values that are
    safety-critical and multi-part (lab results, medications), a dedicated
    model inherits ConfidenceMixin directly instead — see lab_report.py /
    discharge_summary.py.
    """

    value: Optional[T] = None


def iter_confidence_fields(model: BaseModel) -> list[tuple[str, ConfidenceMixin]]:
    """Walk a document model's top-level fields and list-of-item fields,
    collecting every ConfidenceMixin-derived node with a human-readable field
    path (e.g. "results[1]"). Shared by schemas.summary_card (to aggregate
    document-level confidence) and src.extraction (to verify grounding)."""
    nodes: list[tuple[str, ConfidenceMixin]] = []
    for field_name in type(model).model_fields:
        value = getattr(model, field_name)
        if isinstance(value, ConfidenceMixin):
            nodes.append((field_name, value))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, ConfidenceMixin):
                    nodes.append((f"{field_name}[{i}]", item))
    return nodes
