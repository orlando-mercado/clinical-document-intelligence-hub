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


class ConfidenceMixin(BaseModel):
    """Adds confidence + source-grounding to any extracted fact."""

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
        if not self.source_citation or not self.source_citation.strip():
            self.confidence_level = ConfidenceLevel.LOW
        elif self.confidence >= HIGH_CONFIDENCE_THRESHOLD:
            self.confidence_level = ConfidenceLevel.HIGH
        elif self.confidence >= NEEDS_REVIEW_THRESHOLD:
            self.confidence_level = ConfidenceLevel.NEEDS_REVIEW
        else:
            self.confidence_level = ConfidenceLevel.LOW
        return self


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
