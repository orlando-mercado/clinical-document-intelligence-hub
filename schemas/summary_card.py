"""Extraction schema for the Summarization & Recommendation pipeline stage.

Two schemas live here, matching the two-stage nature of the pipeline:

- `SummarizationOutput` is what the summarization LLM call is constrained to
  produce — plain-language summary, risk flag, recommended next steps. It
  never re-emits the already-extracted clinical data; that would let the
  model quietly restate (and potentially drift from) values that were
  already extracted and graded in the prior stage.
- `SummaryCard` is the fully assembled object — `SummarizationOutput` plus
  the source `DischargeSummary`/`LabReport` — that backs the Streamlit
  summary card, PDF export, and audit log. Build it with
  `SummaryCard.from_summarization(...)`.

Document-level confidence (`overall_confidence`, `overall_confidence_level`,
`fields_needing_review`) is never asked of the LLM. It's derived
deterministically from the field-level confidences already computed during
extraction (see `schemas.common.ConfidenceMixin`) — the same "don't trust
the model's self-grading" principle applied one level up.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, computed_field

from schemas.common import (
    ConfidenceLevel,
    HIGH_CONFIDENCE_THRESHOLD,
    NEEDS_REVIEW_THRESHOLD,
    iter_confidence_fields,
)
from schemas.discharge_summary import DischargeSummary
from schemas.lab_report import LabReport

ExtractedDocument = Annotated[
    Union[DischargeSummary, LabReport], Field(discriminator="document_type")
]


class RiskLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class RiskFlag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: RiskLevel
    rationale: str = Field(description="Plain-language reason, referencing the source data.")


class RecommendedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str
    priority: Literal["routine", "urgent", "immediate"] = "routine"


class SummarizationOutput(BaseModel):
    """What the summarization LLM call must produce — nothing else."""

    model_config = ConfigDict(extra="forbid")

    plain_language_summary: str = Field(
        description="3-5 sentence plain-language summary for a clinical/administrative audience."
    )
    risk_flag: RiskFlag
    recommended_actions: list[RecommendedAction]


class SummaryCard(BaseModel):
    """Final pipeline output: plain-language summary + risk flag +
    recommended next steps + the underlying extracted data, ready for the
    Streamlit UI, PDF export, and audit log."""

    # Unlike the LLM-facing schemas above, this model is assembled in code
    # (from_summarization) rather than validated against raw model output, so
    # extra="forbid" buys nothing here — and would break reloading a
    # model_dump()'d card (e.g. from the audit log), since its computed
    # fields serialize out but can't be re-fed in under "forbid".
    model_config = ConfigDict(extra="ignore")

    source_filename: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    plain_language_summary: str
    risk_flag: RiskFlag
    recommended_actions: list[RecommendedAction]

    extracted_data: ExtractedDocument

    @computed_field  # type: ignore[misc]
    @property
    def document_type(self) -> Literal["discharge_summary", "lab_report"]:
        return self.extracted_data.document_type

    @computed_field  # type: ignore[misc]
    @property
    def patient_display_name(self) -> Optional[str]:
        return self.extracted_data.patient_name.value

    @computed_field  # type: ignore[misc]
    @property
    def overall_confidence(self) -> float:
        nodes = iter_confidence_fields(self.extracted_data)
        if not nodes:
            return 0.0
        return sum(node.confidence for _, node in nodes) / len(nodes)

    @computed_field  # type: ignore[misc]
    @property
    def overall_confidence_level(self) -> ConfidenceLevel:
        score = self.overall_confidence
        if score >= HIGH_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.HIGH
        if score >= NEEDS_REVIEW_THRESHOLD:
            return ConfidenceLevel.NEEDS_REVIEW
        return ConfidenceLevel.LOW

    @computed_field  # type: ignore[misc]
    @property
    def fields_needing_review(self) -> list[str]:
        return [
            path
            for path, node in iter_confidence_fields(self.extracted_data)
            if node.confidence_level != ConfidenceLevel.HIGH
        ]

    @classmethod
    def from_summarization(
        cls,
        *,
        source_filename: str,
        extracted_data: DischargeSummary | LabReport,
        summarization: SummarizationOutput,
    ) -> "SummaryCard":
        return cls(
            source_filename=source_filename,
            extracted_data=extracted_data,
            plain_language_summary=summarization.plain_language_summary,
            risk_flag=summarization.risk_flag,
            recommended_actions=summarization.recommended_actions,
        )
