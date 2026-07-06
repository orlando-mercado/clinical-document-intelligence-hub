"""Extraction schema for the Multi-Doc Comparison pipeline stage.

Compares two already-extracted, already-confidence-graded documents of the
SAME type — current vs. prior visit — never mixed types (a discharge summary
against a lab report isn't a meaningful comparison). Mirrors the two-part
split used by schemas.summary_card: `ComparisonOutput` is what the LLM call
is constrained to produce; `ComparisonCard` is the fully assembled object
combining it with both source documents.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from schemas.discharge_summary import DischargeSummary
from schemas.lab_report import LabReport
from schemas.summary_card import ExtractedDocument, RiskFlag


class Trend(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    WORSENING = "worsening"
    MIXED = "mixed"
    NOT_COMPARABLE = "not_comparable"


class KeyChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str = Field(description="What changed, e.g. a lab test name or clinical fact.")
    prior_value: str = Field(description="Value/state on the prior visit. Use 'not documented' if absent.")
    current_value: str = Field(description="Value/state on the current visit. Use 'not documented' if absent.")
    significance: str = Field(description="Plain-language clinical significance of this change.")


class ComparisonOutput(BaseModel):
    """What the comparison LLM call must produce — nothing else."""

    model_config = ConfigDict(extra="forbid")

    narrative: str = Field(
        description="3-5 sentence plain-language description of what changed between the two visits."
    )
    trend: Trend
    key_changes: list[KeyChange]
    risk_flag: RiskFlag


class ComparisonCard(BaseModel):
    """Final comparison output: narrative + trend + key changes + risk flag
    for the trajectory, plus both source documents, ready for the UI."""

    # See SummaryCard for why this differs from the LLM-facing schemas above:
    # assembled in code, not validated against raw model output, and needs to
    # tolerate its own computed fields on reload.
    model_config = ConfigDict(extra="ignore")

    current_filename: str
    prior_filename: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    narrative: str
    trend: Trend
    key_changes: list[KeyChange]
    risk_flag: RiskFlag

    current_data: ExtractedDocument
    prior_data: ExtractedDocument

    @model_validator(mode="after")
    def _validate_same_document_type(self) -> "ComparisonCard":
        if self.current_data.document_type != self.prior_data.document_type:
            raise ValueError(
                f"Cannot compare a {self.current_data.document_type} against a "
                f"{self.prior_data.document_type} — both documents must be the same type."
            )
        return self

    @computed_field  # type: ignore[misc]
    @property
    def document_type(self) -> Literal["discharge_summary", "lab_report"]:
        return self.current_data.document_type

    @computed_field  # type: ignore[misc]
    @property
    def patient_display_name(self) -> Optional[str]:
        return self.current_data.patient_name.value

    @classmethod
    def from_comparison(
        cls,
        *,
        current_filename: str,
        prior_filename: str,
        current_data: DischargeSummary | LabReport,
        prior_data: DischargeSummary | LabReport,
        comparison: ComparisonOutput,
    ) -> "ComparisonCard":
        return cls(
            current_filename=current_filename,
            prior_filename=prior_filename,
            current_data=current_data,
            prior_data=prior_data,
            narrative=comparison.narrative,
            trend=comparison.trend,
            key_changes=comparison.key_changes,
            risk_flag=comparison.risk_flag,
        )
