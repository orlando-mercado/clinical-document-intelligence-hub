"""Extraction schema for lab reports."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

from schemas.common import ConfidenceMixin, ExtractedField


class LabResultFlag(str, Enum):
    NORMAL = "normal"
    HIGH = "high"
    LOW = "low"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class LabResultItem(ConfidenceMixin):
    """One test result row. Carries its own confidence and source citation
    (like MedicationItem) since a single mis-extracted numeric value can
    drive an incorrect risk flag downstream."""

    test_name: str
    value: Optional[str] = None
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    flag: LabResultFlag = LabResultFlag.UNKNOWN


class LabReport(BaseModel):
    """Structured fields extracted from a lab report.

    Every field is required so the extraction prompt is forced to make an
    explicit "not found" statement (value=null, confidence=0.0,
    source_citation=null) rather than silently omitting data.
    """

    model_config = ConfigDict(extra="forbid")

    document_type: Literal["lab_report"] = "lab_report"

    patient_name: ExtractedField[str]
    date_of_birth: ExtractedField[Optional[str]]
    medical_record_number: ExtractedField[Optional[str]]

    collection_date: ExtractedField[Optional[str]]
    ordering_provider: ExtractedField[Optional[str]]
    panel_name: ExtractedField[Optional[str]]

    results: list[LabResultItem]
