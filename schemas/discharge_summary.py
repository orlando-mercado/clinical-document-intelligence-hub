"""Extraction schema for discharge summaries."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

from schemas.common import ConfidenceMixin, ExtractedField


class MedicationItem(ConfidenceMixin):
    """One discharge medication. Dosage errors are safety-critical, so each
    medication carries its own confidence and source citation rather than
    sharing one confidence score across the whole list."""

    name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None
    instructions: Optional[str] = None


class DischargeSummary(BaseModel):
    """Structured fields extracted from a discharge summary.

    Every field is required so the extraction prompt is forced to make an
    explicit "not found" statement (value=null, confidence=0.0,
    source_citation=null) rather than silently omitting data.
    """

    model_config = ConfigDict(extra="forbid")

    document_type: Literal["discharge_summary"] = "discharge_summary"

    patient_name: ExtractedField[str]
    date_of_birth: ExtractedField[Optional[str]]
    medical_record_number: ExtractedField[Optional[str]]

    admission_date: ExtractedField[Optional[str]]
    discharge_date: ExtractedField[Optional[str]]

    admitting_diagnosis: ExtractedField[Optional[str]]
    discharge_diagnosis: ExtractedField[Optional[str]]
    procedures_performed: ExtractedField[list[str]]

    discharge_medications: list[MedicationItem]
    allergies: ExtractedField[list[str]]

    discharge_condition: ExtractedField[Optional[str]]
    follow_up_instructions: ExtractedField[Optional[str]]
    follow_up_appointments: ExtractedField[list[str]]
