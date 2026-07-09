from schemas import DischargeSummary
from src.formatting import NO_SEPARATE_VALUE, iter_field_rows


def _discharge_summary(discharge_diagnosis_value=None, discharge_diagnosis_citation=None) -> DischargeSummary:
    return DischargeSummary.model_validate({
        "patient_name": {"value": "Jane Doe", "confidence": 0.95, "source_citation": "Patient Name: Jane Doe"},
        "date_of_birth": {"value": None, "confidence": 0.0, "source_citation": None},
        "medical_record_number": {"value": None, "confidence": 0.0, "source_citation": None},
        "admission_date": {"value": None, "confidence": 0.0, "source_citation": None},
        "discharge_date": {"value": None, "confidence": 0.0, "source_citation": None},
        "admitting_diagnosis": {"value": "Pneumonia", "confidence": 0.9, "source_citation": "Dx: Pneumonia"},
        "discharge_diagnosis": {
            "value": discharge_diagnosis_value, "confidence": 0.0 if discharge_diagnosis_value is None else 0.9,
            "source_citation": discharge_diagnosis_citation,
        },
        "procedures_performed": {"value": [], "confidence": 0.0, "source_citation": None},
        "discharge_medications": [
            {"name": "Amoxicillin", "dosage": "500mg", "frequency": "TID", "confidence": 0.9,
             "source_citation": "Amoxicillin 500mg TID"},
        ],
        "allergies": {"value": [], "confidence": 0.0, "source_citation": None},
        "discharge_condition": {"value": None, "confidence": 0.0, "source_citation": None},
        "follow_up_instructions": {"value": None, "confidence": 0.0, "source_citation": None},
        "follow_up_appointments": {"value": [], "confidence": 0.0, "source_citation": None},
    })


def _row(rows, label_prefix):
    return next(r for r in rows if r[0].startswith(label_prefix))


def test_extracted_field_with_real_value_shows_that_value():
    rows = iter_field_rows(_discharge_summary())
    label, display_value, node = _row(rows, "Admitting Diagnosis")

    assert display_value == "Pneumonia"
    assert node is not None


def test_extracted_field_with_no_value_is_genuinely_not_found():
    """A field the model never found (None value) is a real 'not found' —
    distinct from NO_SEPARATE_VALUE, which just means 'already in the label'."""
    rows = iter_field_rows(_discharge_summary())
    label, display_value, node = _row(rows, "Discharge Diagnosis")

    assert display_value is None
    assert display_value is not NO_SEPARATE_VALUE
    assert node is not None  # still gradable — confidence 0.0, no citation


def test_medication_row_value_is_no_separate_value_not_none():
    """Regression test: a non-empty list[MedicationItem] row must use the
    NO_SEPARATE_VALUE sentinel, not None, so consumers don't render a
    correctly-extracted medication as 'not found'."""
    rows = iter_field_rows(_discharge_summary())
    label, display_value, node = _row(rows, "Discharge Medications")

    assert "Amoxicillin" in label
    assert display_value is NO_SEPARATE_VALUE
    assert node is not None
    assert node.confidence_level.value == "high"


def test_empty_extracted_field_list_shows_as_not_found_not_none_documented():
    """allergies/procedures_performed/follow_up_appointments are
    ExtractedField[list[str]] (not a raw list of MedicationItem/LabResultItem),
    so an empty value is a genuine 'not found' with a real (low-confidence)
    node — the "None documented" placeholder is reserved for the raw-list
    case below."""
    rows = iter_field_rows(_discharge_summary())
    label, display_value, node = _row(rows, "Allergies")

    assert display_value == []
    assert node is not None
    assert node.confidence_level.value == "low"


def test_empty_raw_list_field_has_none_node_and_placeholder_text():
    """discharge_medications/results are raw list[ConfidenceMixin] fields
    (not ExtractedField-wrapped) — when empty, there's nothing to grade, so
    the row gets the "None documented" placeholder with node=None."""
    extracted = _discharge_summary()
    extracted.discharge_medications.clear()
    rows = iter_field_rows(extracted)
    label, display_value, node = _row(rows, "Discharge Medications")

    assert display_value == "None documented"
    assert node is None
