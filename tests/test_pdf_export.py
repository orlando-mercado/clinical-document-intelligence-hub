import io

from pypdf import PdfReader

from schemas import DischargeSummary, LabReport, SummarizationOutput, SummaryCard
from src.pdf_export import render_summary_card_pdf


def _discharge_summary_card() -> SummaryCard:
    extracted = DischargeSummary.model_validate({
        "patient_name": {"value": "Jane Doe", "confidence": 0.95, "source_citation": "Patient Name: Jane Doe"},
        "date_of_birth": {"value": None, "confidence": 0.0, "source_citation": None},
        "medical_record_number": {"value": None, "confidence": 0.0, "source_citation": None},
        "admission_date": {"value": None, "confidence": 0.0, "source_citation": None},
        "discharge_date": {"value": None, "confidence": 0.0, "source_citation": None},
        "admitting_diagnosis": {"value": "Pneumonia", "confidence": 0.9, "source_citation": "Dx: Pneumonia"},
        "discharge_diagnosis": {"value": None, "confidence": 0.0, "source_citation": None},
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
    summarization = SummarizationOutput.model_validate({
        "plain_language_summary": "Jane Doe was treated for pneumonia and is recovering well.",
        "risk_flag": {"level": "low", "rationale": "No critical findings noted."},
        "recommended_actions": [{"action": "Follow up with PCP in 1 week", "priority": "routine"}],
    })
    return SummaryCard.from_summarization(
        source_filename="discharge.txt", extracted_data=extracted, summarization=summarization,
    )


def _lab_report_card_with_empty_results() -> SummaryCard:
    extracted = LabReport.model_validate({
        "patient_name": {"value": "John Roe", "confidence": 0.9, "source_citation": "Patient: John Roe"},
        "date_of_birth": {"value": None, "confidence": 0.0, "source_citation": None},
        "medical_record_number": {"value": None, "confidence": 0.0, "source_citation": None},
        "collection_date": {"value": None, "confidence": 0.0, "source_citation": None},
        "ordering_provider": {"value": None, "confidence": 0.0, "source_citation": None},
        "panel_name": {"value": None, "confidence": 0.0, "source_citation": None},
        "results": [],
    })
    summarization = SummarizationOutput.model_validate({
        "plain_language_summary": "No lab results could be extracted from this document.",
        "risk_flag": {"level": "none", "rationale": "No data to assess."},
        "recommended_actions": [],
    })
    return SummaryCard.from_summarization(
        source_filename="lab.pdf", extracted_data=extracted, summarization=summarization,
    )


def test_render_summary_card_pdf_produces_valid_pdf():
    card = _discharge_summary_card()

    pdf_bytes = render_summary_card_pdf(card)

    assert pdf_bytes.startswith(b"%PDF")
    reader = PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) >= 1


def test_render_summary_card_pdf_contains_key_content():
    card = _discharge_summary_card()

    pdf_bytes = render_summary_card_pdf(card)
    text = "".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf_bytes)).pages)

    assert "Jane Doe" in text
    assert "Pneumonia" in text
    assert "Risk: Low" in text
    assert "Amoxicillin" in text
    assert "Follow up with PCP" in text
    assert "Dx: Pneumonia" in text  # source citation carried through
    # The medication row's label already carries its value (name + dosage +
    # frequency) — it must not also render as "not found" (other genuinely
    # missing fields like date of birth legitimately do say "not found",
    # so this checks the medication's own row, not the whole document).
    medication_line = next(line for line in text.splitlines() if "Amoxicillin" in line)
    assert "not found" not in medication_line


def test_render_summary_card_pdf_handles_empty_lists_and_no_actions():
    card = _lab_report_card_with_empty_results()

    pdf_bytes = render_summary_card_pdf(card)
    text = "".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf_bytes)).pages)

    assert pdf_bytes.startswith(b"%PDF")
    assert "John Roe" in text
    assert "None specified" in text  # no recommended actions
    assert "None documented" in text  # empty results list
