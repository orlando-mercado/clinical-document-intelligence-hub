import pytest

from schemas import DischargeSummary, RiskLevel
from src.summarization import SummarizationError, summarize
from tests.fakes import FakeAPIError, FakeClient, FakeResponse


def _valid_discharge_summary() -> DischargeSummary:
    return DischargeSummary.model_validate({
        "patient_name": {"value": "Jane Doe", "confidence": 0.95, "source_citation": "Patient Name: Jane Doe"},
        "date_of_birth": {"value": "01/01/1980", "confidence": 0.9, "source_citation": "DOB: 01/01/1980"},
        "medical_record_number": {"value": None, "confidence": 0.0, "source_citation": None},
        "admission_date": {"value": None, "confidence": 0.0, "source_citation": None},
        "discharge_date": {"value": None, "confidence": 0.0, "source_citation": None},
        "admitting_diagnosis": {
            "value": "Pneumonia", "confidence": 0.9,
            "source_citation": "Admitting Diagnosis: Pneumonia",
        },
        "discharge_diagnosis": {"value": None, "confidence": 0.0, "source_citation": None},
        "procedures_performed": {"value": [], "confidence": 0.0, "source_citation": None},
        "discharge_medications": [],
        "allergies": {"value": [], "confidence": 0.0, "source_citation": None},
        "discharge_condition": {"value": None, "confidence": 0.0, "source_citation": None},
        "follow_up_instructions": {"value": None, "confidence": 0.0, "source_citation": None},
        "follow_up_appointments": {"value": [], "confidence": 0.0, "source_citation": None},
    })


def _valid_summarization_output_data() -> dict:
    return {
        "plain_language_summary": "Jane Doe was treated for pneumonia and is recovering well.",
        "risk_flag": {"level": "low", "rationale": "No critical values or safety gaps noted."},
        "recommended_actions": [{"action": "Follow up with PCP in 1 week", "priority": "routine"}],
    }


def test_summarize_success_first_attempt():
    extracted = _valid_discharge_summary()
    client = FakeClient([FakeResponse(parsed_output=_valid_summarization_output_data())])

    card = summarize(extracted, source_filename="discharge.txt", client=client)

    assert card.document_type == "discharge_summary"
    assert card.patient_display_name == "Jane Doe"
    assert card.risk_flag.level == RiskLevel.LOW
    assert card.plain_language_summary.startswith("Jane Doe")
    assert client.messages.calls == 1
    # The LLM call is scoped to the already-extracted data, not raw document text.
    sent_content = client.messages.last_kwargs["messages"][0]["content"]
    assert "Jane Doe" in sent_content
    assert "confidence" in sent_content  # extracted_data.model_dump_json() includes grading


def test_summarize_retries_on_refusal_then_succeeds():
    extracted = _valid_discharge_summary()
    client = FakeClient([
        FakeResponse(parsed_output=None, stop_reason="refusal"),
        FakeResponse(parsed_output=_valid_summarization_output_data()),
    ])

    card = summarize(extracted, source_filename="discharge.txt", client=client, max_attempts=3)

    assert card.risk_flag.level == RiskLevel.LOW
    assert client.messages.calls == 2


def test_summarize_retries_on_api_error_then_succeeds():
    extracted = _valid_discharge_summary()
    client = FakeClient([FakeAPIError(), FakeResponse(parsed_output=_valid_summarization_output_data())])

    card = summarize(extracted, source_filename="discharge.txt", client=client, max_attempts=3)

    assert card.risk_flag.level == RiskLevel.LOW
    assert client.messages.calls == 2


def test_summarize_exhausts_retries_and_raises():
    extracted = _valid_discharge_summary()
    client = FakeClient([
        FakeResponse(parsed_output=None, stop_reason="refusal"),
        FakeResponse(parsed_output=None, stop_reason="refusal"),
        FakeResponse(parsed_output=None, stop_reason="refusal"),
    ])

    with pytest.raises(SummarizationError):
        summarize(extracted, source_filename="discharge.txt", client=client, max_attempts=3)

    assert client.messages.calls == 3


def test_summarize_preserves_extracted_data_and_computed_fields():
    """The LLM never re-emits extracted_data — summarize() must assemble the
    SummaryCard from the original object, so field-level confidence/citation
    grounding from the extraction stage survives unchanged."""
    extracted = _valid_discharge_summary()
    client = FakeClient([FakeResponse(parsed_output=_valid_summarization_output_data())])

    card = summarize(extracted, source_filename="discharge.txt", client=client)

    assert card.extracted_data is extracted
    assert card.extracted_data.admitting_diagnosis.source_citation == "Admitting Diagnosis: Pneumonia"
    assert "medical_record_number" in card.fields_needing_review  # low confidence, no citation
