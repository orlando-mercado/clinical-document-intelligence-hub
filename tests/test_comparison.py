import pytest

from schemas import DischargeSummary, LabReport, Trend
from src.comparison import ComparisonError, compare
from tests.fakes import FakeAPIError, FakeClient, FakeResponse


def _lab_report(potassium_value: str, flag: str) -> LabReport:
    return LabReport.model_validate({
        "patient_name": {"value": "James Whitfield", "confidence": 0.95, "source_citation": "Patient Name: James Whitfield"},
        "date_of_birth": {"value": None, "confidence": 0.0, "source_citation": None},
        "medical_record_number": {"value": None, "confidence": 0.0, "source_citation": None},
        "collection_date": {"value": None, "confidence": 0.0, "source_citation": None},
        "ordering_provider": {"value": None, "confidence": 0.0, "source_citation": None},
        "panel_name": {"value": None, "confidence": 0.0, "source_citation": None},
        "results": [
            {
                "test_name": "Potassium", "value": potassium_value, "unit": "mmol/L", "flag": flag,
                "confidence": 0.95, "source_citation": f"Potassium {potassium_value} mmol/L",
            },
        ],
    })


def _discharge_summary() -> DischargeSummary:
    return DischargeSummary.model_validate({
        "patient_name": {"value": "Maria Alvarez", "confidence": 0.95, "source_citation": "Patient Name: Maria Alvarez"},
        "date_of_birth": {"value": None, "confidence": 0.0, "source_citation": None},
        "medical_record_number": {"value": None, "confidence": 0.0, "source_citation": None},
        "admission_date": {"value": None, "confidence": 0.0, "source_citation": None},
        "discharge_date": {"value": None, "confidence": 0.0, "source_citation": None},
        "admitting_diagnosis": {"value": None, "confidence": 0.0, "source_citation": None},
        "discharge_diagnosis": {"value": None, "confidence": 0.0, "source_citation": None},
        "procedures_performed": {"value": [], "confidence": 0.0, "source_citation": None},
        "discharge_medications": [],
        "allergies": {"value": [], "confidence": 0.0, "source_citation": None},
        "discharge_condition": {"value": None, "confidence": 0.0, "source_citation": None},
        "follow_up_instructions": {"value": None, "confidence": 0.0, "source_citation": None},
        "follow_up_appointments": {"value": [], "confidence": 0.0, "source_citation": None},
    })


def _valid_comparison_output_data() -> dict:
    return {
        "narrative": "Potassium rose from 5.3 to 6.2 mmol/L, now in the critical range.",
        "trend": "worsening",
        "key_changes": [
            {
                "field": "Potassium",
                "prior_value": "5.3 mmol/L",
                "current_value": "6.2 mmol/L",
                "significance": "Now critical; requires immediate attention.",
            }
        ],
        "risk_flag": {"level": "critical", "rationale": "Potassium trending sharply toward critical."},
    }


def test_compare_success_first_attempt():
    current = _lab_report("6.2", "critical")
    prior = _lab_report("5.3", "high")
    client = FakeClient([FakeResponse(parsed_output=_valid_comparison_output_data())])

    card = compare(
        current, prior,
        current_filename="lab_current.txt", prior_filename="lab_prior.txt",
        client=client,
    )

    assert card.trend == Trend.WORSENING
    assert card.risk_flag.level.value == "critical"
    assert card.document_type == "lab_report"
    assert card.patient_display_name == "James Whitfield"
    assert len(card.key_changes) == 1
    assert card.key_changes[0].current_value == "6.2 mmol/L"
    assert client.messages.calls == 1


def test_compare_sends_both_documents_labeled():
    current = _lab_report("6.2", "critical")
    prior = _lab_report("5.3", "high")
    client = FakeClient([FakeResponse(parsed_output=_valid_comparison_output_data())])

    compare(
        current, prior,
        current_filename="lab_current.txt", prior_filename="lab_prior.txt",
        client=client,
    )

    sent_content = client.messages.last_kwargs["messages"][0]["content"]
    assert "CURRENT visit" in sent_content
    assert "PRIOR visit" in sent_content
    assert "lab_current.txt" in sent_content
    assert "lab_prior.txt" in sent_content


def test_compare_rejects_mismatched_document_types_without_calling_llm():
    current = _lab_report("6.2", "critical")
    prior = _discharge_summary()
    client = FakeClient([])  # should never be called

    with pytest.raises(ComparisonError):
        compare(
            current, prior,
            current_filename="lab_current.txt", prior_filename="discharge_prior.txt",
            client=client,
        )

    assert client.messages.calls == 0


def test_compare_retries_on_refusal_then_succeeds():
    current = _lab_report("6.2", "critical")
    prior = _lab_report("5.3", "high")
    client = FakeClient([
        FakeResponse(parsed_output=None, stop_reason="refusal"),
        FakeResponse(parsed_output=_valid_comparison_output_data()),
    ])

    card = compare(
        current, prior,
        current_filename="lab_current.txt", prior_filename="lab_prior.txt",
        client=client, max_attempts=3,
    )

    assert card.trend == Trend.WORSENING
    assert client.messages.calls == 2


def test_compare_retries_on_api_error_then_succeeds():
    current = _lab_report("6.2", "critical")
    prior = _lab_report("5.3", "high")
    client = FakeClient([FakeAPIError(), FakeResponse(parsed_output=_valid_comparison_output_data())])

    card = compare(
        current, prior,
        current_filename="lab_current.txt", prior_filename="lab_prior.txt",
        client=client, max_attempts=3,
    )

    assert card.trend == Trend.WORSENING
    assert client.messages.calls == 2


def test_compare_exhausts_retries_and_raises():
    current = _lab_report("6.2", "critical")
    prior = _lab_report("5.3", "high")
    client = FakeClient([
        FakeResponse(parsed_output=None, stop_reason="refusal"),
        FakeResponse(parsed_output=None, stop_reason="refusal"),
        FakeResponse(parsed_output=None, stop_reason="refusal"),
    ])

    with pytest.raises(ComparisonError):
        compare(
            current, prior,
            current_filename="lab_current.txt", prior_filename="lab_prior.txt",
            client=client, max_attempts=3,
        )

    assert client.messages.calls == 3


def test_comparison_card_rejects_mismatched_types_on_direct_construction():
    """The invariant is enforced on the model itself too, not just compare(),
    so reloading a malformed ComparisonCard (e.g. from a future audit-log
    path) can't silently bypass it."""
    from pydantic import ValidationError

    from schemas import ComparisonCard, ComparisonOutput

    comparison_output = ComparisonOutput.model_validate(_valid_comparison_output_data())

    with pytest.raises(ValidationError):
        ComparisonCard.from_comparison(
            current_filename="a.txt",
            prior_filename="b.txt",
            current_data=_lab_report("6.2", "critical"),
            prior_data=_discharge_summary(),
            comparison=comparison_output,
        )
