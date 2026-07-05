import anthropic
import pytest

from schemas import ConfidenceLevel, DischargeSummary
from src.extraction import ExtractionError, extract, is_verbatim_citation
from src.loader import LoadedDocument

SAMPLE_TEXT = (
    "Patient Name: Jane Doe\n"
    "Date of Birth: 01/01/1980\n"
    "Admitting Diagnosis: Pneumonia\n"
)


def _valid_discharge_summary_data() -> dict:
    return {
        "patient_name": {"value": "Jane Doe", "confidence": 0.95, "source_citation": "Patient Name: Jane Doe"},
        "date_of_birth": {"value": "01/01/1980", "confidence": 0.9, "source_citation": "Date of Birth: 01/01/1980"},
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
    }


class _FakeAPIError(anthropic.APIError):
    """A minimal, constructible stand-in — the real APIError requires a live
    httpx request/response we don't want to build just to test retry logic."""

    def __init__(self, message: str = "fake api error"):
        self._message = message

    def __str__(self) -> str:
        return self._message


class _FakeResponse:
    def __init__(self, parsed_output=None, stop_reason: str = "end_turn"):
        self.parsed_output = parsed_output
        self.stop_reason = stop_reason


class _FakeMessages:
    def __init__(self, items):
        self._items = list(items)
        self.calls = 0
        self.last_kwargs: dict | None = None

    def parse(self, **kwargs):
        self.last_kwargs = kwargs
        item = self._items[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


class _FakeClient:
    def __init__(self, items):
        self.messages = _FakeMessages(items)


def test_is_verbatim_citation_tolerates_column_whitespace():
    source = "Potassium            6.2 mmol/L        (Reference: 3.5-5.0)"
    assert is_verbatim_citation("Potassium 6.2 mmol/L", source)
    assert not is_verbatim_citation("Sodium 999 mmol/L", source)


def test_extract_success_first_attempt():
    doc = LoadedDocument(text=SAMPLE_TEXT, source_filename="note.txt")
    valid = DischargeSummary.model_validate(_valid_discharge_summary_data())
    client = _FakeClient([_FakeResponse(parsed_output=valid)])

    result = extract(doc, "discharge_summary", client=client)

    assert isinstance(result, DischargeSummary)
    assert result.patient_name.value == "Jane Doe"
    assert client.messages.calls == 1
    assert client.messages.last_kwargs["output_format"] is DischargeSummary


def test_extract_accepts_dict_parsed_output():
    doc = LoadedDocument(text=SAMPLE_TEXT, source_filename="note.txt")
    client = _FakeClient([_FakeResponse(parsed_output=_valid_discharge_summary_data())])

    result = extract(doc, "discharge_summary", client=client)

    assert isinstance(result, DischargeSummary)
    assert result.patient_name.value == "Jane Doe"


def test_extract_retries_on_refusal_then_succeeds():
    doc = LoadedDocument(text=SAMPLE_TEXT, source_filename="note.txt")
    valid = DischargeSummary.model_validate(_valid_discharge_summary_data())
    client = _FakeClient([
        _FakeResponse(parsed_output=None, stop_reason="refusal"),
        _FakeResponse(parsed_output=valid),
    ])

    result = extract(doc, "discharge_summary", client=client, max_attempts=3)

    assert isinstance(result, DischargeSummary)
    assert client.messages.calls == 2


def test_extract_retries_on_api_error_then_succeeds():
    doc = LoadedDocument(text=SAMPLE_TEXT, source_filename="note.txt")
    valid = DischargeSummary.model_validate(_valid_discharge_summary_data())
    client = _FakeClient([_FakeAPIError(), _FakeResponse(parsed_output=valid)])

    result = extract(doc, "discharge_summary", client=client, max_attempts=3)

    assert isinstance(result, DischargeSummary)
    assert client.messages.calls == 2


def test_extract_exhausts_retries_and_raises():
    doc = LoadedDocument(text=SAMPLE_TEXT, source_filename="note.txt")
    client = _FakeClient([
        _FakeResponse(parsed_output=None, stop_reason="refusal"),
        _FakeResponse(parsed_output=None, stop_reason="refusal"),
        _FakeResponse(parsed_output=None, stop_reason="refusal"),
    ])

    with pytest.raises(ExtractionError):
        extract(doc, "discharge_summary", client=client, max_attempts=3)

    assert client.messages.calls == 3


def test_extract_downgrades_hallucinated_citation():
    doc = LoadedDocument(text=SAMPLE_TEXT, source_filename="note.txt")
    data = _valid_discharge_summary_data()
    data["admitting_diagnosis"] = {
        "value": "Pneumonia",
        "confidence": 0.95,
        "source_citation": "This exact sentence does not appear anywhere in the document.",
    }
    hallucinated = DischargeSummary.model_validate(data)
    assert hallucinated.admitting_diagnosis.confidence_level == ConfidenceLevel.HIGH

    client = _FakeClient([_FakeResponse(parsed_output=hallucinated)])
    result = extract(doc, "discharge_summary", client=client)

    assert result.admitting_diagnosis.source_citation is None
    assert result.admitting_diagnosis.confidence_level == ConfidenceLevel.LOW


def test_extract_keeps_genuinely_grounded_citation():
    doc = LoadedDocument(text=SAMPLE_TEXT, source_filename="note.txt")
    valid = DischargeSummary.model_validate(_valid_discharge_summary_data())
    client = _FakeClient([_FakeResponse(parsed_output=valid)])

    result = extract(doc, "discharge_summary", client=client)

    assert result.admitting_diagnosis.source_citation == "Admitting Diagnosis: Pneumonia"
    assert result.admitting_diagnosis.confidence_level == ConfidenceLevel.HIGH
