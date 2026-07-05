from schemas import DischargeSummary, SummarizationOutput, SummaryCard
from src.audit_log import get_run, list_runs, log_run


def _summary_card(source_filename: str = "discharge.txt") -> SummaryCard:
    extracted = DischargeSummary.model_validate({
        "patient_name": {"value": "Jane Doe", "confidence": 0.95, "source_citation": "Patient Name: Jane Doe"},
        "date_of_birth": {"value": None, "confidence": 0.0, "source_citation": None},
        "medical_record_number": {"value": None, "confidence": 0.0, "source_citation": None},
        "admission_date": {"value": None, "confidence": 0.0, "source_citation": None},
        "discharge_date": {"value": None, "confidence": 0.0, "source_citation": None},
        "admitting_diagnosis": {"value": "Pneumonia", "confidence": 0.9, "source_citation": "Dx: Pneumonia"},
        "discharge_diagnosis": {"value": None, "confidence": 0.0, "source_citation": None},
        "procedures_performed": {"value": [], "confidence": 0.0, "source_citation": None},
        "discharge_medications": [],
        "allergies": {"value": [], "confidence": 0.0, "source_citation": None},
        "discharge_condition": {"value": None, "confidence": 0.0, "source_citation": None},
        "follow_up_instructions": {"value": None, "confidence": 0.0, "source_citation": None},
        "follow_up_appointments": {"value": [], "confidence": 0.0, "source_citation": None},
    })
    summarization = SummarizationOutput.model_validate({
        "plain_language_summary": "Jane Doe was treated for pneumonia.",
        "risk_flag": {"level": "low", "rationale": "No critical findings."},
        "recommended_actions": [{"action": "Follow up with PCP", "priority": "routine"}],
    })
    return SummaryCard.from_summarization(
        source_filename=source_filename, extracted_data=extracted, summarization=summarization,
    )


def test_log_run_returns_incrementing_ids(tmp_path):
    db_path = tmp_path / "audit_log.db"
    card = _summary_card()

    first_id = log_run(card, model="claude-opus-4-8", db_path=db_path)
    second_id = log_run(card, model="claude-opus-4-8", db_path=db_path)

    assert second_id == first_id + 1


def test_list_runs_most_recent_first(tmp_path):
    db_path = tmp_path / "audit_log.db"
    log_run(_summary_card("first.txt"), model="claude-opus-4-8", db_path=db_path)
    log_run(_summary_card("second.txt"), model="claude-opus-4-8", db_path=db_path)

    entries = list_runs(db_path=db_path)

    assert [e.source_filename for e in entries] == ["second.txt", "first.txt"]
    assert entries[0].document_type == "discharge_summary"
    assert entries[0].patient_display_name == "Jane Doe"
    assert entries[0].risk_level == "low"
    assert entries[0].model == "claude-opus-4-8"


def test_list_runs_respects_limit(tmp_path):
    db_path = tmp_path / "audit_log.db"
    for i in range(5):
        log_run(_summary_card(f"file{i}.txt"), model="claude-opus-4-8", db_path=db_path)

    entries = list_runs(limit=2, db_path=db_path)

    assert len(entries) == 2
    assert entries[0].source_filename == "file4.txt"


def test_get_run_reloads_full_summary_card(tmp_path):
    db_path = tmp_path / "audit_log.db"
    card = _summary_card()
    run_id = log_run(card, model="claude-opus-4-8", db_path=db_path)

    reloaded = get_run(run_id, db_path=db_path)

    assert reloaded is not None
    assert isinstance(reloaded, SummaryCard)
    assert isinstance(reloaded.extracted_data, DischargeSummary)
    assert reloaded.patient_display_name == "Jane Doe"
    assert reloaded.plain_language_summary == card.plain_language_summary
    # Field-level grounding must survive the JSON round-trip too.
    assert reloaded.extracted_data.admitting_diagnosis.source_citation == "Dx: Pneumonia"


def test_get_run_returns_none_for_missing_id(tmp_path):
    db_path = tmp_path / "audit_log.db"
    log_run(_summary_card(), model="claude-opus-4-8", db_path=db_path)

    assert get_run(999, db_path=db_path) is None


def test_db_file_created_on_first_use(tmp_path):
    db_path = tmp_path / "nested" / "audit_log.db"
    assert not db_path.exists()

    log_run(_summary_card(), model="claude-opus-4-8", db_path=db_path)

    assert db_path.exists()
