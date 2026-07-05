"""Audit Log: persists every pipeline run to SQLite (the pipeline diagram's
"Audit Log — SQLite/JSON, every run" stage).

The full SummaryCard is stored as a JSON blob (summary_card_json) for
complete fidelity and reload; a handful of other columns are flat and
indexed so runs can be listed/filtered without deserializing every row.

Logging is a separate, explicit step the caller (the Streamlit UI) takes
after a pipeline run completes — it is not baked into extract()/summarize(),
so those stay pure LLM calls with no filesystem side effects and stay easy
to unit-test in isolation.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from schemas import SummaryCard

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "audit_log.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    source_filename TEXT NOT NULL,
    document_type TEXT NOT NULL,
    patient_display_name TEXT,
    model TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    overall_confidence REAL NOT NULL,
    overall_confidence_level TEXT NOT NULL,
    fields_needing_review_count INTEGER NOT NULL,
    summary_card_json TEXT NOT NULL
);
"""

DbPath = Union[str, Path]


@dataclass
class AuditLogEntry:
    """Flat, queryable view of one run — no JSON deserialization needed just
    to list or filter runs. Use get_run(id) to reload the full SummaryCard."""

    id: int
    logged_at: str
    generated_at: str
    source_filename: str
    document_type: str
    patient_display_name: Optional[str]
    model: str
    risk_level: str
    overall_confidence: float
    overall_confidence_level: str
    fields_needing_review_count: int


def _connect(db_path: DbPath) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(_SCHEMA)
    return conn


def log_run(card: SummaryCard, *, model: str, db_path: DbPath = DEFAULT_DB_PATH) -> int:
    """Persist one pipeline run. Returns the new row's id."""
    conn = _connect(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO runs (
                logged_at, generated_at, source_filename, document_type,
                patient_display_name, model, risk_level, overall_confidence,
                overall_confidence_level, fields_needing_review_count, summary_card_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                card.generated_at.isoformat(),
                card.source_filename,
                card.document_type,
                card.patient_display_name,
                model,
                card.risk_flag.level.value,
                card.overall_confidence,
                card.overall_confidence_level.value,
                len(card.fields_needing_review),
                card.model_dump_json(),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def list_runs(*, limit: int = 50, db_path: DbPath = DEFAULT_DB_PATH) -> list[AuditLogEntry]:
    """Most recent runs first."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT id, logged_at, generated_at, source_filename, document_type,
                   patient_display_name, model, risk_level, overall_confidence,
                   overall_confidence_level, fields_needing_review_count
            FROM runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [AuditLogEntry(*row) for row in rows]
    finally:
        conn.close()


def get_run(run_id: int, *, db_path: DbPath = DEFAULT_DB_PATH) -> Optional[SummaryCard]:
    """Reload the full SummaryCard for one run, or None if run_id doesn't exist."""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT summary_card_json FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return SummaryCard.model_validate(json.loads(row[0]))
    finally:
        conn.close()
