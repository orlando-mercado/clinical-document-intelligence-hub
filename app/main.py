"""Streamlit UI: upload a clinical document, run the extraction +
summarization pipeline, and render a color-coded summary card.

Run with: streamlit run app/main.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow `streamlit run app/main.py` to find the src/ and schemas/ packages
# regardless of Streamlit's working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from schemas import ComparisonCard, ConfidenceLevel, ConfidenceMixin, RiskLevel, SummaryCard, Trend
from src.audit_log import get_run, list_runs, log_run
from src.comparison import ComparisonError, compare
from src.config import DEFAULT_MODEL
from src.extraction import DocumentType, ExtractionError, extract
from src.formatting import iter_field_rows
from src.loader import UnsupportedDocumentError, load_document
from src.pdf_export import render_summary_card_pdf
from src.summarization import SummarizationError, summarize

st.set_page_config(page_title="Clinical Document Intelligence Hub", page_icon="🩺", layout="wide")

CONFIDENCE_ICON = {
    ConfidenceLevel.HIGH: "🟢",
    ConfidenceLevel.NEEDS_REVIEW: "🟡",
    ConfidenceLevel.LOW: "🔴",
}

RISK_ICON = {
    RiskLevel.NONE: "⚪",
    RiskLevel.LOW: "🟢",
    RiskLevel.MODERATE: "🟡",
    RiskLevel.HIGH: "🟠",
    RiskLevel.CRITICAL: "🔴",
}

TREND_ICON = {
    Trend.IMPROVING: "📈🟢",
    Trend.STABLE: "🔵",
    Trend.WORSENING: "📉🔴",
    Trend.MIXED: "🟡",
    Trend.NOT_COMPARABLE: "⚪",
}

DOCUMENT_TYPE_LABELS: dict[DocumentType, str] = {
    "discharge_summary": "Discharge Summary",
    "lab_report": "Lab Report",
}


def _run_pipeline(raw_bytes: bytes, filename: str, document_type: DocumentType) -> SummaryCard:
    document = load_document(raw_bytes, filename=filename)
    extracted = extract(document, document_type)
    return summarize(extracted, source_filename=filename)


def _render_confidence_row(label: str, display_value, node: ConfidenceMixin | None) -> None:
    if node is None:
        st.markdown(f"⚪ **{label}**: {display_value}")
        return
    icon = CONFIDENCE_ICON[node.confidence_level]
    if isinstance(display_value, list):
        display_value = ", ".join(str(v) for v in display_value) if display_value else None
    text = f"{icon} **{label}**: " + (str(display_value) if display_value not in (None, "") else "_not found_")
    st.markdown(text)
    citation = f"“{node.source_citation}”" if node.source_citation else "_no source citation_"
    st.caption(f"{citation} — confidence {node.confidence:.2f}")


def _render_document_fields(extracted) -> None:
    for label, display_value, node in iter_field_rows(extracted):
        _render_confidence_row(label, display_value, node)


def _render_summary_card(card: SummaryCard, source_filename: str) -> None:
    st.subheader(
        f"{DOCUMENT_TYPE_LABELS[card.document_type]} — {card.patient_display_name or 'Unknown patient'}"
    )
    st.caption(f"Source: {source_filename}")

    risk = card.risk_flag
    st.markdown(f"### {RISK_ICON[risk.level]} Risk: {risk.level.value.title()}")
    st.write(risk.rationale)

    st.markdown("#### Summary")
    st.write(card.plain_language_summary)

    st.markdown("#### Recommended Actions")
    for action in card.recommended_actions:
        st.markdown(f"- **[{action.priority.upper()}]** {action.action}")

    overall_icon = CONFIDENCE_ICON[card.overall_confidence_level]
    overall_label = card.overall_confidence_level.value.replace("_", " ").title()
    st.markdown(f"#### Extraction Confidence: {overall_icon} {overall_label} ({card.overall_confidence:.0%} avg)")
    if card.fields_needing_review:
        st.warning(
            "Fields needing clinical review (low confidence or no source citation): "
            + ", ".join(card.fields_needing_review)
        )

    with st.expander("Extracted fields (source-grounded)", expanded=False):
        _render_document_fields(card.extracted_data)


def _render_comparison_card(card: ComparisonCard) -> None:
    st.subheader(
        f"Comparison — {card.patient_display_name or 'Unknown patient'} "
        f"({DOCUMENT_TYPE_LABELS.get(card.document_type, card.document_type)})"
    )
    st.caption(f"Current: {card.current_filename}  ·  Prior: {card.prior_filename}")

    st.markdown(f"### {TREND_ICON[card.trend]} Trend: {card.trend.value.replace('_', ' ').title()}")

    risk = card.risk_flag
    st.markdown(f"**{RISK_ICON[risk.level]} Trajectory Risk: {risk.level.value.title()}**")
    st.write(risk.rationale)

    st.markdown("#### What Changed")
    st.write(card.narrative)

    if card.key_changes:
        st.markdown("#### Key Changes")
        for change in card.key_changes:
            st.markdown(f"- **{change.field}:** {change.prior_value} → {change.current_value}")
            st.caption(change.significance)


def _render_recent_runs() -> None:
    entries = list_runs(limit=20)
    if not entries:
        st.caption("No runs logged yet.")
        return
    for entry in entries:
        icon = RISK_ICON[RiskLevel(entry.risk_level)]
        st.markdown(f"**#{entry.id}** {icon} {entry.source_filename}")
        st.caption(
            f"{DOCUMENT_TYPE_LABELS.get(entry.document_type, entry.document_type)} · "
            f"{entry.patient_display_name or 'Unknown patient'} · {entry.logged_at[:19]}"
        )


def _render_comparison_section(current_card: SummaryCard) -> None:
    current_run_id = st.session_state.get("summary_card_run_id")
    candidates = [
        entry
        for entry in list_runs(limit=50)
        if entry.document_type == current_card.document_type and entry.id != current_run_id
    ]

    with st.expander("Compare with a prior visit", expanded=False):
        if not candidates:
            st.caption(
                f"No other logged {DOCUMENT_TYPE_LABELS.get(current_card.document_type, current_card.document_type)} "
                "runs to compare against yet — process another one first."
            )
            return

        options = {
            f"#{e.id} — {e.source_filename} ({e.patient_display_name or 'unknown patient'}, {e.logged_at[:19]})": e.id
            for e in candidates
        }
        selected_label = st.selectbox("Prior run", list(options.keys()), key="comparison_prior_choice")

        if st.button("Compare", key="compare_button"):
            prior_card = get_run(options[selected_label])
            if prior_card is None:
                st.error("That run could no longer be found in the audit log.")
            else:
                with st.spinner(f"Comparing with {DEFAULT_MODEL}…"):
                    try:
                        comparison_card = compare(
                            current_card.extracted_data,
                            prior_card.extracted_data,
                            current_filename=current_card.source_filename,
                            prior_filename=prior_card.source_filename,
                        )
                        st.session_state["comparison_card"] = comparison_card
                    except ComparisonError as exc:
                        st.error(f"Comparison failed: {exc}")
                    except Exception as exc:
                        st.error(f"Unexpected error while comparing: {exc}")

        comparison_card = st.session_state.get("comparison_card")
        if comparison_card is not None:
            _render_comparison_card(comparison_card)


def main() -> None:
    st.title("🩺 Clinical Document Intelligence Hub")
    st.caption(
        "Upload a discharge summary or lab report (text or text-layer PDF — no OCR/image input; "
        "see docs/scope.md) to get a structured, confidence-graded, source-cited summary."
    )

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        st.info(
            "No ANTHROPIC_API_KEY found in the environment. Copy .env.example to .env and add "
            "your key before processing a document.",
            icon="⚠️",
        )

    document_type_label = st.selectbox("Document type", list(DOCUMENT_TYPE_LABELS.values()))
    document_type: DocumentType = next(
        dt for dt, label in DOCUMENT_TYPE_LABELS.items() if label == document_type_label
    )

    uploaded_file = st.file_uploader("Upload document", type=["txt", "pdf"])

    if uploaded_file is not None and st.button("Process document", type="primary"):
        with st.spinner(f"Extracting and summarizing with {DEFAULT_MODEL}…"):
            try:
                card = _run_pipeline(uploaded_file.getvalue(), uploaded_file.name, document_type)
            except UnsupportedDocumentError as exc:
                st.error(f"Couldn't load this document: {exc}")
                card = None
            except ExtractionError as exc:
                st.error(f"Extraction failed: {exc}")
                card = None
            except SummarizationError as exc:
                st.error(f"Summarization failed: {exc}")
                card = None
            except Exception as exc:
                # Last-resort safety net: e.g. a missing/invalid API key raises
                # a plain TypeError from the SDK, not an anthropic.APIError
                # subclass, so it isn't caught by ExtractionError/SummarizationError
                # above. Never let the user see a raw traceback.
                st.error(f"Unexpected error while processing this document: {exc}")
                card = None

        if card is not None:
            st.session_state["summary_card"] = card
            st.session_state["summary_card_filename"] = uploaded_file.name
            st.session_state["summary_card_run_id"] = None
            st.session_state.pop("comparison_card", None)
            try:
                run_id = log_run(card, model=DEFAULT_MODEL)
                st.session_state["summary_card_run_id"] = run_id
                st.toast(f"Logged as run #{run_id}", icon="🗒️")
            except Exception as exc:
                # Audit logging is a side effect, not the deliverable — a
                # write failure shouldn't hide the summary card just computed.
                st.warning(f"Couldn't write to the audit log: {exc}")

    stored_card = st.session_state.get("summary_card")
    if stored_card is not None:
        _render_summary_card(stored_card, st.session_state["summary_card_filename"])
        st.download_button(
            "Download PDF summary",
            data=render_summary_card_pdf(stored_card),
            file_name=f"{Path(stored_card.source_filename).stem}_summary.pdf",
            mime="application/pdf",
        )
        _render_comparison_section(stored_card)

    with st.sidebar:
        st.markdown("### Audit Log — Recent Runs")
        _render_recent_runs()


if __name__ == "__main__":
    main()
