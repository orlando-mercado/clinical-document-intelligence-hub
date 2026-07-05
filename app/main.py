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

from schemas import ConfidenceLevel, ConfidenceMixin, ExtractedField, RiskLevel, SummaryCard
from src.config import DEFAULT_MODEL
from src.extraction import DocumentType, ExtractionError, extract
from src.loader import UnsupportedDocumentError, load_document
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

DOCUMENT_TYPE_LABELS: dict[DocumentType, str] = {
    "discharge_summary": "Discharge Summary",
    "lab_report": "Lab Report",
}


def _run_pipeline(raw_bytes: bytes, filename: str, document_type: DocumentType) -> SummaryCard:
    document = load_document(raw_bytes, filename=filename)
    extracted = extract(document, document_type)
    return summarize(extracted, source_filename=filename)


def _describe_item(item: ConfidenceMixin) -> str:
    """Short label for a MedicationItem / LabResultItem row."""
    if hasattr(item, "test_name"):  # LabResultItem
        parts = [item.test_name]
        if item.value:
            parts.append(str(item.value))
        if item.unit:
            parts.append(item.unit)
        if item.reference_range:
            parts.append(f"(ref: {item.reference_range})")
        if getattr(item, "flag", None) is not None and item.flag.value != "unknown":
            parts.append(f"[{item.flag.value.upper()}]")
        return " ".join(parts)
    if hasattr(item, "name"):  # MedicationItem
        parts = [item.name]
        for attr in ("dosage", "frequency", "route"):
            value = getattr(item, attr, None)
            if value:
                parts.append(value)
        return " ".join(parts)
    return str(item)


def _render_confidence_row(label: str, display_value, node: ConfidenceMixin) -> None:
    icon = CONFIDENCE_ICON[node.confidence_level]
    if isinstance(display_value, list):
        display_value = ", ".join(str(v) for v in display_value) if display_value else None
    text = f"{icon} **{label}**: " + (str(display_value) if display_value not in (None, "") else "_not found_")
    st.markdown(text)
    citation = f"“{node.source_citation}”" if node.source_citation else "_no source citation_"
    st.caption(f"{citation} — confidence {node.confidence:.2f}")


def _render_document_fields(extracted) -> None:
    for field_name in type(extracted).model_fields:
        if field_name == "document_type":
            continue
        value = getattr(extracted, field_name)
        label = field_name.replace("_", " ").title()

        if isinstance(value, ExtractedField):
            _render_confidence_row(label, value.value, value)
        elif isinstance(value, list) and value and isinstance(value[0], ConfidenceMixin):
            st.markdown(f"**{label}**")
            for item in value:
                _render_confidence_row(_describe_item(item), None, item)
        elif isinstance(value, list):
            st.markdown(f"**{label}:** " + (", ".join(str(v) for v in value) if value else "_None documented_"))


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

    stored_card = st.session_state.get("summary_card")
    if stored_card is not None:
        _render_summary_card(stored_card, st.session_state["summary_card_filename"])


if __name__ == "__main__":
    main()
