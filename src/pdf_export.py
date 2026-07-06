"""PDF Export: renders a SummaryCard as a clinician-ready PDF artifact.

Uses reportlab's platypus flowables (Paragraph/Table/SimpleDocTemplate)
rather than raw canvas drawing — closer to a real document layout
(paragraphs, bulleted lists, a color-coded field table) and easier to
maintain than manual coordinate placement.
"""

from __future__ import annotations

import io

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from schemas import ConfidenceLevel, RiskLevel, SummaryCard
from src.formatting import iter_field_rows

RISK_COLOR = {
    RiskLevel.NONE: colors.grey,
    RiskLevel.LOW: colors.HexColor("#2e7d32"),
    RiskLevel.MODERATE: colors.HexColor("#b8860b"),
    RiskLevel.HIGH: colors.HexColor("#d2691e"),
    RiskLevel.CRITICAL: colors.HexColor("#c62828"),
}

CONFIDENCE_COLOR = {
    ConfidenceLevel.HIGH: colors.HexColor("#2e7d32"),
    ConfidenceLevel.NEEDS_REVIEW: colors.HexColor("#b8860b"),
    ConfidenceLevel.LOW: colors.HexColor("#c62828"),
}

DOCUMENT_TYPE_LABELS = {
    "discharge_summary": "Discharge Summary",
    "lab_report": "Lab Report",
}


def render_summary_card_pdf(card: SummaryCard) -> bytes:
    """Render a SummaryCard as a PDF and return its bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="Clinical Document Intelligence Hub — Patient Summary",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], alignment=TA_CENTER)
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Heading3"], alignment=TA_CENTER, textColor=colors.grey
    )
    h2_style = styles["Heading2"]
    body_style = styles["BodyText"]
    caption_style = ParagraphStyle("Caption", parent=styles["BodyText"], fontSize=8, textColor=colors.grey)
    warn_style = ParagraphStyle("Warn", parent=body_style, textColor=colors.HexColor("#b8860b"))

    story = [
        Paragraph("Clinical Document Intelligence Hub", title_style),
        Paragraph("Patient Summary", subtitle_style),
        Spacer(1, 0.2 * inch),
    ]

    doc_type_label = DOCUMENT_TYPE_LABELS.get(card.document_type, card.document_type)
    story.append(Paragraph(f"<b>{doc_type_label}</b> — {card.patient_display_name or 'Unknown patient'}", h2_style))
    story.append(
        Paragraph(
            f"Source: {card.source_filename} | Generated: {card.generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
            caption_style,
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    risk_style = ParagraphStyle("Risk", parent=h2_style, textColor=RISK_COLOR[card.risk_flag.level])
    story.append(Paragraph(f"Risk: {card.risk_flag.level.value.title()}", risk_style))
    story.append(Paragraph(card.risk_flag.rationale, body_style))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Summary", h2_style))
    story.append(Paragraph(card.plain_language_summary, body_style))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Recommended Actions", h2_style))
    if card.recommended_actions:
        story.append(
            ListFlowable(
                [
                    ListItem(Paragraph(f"<b>[{a.priority.upper()}]</b> {a.action}", body_style))
                    for a in card.recommended_actions
                ],
                bulletType="bullet",
            )
        )
    else:
        story.append(Paragraph("None specified.", body_style))
    story.append(Spacer(1, 0.15 * inch))

    conf_style = ParagraphStyle("Conf", parent=body_style, textColor=CONFIDENCE_COLOR[card.overall_confidence_level])
    overall_label = card.overall_confidence_level.value.replace("_", " ").title()
    story.append(
        Paragraph(
            f"<b>Extraction Confidence:</b> {overall_label} ({card.overall_confidence:.0%} average)", conf_style
        )
    )
    if card.fields_needing_review:
        story.append(
            Paragraph(
                "<b>Fields needing clinical review:</b> " + ", ".join(card.fields_needing_review), warn_style
            )
        )
    story.append(Spacer(1, 0.2 * inch))
    story.append(HRFlowable(width="100%", color=colors.lightgrey))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Extracted Fields (Source-Grounded)", h2_style))
    table_data = [["Field", "Value", "Confidence", "Source Citation"]]
    for label, display_value, node in iter_field_rows(card.extracted_data):
        if isinstance(display_value, list):
            display_value = ", ".join(str(v) for v in display_value) if display_value else None
        value_text = str(display_value) if display_value not in (None, "") else "not found"

        if node is None:
            confidence_para = Paragraph("—", caption_style)
            citation_para = Paragraph("—", caption_style)
        else:
            confidence_text = node.confidence_level.value.replace("_", " ").title()
            confidence_style = ParagraphStyle(
                f"conf_{node.confidence_level.value}", parent=body_style, textColor=CONFIDENCE_COLOR[node.confidence_level]
            )
            confidence_para = Paragraph(confidence_text, confidence_style)
            citation_para = Paragraph(node.source_citation or "no source citation", caption_style)

        table_data.append([Paragraph(label, body_style), Paragraph(value_text, body_style), confidence_para, citation_para])

    table = Table(table_data, colWidths=[1.6 * inch, 1.6 * inch, 1.0 * inch, 2.3 * inch], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(table)

    doc.build(story)
    return buffer.getvalue()
