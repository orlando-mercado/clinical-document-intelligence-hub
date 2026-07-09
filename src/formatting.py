"""Shared presentation helpers for rendering extracted clinical data — used
by both the Streamlit UI (app/main.py) and PDF export (src/pdf_export.py) so
the two don't drift on how fields are described or flattened.
"""

from __future__ import annotations

from typing import Optional, Union

from schemas import ConfidenceMixin, DischargeSummary, ExtractedField, LabReport


def describe_confidence_item(item: ConfidenceMixin) -> str:
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


class _NoSeparateValue:
    """Sentinel meaning "this row's label already carries its value" (a
    MedicationItem/LabResultItem description) — distinct from a plain
    ExtractedField whose `.value` is genuinely `None` (not found in the
    document). Consumers must not render both the same way."""

    def __repr__(self) -> str:
        return "NO_SEPARATE_VALUE"


NO_SEPARATE_VALUE = _NoSeparateValue()

FieldRow = tuple[str, object, Optional[ConfidenceMixin]]


def iter_field_rows(extracted: Union[DischargeSummary, LabReport]) -> list[FieldRow]:
    """Flatten a DischargeSummary/LabReport into (label, display_value, node)
    rows, handling all three field shapes present in the schemas:

    - ExtractedField[...] — names, dates, diagnoses, or a list of plain
      strings (e.g. procedures_performed); display_value is `.value`, which
      may legitimately be None/empty if the field wasn't found.
    - non-empty list[MedicationItem | LabResultItem] — one row per item,
      labeled "<field> — <item description>"; display_value is
      NO_SEPARATE_VALUE since the description already carries it — this is
      NOT "not found".
    - empty list[MedicationItem | LabResultItem] — one "None documented" row
      with node=None (nothing to grade).

    `node` is the ConfidenceMixin to read confidence/citation from, or None
    when there's nothing to grade.
    """
    rows: list[FieldRow] = []
    for field_name in type(extracted).model_fields:
        if field_name == "document_type":
            continue
        value = getattr(extracted, field_name)
        label = field_name.replace("_", " ").title()

        if isinstance(value, ExtractedField):
            rows.append((label, value.value, value))
        elif isinstance(value, list) and value and isinstance(value[0], ConfidenceMixin):
            for item in value:
                rows.append((f"{label} — {describe_confidence_item(item)}", NO_SEPARATE_VALUE, item))
        elif isinstance(value, list):
            rows.append((label, "None documented", None))
    return rows
