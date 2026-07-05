# Synthetic sample documents

Entirely fictional patients/data — see [docs/scope.md](../docs/scope.md) (2 document types, no real/proprietary data).
Plain `.txt`, matching the locked input scope (text or PDF-text, no OCR).

Two patients, each with a **current** and **prior** visit, to exercise both single-document
extraction and the multi-doc comparison feature (current vs. prior visit):

- **Maria Alvarez** (`discharge_summary_alvarez_*.txt`) — second COPD exacerbation admission in
  4 months; the current visit adds a pneumonia co-diagnosis and a new pulmonology referral that
  wasn't needed on the prior visit, i.e. a visible escalation of care to compare against.
- **James Whitfield** (`lab_report_whitfield_*.txt`) — same metabolic panel three weeks apart,
  trending from mild hyperkalemia/stage 3 CKD to critical hyperkalemia with acute-on-chronic
  kidney injury (potassium 5.3 → 6.2 mmol/L, eGFR 45 → 32), to exercise the risk-flag and
  trend-comparison logic.
