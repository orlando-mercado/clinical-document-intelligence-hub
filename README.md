# Clinical Document Intelligence Hub

An AI prototype that ingests unstructured clinical documents (discharge summaries, lab reports)
and produces structured, confidence-graded, source-cited summaries with risk flags and
recommended next steps — reducing manual review burden for clinical/administrative staff. Built
as a 5-day proof-of-concept.

## Approach

Pipeline: **Document Loader → Schema-Constrained Extraction → Confidence & Source-Grounding →
Summarization & Recommendation → Streamlit UI**, fanning out to **PDF Export**, **Audit Log**, and
**Multi-Doc Comparison**.

1. **Document Loader** (`src/loader.py`) extracts raw text from `.txt` files and text-layer PDFs
   (`pdfplumber`, falling back to `pypdf`). Scanned/image PDFs and image files are explicitly
   rejected — see Assumptions.
2. **Schema-Constrained Extraction** (`src/extraction.py`) calls Claude with structured outputs,
   constrained to a Pydantic schema (`schemas/discharge_summary.py`, `schemas/lab_report.py`) that
   requires a `confidence` score and a verbatim `source_citation` for every field. A bounded retry
   loop (`src/llm.py`) handles safety refusals, API errors, and the numeric bounds structured
   outputs don't enforce server-side.
3. **Confidence & Source-Grounding**: confidence bands (🟢/🟡/🔴) are never trusted from the
   model's own self-report — they're recomputed deterministically from the numeric score and,
   critically, from whether the claimed citation is *actually* a verbatim (whitespace-normalized)
   substring of the source document. A hallucinated citation is caught and downgraded
   automatically (`schemas/common.py`, `src/extraction.py`).
4. **Summarization & Recommendation** (`src/summarization.py`) takes the already-extracted,
   already-graded data (not raw text) and produces a plain-language summary, a risk flag, and
   recommended next steps — explicitly instructed to treat low-confidence/ungrounded fields as
   uncertain, not fact.
5. **Multi-Doc Comparison** (`src/comparison.py`) compares two same-type documents (current vs.
   prior visit) for trend and *trajectory* risk — can escalate risk even when neither visit alone
   was flagged critical.
6. **Streamlit UI** (`app/main.py`) ties it together: upload → pick document type → process →
   color-coded summary card, PDF export, audit log (SQLite, every run), and a prior-visit picker
   for comparison.

## AI Models & Tools

- **Claude (`claude-opus-4-8`)** via the `anthropic` Python SDK, using structured outputs
  (`client.messages.parse(..., output_format=...)`) for extraction, summarization, and comparison.
- **Pydantic v2** for every schema — extraction output, summarization/comparison output, and the
  assembled `SummaryCard` / `ComparisonCard`.
- **Streamlit** for the UI.
- **pdfplumber** / **pypdf** for PDF text extraction; **reportlab** for PDF export.
- **SQLite** (stdlib `sqlite3`) for the audit log.

## Assumptions

- **Two document types only**: discharge summaries and lab reports (not intake forms or physician
  notes, though the pattern generalizes). See `docs/scope.md`.
- **No OCR / no image input.** A PDF must already have a text layer; scanned/image documents are
  rejected with a clear error rather than silently degrading to OCR or vision.
- **Document type is user-selected**, not auto-classified.
- **Confidence is grounded, not self-reported** — the model's own confidence label is only a
  starting point; the deciding factor is whether its citation is verifiably present in the source.
- **Synthetic data only** — no real or proprietary patient data (`data/synthetic/`).

## Setup

```bash
git clone <repo-url> && cd clinical-document-intelligence-hub
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then set ANTHROPIC_API_KEY
streamlit run app/main.py
```

## Example: Input → Output

**Input** — `data/synthetic/lab_report_whitfield_current.txt` (excerpt):

```
Patient Name: James Whitfield
Date of Birth: 11/05/1965
...
Test Panel: Comprehensive Metabolic Panel (CMP)
Results:
Potassium            6.2 mmol/L        (Reference: 3.5-5.0)          HIGH — CRITICAL
...
```

**Output** — live run against `claude-opus-4-8` (real generated `SummaryCard`, not hand-written):

> **Risk: 🔴 Critical**
> Potassium is flagged CRITICAL at 6.2 mmol/L (reference 3.5-5.0), a life-threatening value with
> risk of cardiac arrhythmia. This is compounded by significant renal impairment (eGFR 32,
> creatinine 2.1, BUN 48), which limits potassium clearance. All driving values are high-confidence.
>
> **Summary**
> James Whitfield (DOB 11/05/1965, MRN-117734) had a Comprehensive Metabolic Panel collected on
> 07/03/2026, ordered by Dr. Aisha Bello of Nephrology. The panel shows a critically high
> potassium of 6.2 mmol/L (reference 3.5-5.0), which is life-threatening and can cause dangerous
> heart rhythm problems. It also shows markedly reduced kidney function, with BUN 48 mg/dL,
> creatinine 2.1 mg/dL, and an eGFR of 32 mL/min/1.73m2, along with slightly low CO2 (22) and
> mildly elevated glucose (104). All results are high-confidence and grounded in the source
> document; the combination of severe hyperkalemia and impaired kidney function requires urgent
> attention.
>
> **Recommended Actions**
> - **[IMMEDIATE]** Notify the ordering provider (Dr. Bello, Nephrology) per critical-value protocol.
> - **[IMMEDIATE]** Obtain an urgent ECG to evaluate for hyperkalemic cardiac changes.
> - **[IMMEDIATE]** Repeat/confirm potassium (assess for hemolysis) while treatment is underway.
> - **[URGENT]** Arrange nephrology evaluation for reduced kidney function.
> - **[ROUTINE]** Follow up on mildly elevated glucose at the next routine visit.
>
> **Extraction Confidence: 🟢 High (100% avg)** — every field carried a verbatim source citation,
> e.g. `Potassium` is grounded in exactly
> `"Potassium            6.2 mmol/L        (Reference: 3.5-5.0)          HIGH — CRITICAL"`.

### Bonus: Multi-Doc Comparison (current vs. prior visit)

Comparing this run against `lab_report_whitfield_prior.txt` (3 weeks earlier, also a live run):

> **Trend: 📉🔴 Worsening** · **Trajectory Risk: 🔴 Critical**
> Potassium escalated from a mildly high 5.3 mmol/L to a critical 6.2 mmol/L, alongside worsening
> renal function (eGFR 45→32, creatinine 1.6→2.1) and new metabolic acidosis (CO2 24→22) — a clear
> worsening trajectory with an acutely dangerous electrolyte abnormality.

## Documentation & Tests

- `docs/scope.md` — locked scope and explicit non-goals.
- `/schemas`, `/src`, `/app` — see inline module docstrings for design rationale.
- `pytest` — 37 tests (all mock the Anthropic client except this README's captured live examples).
