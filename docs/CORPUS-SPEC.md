# Corpus and question set (synthetic)

The made-up company: **Harbor Goods**, a small online shop selling outdoor gear, shipping from one
warehouse. No real company, person, supplier or price list.

## The 20 documents (`data/corpus/<doc_id>.md`, 300 to 800 words each)

| # | doc_id | Topic | Planted facts (examples) |
|---|---|---|---|
| 1 | shipping-policy | Delivery promises | free-shipping threshold, handling time |
| 2 | shipping-rates-zones | Rates by zone | rate table, remote-area surcharge |
| 3 | carrier-rules | Two carriers, which to use when | weight limits, cut-off times |
| 4 | returns-policy | Returns (current) | window in days, who pays return shipping |
| 5 | returns-policy-2025 | Returns (old, superseded) | **decoy**: different window, marked superseded |
| 6 | refunds-chargebacks | Refund timing, disputes | refund days, evidence needed |
| 7 | warranty | Warranty terms | length per product class, exclusions |
| 8 | supplier-terms-alpine | Supplier A terms | payment terms net 30, min order |
| 9 | supplier-terms-trailworks | Supplier B terms | **decoy pair**: net 45, different min order |
| 10 | purchase-orders | How POs are raised and approved | approval limits |
| 11 | invoice-rules | Paying supplier invoices | three-way match, late fee |
| 12 | payment-methods | Customer payment options | accepted methods, fees |
| 13 | inventory-counts | Cycle counts SOP | count frequency, variance limit |
| 14 | receiving-goods | Receiving SOP | inspection steps, damage report window |
| 15 | packing | Packing SOP | box rules, fragile handling |
| 16 | damaged-goods | Damage handling SOP | photos required, replacement vs refund |
| 17 | support-hours-sla | Support hours and response times | hours, first-response target |
| 18 | data-retention | Privacy and retention | retention periods, deletion requests |
| 19 | holiday-schedule | Closures and cut-offs | dates, last order day |
| 20 | promo-price-rules | Promotions and price changes | stacking rules, approval |

Each document has a header with `doc_id`, `title`, `version`, `effective` date. Documents contain
specific numbers, dates and exceptions, so answers can be checked exactly. Two decoy pairs (4 and 5;
8 and 9) look alike on purpose.

### Four documents are PDFs

`supplier-terms-alpine`, `supplier-terms-trailworks`, `warranty` and `data-retention` are rendered
to PDF by `scripts/make_pdfs.py` and ingested through `POST /documents/pdf` (not also as markdown,
so there are no duplicates). Reasons: supplier contracts and warranties are PDFs in real life, and
the supplier pair is a decoy pair, so page-level citations matter.

Rules for the PDFs:

- Each PDF spans at least 2 pages (make the text long enough, add a page break before a named
  section) so that a page number is meaningful.
- Text-based PDFs only; use a standard font so text extraction is reliable. No scans, no images
  holding text.
- At least 4 questions have their key facts on page 2 or later.
- The markdown source of each PDF stays in `data/corpus/` as `<doc_id>.source.md`, and the PDF is
  written to `data/corpus/<doc_id>.pdf`. Ingest the PDF; the source file is for the generator and
  tests only.
- Keep one PDF where a quote crosses a line break in the layout, to prove whitespace
  normalisation works. Record any hyphenation or ligature problem as a known limit instead of
  changing the PDF to hide it.

## The 60 questions (`data/questions/questions.json`)

| Type | Count | What it tests |
|---|---|---|
| `single` | 40 | answer is in one document (includes 8 asking about the decoy pairs) |
| `multi` | 10 | needs two documents (e.g. late fee plus a supplier's payment terms) |
| `unanswerable` | 10 | the documents do not say; the correct result is `abstained` |

## Answer-key format

```json
{
  "id": "Q001",
  "type": "single",
  "question": "How many days do customers have to return unused items?",
  "expected_doc_ids": ["returns-policy"],
  "expected_quotes": ["Customers may return unused items within 30 days of delivery."],
  "expected_pages": [null],
  "must_contain": ["30"],
  "notes": "decoy: returns-policy-2025 says a different number"
}
```

`expected_pages` has one entry per expected quote: the 1-based page for PDF documents, `null` for
markdown documents. For `unanswerable`: `expected_doc_ids`, `expected_quotes` and `expected_pages`
are empty and `"type": "unanswerable"`.

Question mix for the three retrieval modes: include at least 8 questions where the exact term
matters (a rate, a limit, a code like a PO threshold), because that is where hybrid should help,
and at least 8 where the wording differs from the document, because that is where vector search
should help. Tag them with `"tags": ["exact-term"]` or `["paraphrase"]` so the README can report
each group. Do this before any measurement, not after.

## Rules for the generator (`scripts/make_corpus.py`)

- Repeatable: same output on every run (fixed seed, no timestamps in the data).
- A test checks every `expected_quote` appears verbatim in its document (for PDFs: in the text
  extracted from the page named in `expected_pages`, after whitespace normalisation), and every
  `unanswerable` question's key facts appear in no document.
- `scripts/make_pdfs.py` is repeatable too (no creation timestamps that change the extracted text).
- The answer keys are written by the generator, never by Claude at question time.
