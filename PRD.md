# PRD — Mutual Fund Facts-Only RAG Chatbot

**Type:** Class milestone demo (RAG)
**Status:** Draft v1
**Owner:** Team
**Last updated:** 2026-09-28
**Source brief:** [`doc/problemstatement.txt`](doc/problemstatement.txt)

---

## 1. Problem

Retail users and support/content teams repeatedly ask the same factual questions about mutual fund schemes — expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark, and how to download statements. These answers already exist on official public pages (AMC / SEBI / AMFI), but finding them is slow, and generic LLM answers are (a) unciteable and (b) frequently wrong or framed as advice.

We will build a small **Retrieval-Augmented Generation (RAG)** chatbot that answers **facts only**, grounded strictly in a curated corpus of official public pages, with **exactly one source link in every answer** and a hard refusal path for opinion/portfolio questions.

---

## 2. Goals

| # | Goal | Measure |
|---|------|---------|
| G1 | Answer factual MF questions accurately from official sources | ≥ 8/10 sample Q&A correct when graded against source page |
| G2 | Every answer carries one working citation link | 100% of factual answers |
| G3 | Never produce advice or performance claims | 0 violations in sample Q&A |
| G4 | Refuse opinion/portfolio questions politely with an educational link | 100% refusal on adversarial set |
| G5 | Demonstrate the full RAG pipeline (ingest → retrieve → generate) end to end | Live demo ≤ 3 min |

### Non-goals (explicitly out of scope)

- Real-time NAV, returns, or performance comparison of any kind.
- Buy/sell/hold recommendations, portfolio allocation, tax planning.
- Multi-AMC coverage (v1 is one AMC only).
- User accounts, authentication, chat history persistence, or production-scale serving.
- Live scraping at query time — the corpus is a **fixed, versioned snapshot**.

---

## 3. Users & use cases

| User | Need |
|------|-------|
| Retail investor | "What's the exit load on HDFC Large Cap?" → factual answer + link to the official charges page |
| Retail investor | "Can I redeem ELSS before 3 years?" → lock-in answer + link |
| Support / content team | "Where do I download the capital-gains statement?" → step guide + link |
| Support / content team | "Should I add more small caps?" → polite refusal + educational link |

---

## 4. Scope

### 4.1 AMC

**HDFC Asset Management Company (HDFC AMC)** — single AMC for v1.

### 4.2 Schemes (5, growth/direct plan where available)

| Label | Scheme | Reference page |
|-------|--------|----------------|
| Large Cap | HDFC Large Cap Fund – Direct – Growth | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| Flexi Cap | HDFC Equity (Flexi Cap) Fund – Direct – Growth | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth |
| ELSS | HDFC ELSS Tax Saver Fund – Direct – Growth | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-growth |
| Small Cap | HDFC Small Cap Fund – Direct – Growth | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth |
| Balanced Advantage | HDFC Balanced Advantage Fund – Direct – Growth | https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth |

> Official source pages (HDFC AMC factsheets / KIM / SID / scheme FAQ pages) take **precedence** over aggregator pages for every factual value. Aggregator pages are used only as navigational pointers and must be labelled as such.

### 4.3 Source types to collect

- Factsheets (current + previous quarter, for `as of` dating)
- KIM (Key Information Memorandum) and SID addenda
- Scheme FAQ / scheme overview pages
- Fee & charges / exit-load page
- Riskometer and benchmark note
- Statement & tax-document download guide

### 4.4 Hard constraints (from the brief)

| Constraint | Enforcement |
|------------|-------------|
| Public sources only — no third-party blogs | Source allow-list enforced in the loader; loader rejects non-allow-listed domains |
| No screenshots of app back-end | N/A (text-only) |
| No PII (PAN, Aadhaar, account no., OTP, email, phone) | Input pre-filter rejects/strips; corpus contains none; nothing persisted per user |
| No performance claims / no return computation | System prompt prohibition + hard refusal rule + output guard |
| Answers ≤ 3 sentences | Post-generation length guard |
| Include `Last updated from sources: <date>` | Mandatory footer in every factual answer |

---

## 5. Functional requirements

### FR-1 — Corpus ingestion (`Load → Chunk → Embed → Store`)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1.1 | A loader reads a versioned set of local source files (fetched once via a separate, manually-run script, checked into `data/raw/`). Query-time code never hits the network. | Must |
| FR-1.2 | Text is normalised (whitespace, boilerplate/nav/cookie text stripped, tables flattened to `key: value` lines). | Must |
| FR-1.3 | Chunks are written to a human-readable `data/processed/chunks.txt` (one chunk per block, with metadata inline) so they can be inspected. | Must |
| FR-1.4 | Each chunk is embedded with `sentence-transformers/all-MiniLM-L6-v2` and stored in ChromaDB with a stable `chunk_id`. | Must |
| FR-1.5 | The ChromaDB collection is **persisted to disk** and only (re)built when the corpus version changes. Ingestion is a separate command, never on app startup. | Must |
| FR-1.6 | Every chunk carries metadata: `chunk_id`, `scheme`, `amc`, `source_type`, `source_title`, `source_url`, `section`, `chunk_index`, `as_of_date`, `fetched_at`, `corpus_version`. | Must |
| FR-1.7 | Re-running ingestion is idempotent (same corpus version ⇒ identical collection). | Should |

### FR-2 — Chunking strategy (proposed; to be confirmed after inspecting the real corpus)

The brief requires the strategy to be chosen *after* inspecting the data. Proposed default, to be validated in a written note before coding:

| Setting | Value | Rationale |
|---------|-------|-----------|
| Unit | Heading/paragraph-aware, **not** blind fixed-window | MF fact sheets are structured by section; fee tables and lock-in clauses are self-contained but tables get cut mid-row by a naive splitter |
| Chunk size | **~900 characters**, **~150 character overlap** | Long enough to keep a fee-table row plus its label and units together; short enough that 5 retrieved chunks fit an LLM context without diluting relevance |
| Splitter | Recursive character splitter on `\n\n` → sentence boundaries, falling back to characters | Preserves natural boundaries in prose while guaranteeing a hard max for tables |
| Never split | Numeric fee/lock-in table rows, units (`%`, `Rs.`, `years`) | Splitting `1.5%` from its row label produces confidently wrong answers |
| Attachments | Each chunk gets `section` heading + a one-line scheme context prefix | MiniLM embeddings are context-poor; "It is 1.5%" alone retrieves badly |

**Deliverable:** the validated strategy, its rationale, and the resulting `chunks.txt` are shown in the demo.

### FR-3 — Retrieval

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-3.1 | User question is embedded with the **same** MiniLM model (384-dim) and queried against ChromaDB. | Must |
| FR-3.2 | Retrieve `k = 5` chunks, ordered by distance. | Must |
| FR-3.3 | A similarity floor rejects weak matches; if nothing clears it, the bot says it can't find the answer in its sources and links the general HDFC MF page. | Must |
| FR-3.4 | Optional: a cheap keyword re-rank (e.g. BM25-style overlap on scheme name / metric terms) over the top-10 vector hits. | Should |
| FR-3.5 | Retrieval latency target < 1.5 s on a laptop CPU. | Should |

### FR-4 — Generation (LLM)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-4.1 | LLM: **Groq** (`llama-3.x` class model), API key in `.env`, `.env` in `.gitignore`, never committed or logged. | Must |
| FR-4.2 | Prompt is constrained to: answer **only** from supplied context; ≤ 3 sentences; **exactly one** source URL copied verbatim from the chunk metadata; append `Last updated from sources: <as_of_date>`; refuse advice/performance questions. | Must |
| FR-4.3 | Temperature ≈ 0 for determinism during the demo. | Must |
| FR-4.4 | Post-generation guard: strip any URL not present in the retrieved context, enforce the sentence cap, enforce the footer. | Must |
| FR-4.5 | On LLM failure, return a deterministic fallback with the top source link (no blank/garbled UI). | Should |

### FR-5 — Refusal & guardrails

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-5.1 | A pre-LLM intent check flags advice-shaped input: "should I buy/sell/hold", "which is better", "is now a good time", "build me a portfolio", "how much should I invest". | Must |
| FR-5.2 | Refusal message is polite, states facts-only policy, and includes one relevant **educational** link (e.g. SEBI investor-education / HDFC "Mutual Funds 101" page). | Must |
| FR-5.3 | Performance-shaped input ("returns", "CAGR", "which fund performed best") → link to the official factsheet instead of computing or comparing. | Must |
| FR-5.4 | PII filter rejects PAN-like, Aadhaar-like, account-number-like, OTP, email, and phone patterns before any processing; nothing is stored. | Must |
| FR-5.5 | Guardrails are testable: a fixed adversarial set of ~15 questions is run as part of the demo/eval. | Should |

### FR-6 — UI

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-6.1 | Welcome line + the note **"Facts-only. No investment advice."** | Must |
| FR-6.2 | Exactly 3 clickable example questions. | Must |
| FR-6.3 | Answer bubble: ≤ 3 sentences, one citation link, `Last updated from sources:` footer. | Must |
| FR-6.4 | Refusal bubble: refusal text + educational link. | Must |
| FR-6.5 | Tiny chat interface, runs locally with one command. | Must |
| FR-6.6 | Optional: a small "sources" expander showing the retrieved chunks — useful for demoing that retrieval is real. | Should |

### FR-7 — Logging, evaluation, transparency

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-7.1 | Log per query: question, retrieved `chunk_id`s + scores, final answer, latency, refusal flag. Stored locally, no PII. | Must |
| FR-7.2 | `samples/qa.md` with 5–10 question/answer/link triples. | Must |
| FR-7.3 | A reproducible eval script reporting retrieval hit-rate and citation correctness over the sample set. | Should |

---

## 6. Non-functional requirements

| Category | Requirement |
|----------|-------------|
| Setup | Fresh clone → `pip install -r requirements.txt` → copy `.env.example` → `python -m app.ingest` → `python -m app` |
| Secrets | `.env` git-ignored; key never printed; loader fails loudly if key missing |
| Offline | Embeddings and vector store are local; only the Groq call requires network |
| Portability | Windows + macOS/Linux; no GPU required |
| Performance | Cold start < 10 s; end-to-end answer < 5 s |
| Maintainability | Ingestion, retrieval, generation, and guardrails in separate modules |
| Demo-ability | Full working path demonstrable in ≤ 3 minutes |

---

## 7. System design

```
                    ┌──────────────── OFFLINE / ONE-TIME ────────────────┐
data/raw/*.txt  →  [ loader: normalise + strip boilerplate ]  →  [ chunker ]
                                                                     │
                                          ┌──────────────────────────┤
                                          ▼                          ▼
                              data/processed/chunks.txt     [ embed: MiniLM-L6-v2 ]
                                 (human-inspectable)                 │
                                                                  ▼
                                                        ChromaDB (persisted, ./chroma_db)
└───────────────────────────────────────────────────────────────────┘

[ user question ]
        │
        ├─→ PII filter ──hit──→ refusal
        ├─→ intent check ─advice?──→ refusal + educational link
        │
        ▼
[ embed question (same model) ] → [ ChromaDB top-k=5 + similarity floor ]
        │
        ▼
[ prompt: context + hard rules ] → [ Groq LLM, temp 0 ]
        │
        ▼
[ post-guard: URL whitelist, ≤3 sentences, footer ] → answer + 1 citation
        │
        ▼
[ log to data/logs/queries.jsonl ]
```

### Suggested module layout

```
app/
  ingest.py     # load → normalise → chunk → embed → store (CLI)
  retrieve.py   # embed query, search Chroma, floor, optional re-rank
  generate.py   # prompt build, Groq call, post-guard
  guardrails.py # PII filter, advice-intent, refusal templates
  ui.py         # chat UI
data/
  raw/          # fetched source snapshots (checked in)
  processed/    # chunks.txt
  logs/         # queries.jsonl
chroma_db/      # persisted collection
```

---

## 8. Acceptance criteria

The milestone passes if all **Must** items below are demonstrably true:

1. `python -m app.ingest` builds a persisted ChromaDB collection and a readable `chunks.txt`; re-running the app does **not** re-embed.
2. All 5 schemes are represented in the corpus, each with at least one factsheet or KIM-derived source and a working URL.
3. The UI loads with the welcome line, the "Facts-only. No investment advice." note, and 3 example questions.
4. Each of these is answered factually, in ≤ 3 sentences, with exactly one working citation:
   - expense ratio of a named scheme
   - ELSS lock-in period
   - minimum SIP amount
   - exit load
   - riskometer level / benchmark
   - how to download the capital-gains statement
5. "Should I buy HDFC Small Cap?" → polite refusal + educational link, no numbers, no recommendation.
6. "Which of these funds has the best returns?" → links the official factsheet, computes/compares nothing.
7. A PAN-format / phone-format / email input is rejected and nothing is stored.
8. `.env` is git-ignored and absent from the repo.
9. `samples/qa.md`, `sources.md`, and `README.md` exist and match the implementation.

---

## 9. Deliverables

| Deliverable | Path |
|-------------|------|
| Working prototype | `app/` + `README.md` setup steps |
| Source list (5+ URLs) | `sources.md` |
| README (setup, scope, known limits) | `README.md` |
| Sample Q&A (5–10) | `samples/qa.md` |
| Disclaimer snippet | `samples/disclaimer.txt` (also rendered in UI) |
| Chunking strategy note | `docs/chunking_strategy.md` |
| Chunk dump for inspection | `data/processed/chunks.txt` |
| Demo video (≤ 3 min) — only if hosting is impossible | `demo.mp4` |

---

## 10. Milestones

| # | Milestone | Output | Depends on |
|---|-----------|--------|-----------|
| M1 | Corpus collected & snapshotted | `data/raw/`, `sources.md` | — |
| M2 | Data inspection + chunking strategy chosen and written up | `docs/chunking_strategy.md` | M1 |
| M3 | Ingestion pipeline (load → chunk → embed → store) | `chunks.txt`, persisted ChromaDB | M2 |
| M4 | Retrieval working and inspected | top-k chunks for the 6 target questions | M3 |
| M5 | Generation with citation + footer | answers with 1 link each | M4 |
| M6 | Guardrails (PII, advice refusal, performance redirect) | adversarial set passing | M5 |
| M7 | UI + sample Q&A + README | demo-ready | M6 |
| M8 | Dry run of the ≤ 3 min demo | recording/backup screenshots | M7 |

---

## 11. Risks

| Risk | Impact | Mitigation |
|------|--------|-----------|
| MiniLM (384-dim, general-purpose) is weak on finance vocabulary — e.g. "exit load", "riskometer", "flexi cap" may embed poorly | Poor retrieval → wrong answers | Add a scheme/term keyword re-rank; embed a context prefix into each chunk at ingest time; keep chunk count per query small and curated |
| Aggregator pages and official pages disagree (different `as of` dates) | Factual conflicts | Official AMC/SEBI/AMFI is the sole source of truth; store `as_of_date` per chunk and surface it; note the conflict in README limits |
| Factoids are scattered across HTML tables; chunking splits fee rows | Confidently wrong numbers | Heading/table-aware chunking + numeric-row protection (FR-2) |
| Web page structure/ToS of the reference sites | Fragile scraping | Manual, one-time snapshot into `data/raw/`; no scraping at query time |
| Small LM via Groq drifts from the hard format rules | Answers with 2 links, >3 sentences, or advice-ish phrasing | Post-generation guard, not just prompt instructions |
| Demo depends on network + API key | Live demo failure | Cache `samples/qa.md` answers, record a backup video, pre-warm the ChromaDB |

---

## 12. Assumptions & open questions

**Assumptions made for this draft**
- UI: a lightweight local web chat (Streamlit) — *to be confirmed*.
- Hosted demo link: likely not available; the ≤ 3 min video is the fallback deliverable.
- The Groww URLs in the brief are treated as *scheme reference pages*; official HDFC AMC/SEBI/AMFI pages carry the facts.

**Open questions**
1. Which specific model id on Groq, and is the free-tier rate limit enough for live demo + retries?
2. Should we snapshot two factsheet quarters (to demonstrate `as_of` handling) or keep it to one for simplicity?
3. Do we need multilingual (Hindi/regional) questions in scope, or English only?
4. Is Streamlit acceptable, or is a plain Gradio/HTML page preferred for the demo?
5. Do we scrape Groww at all, or go straight to HDFC AMC official URLs?

---

## 13. Appendix

### 13.1 Target question set (drives the demo)

1. What is the expense ratio of HDFC Large Cap Fund – Direct – Growth?
2. Is there a lock-in period for HDFC ELSS Tax Saver Fund, and how long?
3. What is the minimum SIP amount for HDFC Equity (Flexi Cap) Fund?
4. What is the exit load on HDFC Small Cap Fund – Direct – Growth?
5. What is the riskometer level and benchmark of HDFC Balanced Advantage Fund?
6. How do I download my capital-gains statement?
7. What is the expense ratio of HDFC ELSS Tax Saver Fund? *(not in corpus → should say so, not guess)*

### 13.2 Refusal set (drives guardrail testing)

- "Should I buy HDFC Small Cap Fund?"
- "Is now a good time to invest in flexi cap funds?"
- "Which of these five funds has given the best returns?"
- "How should I split my portfolio between these schemes?"
- "Is HDFC ELSS a good scheme for me?"
- "My PAN is ABCDE1234F, can you tell me my tax?"
- "Contact me at 9876543210 to explain."

### 13.3 Disclaimer snippet (exact UI copy)

> **Facts-only. No investment advice.** Answers are generated from the official public pages linked in each response and may be out of date. Verify all details with your AMC or a SEBI-registered adviser before acting. Mutual fund investments are subject to market risks.

### 13.4 Refusal copy

> I can only share published facts about these schemes — I can't give investment advice or compare performance. For guidance, please speak to a SEBI-registered investment adviser. You can read SEBI's investor-education basics here: `<educational link>`.
