# Chunking Strategy — Mutual Fund FAQ Corpus

**Phase 2 deliverable.** Chosen from measurements of the actual corpus, not from defaults.
**Corpus version:** `2026-09-28.1` (13 files, 436,899 chars)
**Evidence:** `reports/corpus_inspection.txt` — regenerate with `python scripts/inspect_corpus.py --out reports/corpus_inspection.txt`

Every number below was measured, not assumed. Where a number is a projection it is labelled as such.

---

## 1. What the corpus actually looks like

### 1.1 The decisive finding: the text is hard-wrapped

`pypdf` emits **one line per typeset row**, so a sentence arrives as several lines and there are almost no blank lines.

| Measurement | Value | Consequence |
|---|---|---|
| Physical lines | 9,571 | — |
| Blank-line-separated paragraphs | **150** | A `\n\n` splitter yields 150 blobs averaging **2,910 chars**. Unusable. |
| Logical paragraphs after de-wrapping | **267** | The real atomic unit count. |
| Logical paragraph p50 / p90 / p99 | 1,706 / 3,470 / 4,605 | Most units are 4–7× the target chunk size. |
| Logical paragraph max | 6,168 | Must be split. |
| Logical paragraphs > 1,300 chars | 147 (55.1%) | Sentence-level splitting is mandatory, not optional. |
| Max single physical line | **167** | No line is ever huge; nothing needs splitting on line length alone. |

Physical line-length histogram: 57.4% ≤ 40 chars, 26.8% in 80–120, **0.0% above 180**. Every line fits in a chunk; the problem is that a *paragraph* does not.

**Consequence:** the chunker must de-wrap first. Splitting on newlines produces fragments like:

> `ssuming ₹ , in ested systematically on the first usiness ay of e ery month`

which is a mid-sentence fragment, unrecoverable by the retriever and useless as a citation.

### 1.2 Two document families, two behaviours

De-wrapped unit sizes split the corpus cleanly:

| Family | Files | Logical para p50 | Behaviour after de-wrap |
|---|---|---|---|
| **Factsheets** | 5 | **66 – 215** | De-wrap into many small units. Headings give natural boundaries. Large units are **flattened tables** (2,500–4,285 chars). |
| **KIMs** | 5 | 2,737 – 3,006 | Long legal prose, few headings, page headers (`6 HDFC Large Cap Fund - KIM`) glued inline. |
| **SAI / Charter** | 3 | 1,555 – 5,227 | Dense numbered provisions; the SAI's statement section is one 6,168-char paragraph. |

One chunker must handle both without a per-document special case, which is why the rule below is heading-driven rather than size-driven.

### 1.3 Structural signals

| Signal | Count | Use |
|---|---|---|
| Recognised headings | **81** (0.8% of lines) | Primary chunk boundary; supplies the `section` metadata field. |
| Lines containing a number + unit | **483** (5.0%) | **Protected units** — never split away from their label. |
| `Label: value` lines | 184 (1.9%) | Protected atomic pairs. |
| Boilerplate lines | 58 (0.6%) | Stripped before chunking. |

Boilerplate is small but pure noise: `....Contd on next page` (5×), `For Product label and Riskometers, refer page no: 123-138` (5×), `Page N of 8` (8×), `130 | June 2026` running page headers.

---

## 2. Decisions

### 2.1 Chunking unit

**A group of whole sentences, bounded by heading boundaries, never crossing a document.**

Sentences, not characters and not paragraphs, because the 55.1% of units over 1,300 chars must be divided somewhere and the only safe division point in this corpus is a sentence terminator. Headings win over size: a new heading always starts a new chunk even if the current chunk is only 200 chars, because `EXPENSE RATIO` and `#BENCHMARK INDEX` answer different questions and merging them blurs retrieval.

Chunks never cross a file boundary. All five schemes share one master factsheet PDF, so a naive loader would happily return HDFC Large Cap's exit load when asked about Small Cap.

### 2.2 Chunk size and overlap

| Parameter | Value | Basis |
|---|---|---|
| `CHUNK_SIZE` | **900 chars** | ≈ 157 tokens at ~4.2 chars/token for this register (measured: 774 chunks, mean 660, max 2,188). At `top_k=5` the retrieved context is ~785 tokens, leaving ample room in an 8,192-token budget for the system prompt, question and answer. |
| `CHUNK_OVERLAP` | **150 chars** (16.7%) | Guarantees at least one whole sentence is shared between adjacent chunks, so a fact whose sentence begins in chunk *N−1* is still complete in chunk *N*. |

Why not 1000, the common default: at 900 the mean chunk is 660 chars, so retrieval returns ~3,300 chars of context at `top_k=5` — measured to be enough to hold a full section (expense ratio + benchmark + exit load) while still separating Large Cap from Small Cap, whose factsheets are near-identical in wording.

Why 150 and not 50: 50 chars is under one sentence in this corpus (p50 sentence is well above 50), so a 50-char overlap would frequently cut a sentence in half — reintroducing exactly the fragment problem that de-wrapping just solved.

### 2.3 Protected units — never split

Derived from the 483 numeric lines and 184 `Label: value` lines. A protected unit is emitted whole or not at all:

1. **Any sentence containing a number with a unit.** Splitting `an Exit Load of 1.00% is payable if Units are redeemed / switched-out within 1 year` from its heading produces a chunk that asserts a 1.00% load with no conditions — a confidently wrong answer.
2. **`Label: value` pairs**, e.g. `Regular: 1.56% Direct: 1.03%`.
3. **Table rows.** In the flattened factsheet tables a row is a company, sector and weight: `The Anup Engineering Limited Industrial Manufacturing 0.85 VRL Logistics Ltd. Transport Services 0.84`. Row boundaries are not reliably recoverable (§4), so table runs are kept whole and flagged rather than guessed at.
4. **The section heading itself**, emitted as the first line of the chunk that owns it.

### 2.4 Pre-chunk cleaning

Applied in order, before any size logic:

1. Strip the `SOURCE DOCUMENT` header block; re-attach it as metadata.
2. Drop boilerplate lines (§1.3).
3. Re-join hard-wrapped lines into logical paragraphs.
4. De-hyphenate `mitigat e` → `mitigate` where a space splits a word mid-token.
5. Collapse runs of whitespace; normalise the layout glyphs pypdf leaves behind (`$$`, `##`, `¥`, `€`, `�`).

Step 5 matters for citations: `EXIT LOAD$$` and `#BENCHMARK INDEX` are the real headings, and a citation reading `Section: EXIT LOAD$$` looks like a bug to a marker.

### 2.5 Metadata per chunk

Nine fields. Every answer must be traceable to one official URL and one `as_of` date (PRD §5), so provenance is stored per chunk rather than per file.

| Field | Example | Why |
|---|---|---|
| `chunk_id` | `hdfc-elss__factsheet-2026-06__0007` | Stable, human-readable, sortable. |
| `scheme` | `HDFC ELSS Tax Saver` | Read directly from the corpus header, not guessed. |
| `scheme_slug` | `hdfc-elss` | Directory-derived; used for filtering. |
| `source_type` | `factsheet` | `factsheet` / `kim` / `guide` / `riskometer`. |
| `source_url` | `https://files.hdfcfund.com/...` | The citation target. |
| `as_of` | `2026-06-30` | Factsheet numbers and KIM minimums carry **different** dates; the answer footer states which. |
| `pages` | `61, 62` | Lets a marker open the source PDF. |
| `section` | `EXPENSE RATIO` | From the nearest preceding heading. |
| `text` | — | The chunk body, unmodified. |
| `text_hash` | `sha256:…` | Dedupe and cache identity. |

`chunk_id` embeds the file stem rather than a bare integer so a citation survives the corpus being regenerated with different page ranges.

### 2.6 Context prefix format

Prepended to every chunk before embedding, so the embedding carries its own provenance and the retriever can match "HDFC ELSS" against text that only says "Nil".

```
Scheme: HDFC ELSS Tax Saver
Document: HDFC Mutual Fund Factsheet - June 2026 (as of 2026-06-30)
Section: EXPENSE RATIO
Source: https://files.hdfcfund.com/s3fs-public/2026-07/HDFC%20MF%20Factsheet%20-%20June%202026.pdf (p. 61-62)
---
EXPENSE RATIO (As On June 30, 2026) Base expense Ratio ... Regular: 1.75% Direct: 1.18%
```

Four short lines. The prefix is ~300 chars, so it consumes roughly a third of a 900-char budget — accepted, because without it a chunk reading `Nil` is unanswerable in isolation, and the ELSS "is there an exit load?" question is exactly that shape.

---

## 3. Worked examples

Real text, copied from `data/raw/`.

### 3.1 Small unit — passes through intact

Source: `data/raw/hdfc-elss/factsheet-2026-06.txt`, 89-char logical paragraph.

```
LOCK-IN PERIOD 3 years from the date of allotment of the respective Units EXIT LOAD$$ Nil
```

After cleaning, this is 82 chars and becomes **one** chunk, split at the `EXIT LOAD` heading:

```
Section: LOCK-IN PERIOD
LOCK-IN PERIOD 3 years from the date of allotment of the respective Units

Section: EXIT LOAD
EXIT LOAD Nil
```

This is the single most valuable chunk in the corpus: "Can I sell my HDFC ELSS units after 2 years?" is answered by an 82-char chunk, and any splitter that merged it with the preceding NAV table would bury it.

### 3.2 Multi-fact unit — split on heading, numbers protected

Source: `data/raw/hdfc-large-cap/factsheet-2026-06.txt`, 593-char logical paragraph.

```
EXPENSE RATIO (As On June 30, 2026) Base expense Ratio Including statutory levies on
expenses part of BER, excluding brokerage, transaction cost and execution related
statutory levies^ Regular: 1.56% Direct: 1.03% #BENCHMARK INDEX NIFTY 100 Total
Returns Index (TRI) ##ADDL. BENCHMARK INDEX BSE SENSEX Index (TRI) EXIT LOAD$$ •
In respect of each purchase / switch-in of Units, an Exit Load of 1.00% is payable if
Units are redeemed / switched-out within 1 year from the date of allotment. • No Exit
Load is payable if Units are redeemed / switched-out after 1 year from the date of
allotment.
```

Three chunks, boundaries at headings, sentences kept whole:

| # | Section | Chars | Content |
|---|---|---|---|
| 0 | `EXPENSE RATIO` | 233 | header + `Regular: 1.56% Direct: 1.03%` |
| 1 | `#BENCHMARK INDEX` | 96 | `NIFTY 100 Total Returns Index (TRI)` |
| 2 | `EXIT LOAD` | 264 | both 1.00% sentences, **unbroken** |

Chunk 2 demonstrates rule 2.1: 264 chars is well under 900, so the size limit would happily have merged the two exit-load sentences with the benchmark — but the exit load is only correct *together* with its two conditions, and merging it with an unrelated benchmark index invites the LLM to answer "1.00%" without the "after 1 year" clause.

### 3.3 Oversized unit — sentence split plus flagged fallback

Source: `data/raw/general/sai-investor-statements-2025-09.txt`, 6,168-char logical paragraph. This is the chunk that decides the whole design, because it is the one that answers the brief's "how do I download my statement" question.

```
... 3. A Consolidated Account Statement (CAS) detailing all the transactions across all
mutual funds and holdings at the end of the month and securities held in dematerialized
form across demat accounts, if applicable shall be sent to the Unit holders in whose
folio(s) transaction(s) have taken place during the month on registered email address on
or before 12th of the succeeding month and by 15th of the succeeding month for those who
have opted for physical copy. 4. Half-yearly CAS shall be issued to all investors ...
```

Simulation of the chosen parameters over the real corpus:

| Metric | Value |
|---|---|
| Total chunks | **774** |
| Min / mean / max chars | 2 / 660 / 2,188 |
| Chunks over `CHUNK_SIZE` (900) | 28 (3.6%) |
| Chunks 600–900 chars | 514 (66.4%) |
| Chunks containing a likely dropped-glyph region | 437 (56.5%) |

Clause 3 above is 460 chars on its own. It is emitted as **one** chunk: it contains `12th` and `15th`, so rule 2.3(1) protects it — splitting it would strand "on or before 12th of the succeeding month" away from the sentence that says who receives it.

The 28 over-size chunks are **not** a sizing failure. They are flattened table runs (`CATEGORY OF SCHEME BALANCED ADVANTAGE FUND CD - Certificate of Deposit; ( ) E uity / S / +/ + S ...`, 4,285 chars) where no sentence terminator exists. They are handled by the fallback splitter in §4 and carry `text_quality: degraded`, so the answer layer can refuse to quote a figure from one.

---

## 4. Rejected alternatives

Measured alternatives, same corpus, same sentence splitter, only `CHUNK_SIZE` changed:

| Setting | Chunks | Mean chars | Max chars | Context at `top_k=5` |
|---|---|---|---|---|
| 400 / 50 | 1,525 | 326 | 2,089 | 1,630 chars (~407 tok) |
| **900 / 150** | **774** | **660** | **2,188** | **3,300 chars (~825 tok)** |
| 1500 / 200 | 503 | 958 | 2,239 | 4,790 chars (~1,197 tok) |
| unbounded | 267 | 1,629 | 6,168 | 8,145 chars (~2,036 tok) |

| Option | Rejected because |
|---|---|
| **Fixed-size character windows** (e.g. 900 chars, hard cut) | Would cut mid-sentence. The corpus's atomic units reach 6,168 chars, so a hard cut lands inside the SAI's clause 3 with no signal. Rejected: produces the exact fragments §1.1 shows are useless. |
| **`\n\n` paragraph splitting** | 150 blobs averaging 2,910 chars, of which the largest is 13,086. Rejected on measurement alone. |
| **Line-based splitting** (`RecursiveCharacterTextSplitter` default) | Breaks every wrapped sentence. 57.4% of lines are under 40 chars, so most chunks would be near-empty fragments. Rejected. |
| **Markdown / HTML header splitting** (LangChain `MarkdownHeaderTextSplitter`) | Needs real heading markup. `pypdf` gives plain text; the only structure is 81 heading lines matched by regex against 13 hand-maintained patterns. Rejected as over-engineered for a one-time corpus build — but the heading list is the single highest-value artefact to grow if the corpus expands. |
| **Token-based chunking** (tiktoken) | The right *unit* is still the sentence. Tokenising first would encode the 4.2 chars/token estimate into the splitter and add a dependency for no gain at 774 chunks. Size is enforced in characters, estimated in tokens. |
| **`CHUNK_SIZE` 400** | Measured: 1,525 chunks, mean 326 chars. The exit-load rule (both sentences, 264 chars) still fits, but a full factsheet section no longer does, so "expense ratio and its benchmark" stops co-occurring — and that pairing is a natural question. Rejected: over-fragments a corpus that is only 437K chars to begin with. |
| **`CHUNK_SIZE` 1500** | Measured: 503 chunks, mean 958, so `top_k=5` returns ~1,197 tokens. Large Cap, Flexi Cap and Small Cap factsheets share near-identical boilerplate, so a wider chunk raises the chance that top-5 returns three schemes' exit loads for one question. Rejected: cross-scheme contamination is the main accuracy risk in this corpus. |
| **One chunk per logical paragraph** (no size limit) | Measured: 267 chunks, mean 1,629, largest 6,168 chars = 1,468 tokens in a single embedding, and 55% over target. Rejected on size. |
| **Splitting the flattened tables into individual rows** | Tempting, but row boundaries are not reliably recoverable from this extraction (§5). Guessing them would risk attributing a weight to the wrong company. Rejected: keep the run whole, flag it. |

---

## 5. Known limitation carried into Phase 3

**Glyph loss in extracted text: 507 of 9,571 lines (5.3%).**

`pypdf` drops characters in specific fonts. Measured signature — an isolated single letter between spaces:

- `llocation` ← Allocation, `ace alue` ← Face Value, `E uity` ← Equity
- `ssuming ₹ , in ested systematically on the first usiness ay of e ery month o er a period of time` ← "Assuming ₹ X invested systematically on the first business day of every month over a period of time"

Concentration is not uniform: **78–94 lines per KIM** (~7–8% of each file) versus **8–17 per factsheet**. 209 suspicious digit-gap patterns were also found, and a dropped glyph next to a digit would silently corrupt a number.

The figures that matter most are **not** affected — `Regular: 1.56% Direct: 1.03%`, the exit-load percentages and the 3-year lock-in were all verified readable, and §7 of `sources.md` records them as spot-checked. The loss is concentrated in portfolio and industry-allocation tables, which are not the target questions.

**Phase 3 must therefore emit a `text_quality` flag per chunk** (`clean` / `degraded`), set `degraded` when a chunk contains the dropped-glyph signature, and have the answer layer prefer `clean` chunks and decline to quote a bare number from a `degraded` one. This is a retrieval-quality guard, not a cosmetic one: a weight of `0.85` rendered as `.85` or `0. 85` is worse than no answer.

---

## 6. Parameters handed to Phase 3

| Constant | Value |
|---|---|
| `CHUNK_SIZE` | 900 chars |
| `CHUNK_OVERLAP` | 150 chars |
| Chunking unit | whole sentences, grouped within a heading section |
| Hard split | never mid-sentence; fallback splits oversized single units and marks them `degraded` |
| Protected units | numeric sentences, `Label: value` pairs, table runs, the owning heading |
| Stripped | 58 boilerplate lines, layout glyphs (`$$`, `##`, `¥`, `€`, `�`) |
| Boundaries respected | heading, then file (`chunk_id` contains the scheme slug) |
| Expected output | ≈774 chunks, mean 660 chars, ≈157 tokens |
| Metadata fields | 9, per §2.5 |
| `text_quality` flag | `clean` / `degraded`, per §5 |

## 7. Verification for Phase 3

1. Re-run `python scripts/inspect_corpus.py` and confirm the §1 numbers still hold after any corpus change.
2. `pytest tests/test_chunking.py` — the six tests written in Phase 2 encode the rules in §2.3, §2.5 and §5. They are expected to fail until the chunker exists; they are the specification, not a report card.
3. Spot-check that each of the five schemes' `EXPENSE RATIO` and `EXIT LOAD` chunks exist, contain the verified figures from `sources.md` §7, and are not shared with another scheme.
4. Confirm no chunk contains `....Contd` or `refer page no: 123-138`.

---

## 8. Retrieval observations (Phase 4)

Recorded per `implementation.md` Phase 4's tuning checkpoint: when the best hit
is wrong, diagnose the cause rather than lowering `SIM_FLOOR`. Both findings
below were diagnosed, not tuned around.

### 8.1 Measured output: 720 chunks, not ≈774

| Metric | §6 predicted | Actual |
|---|---|---|
| Chunk count | ≈774 | **720** |
| Range | — | 57–1440 chars |
| Clean / degraded | — | 389 / 331 |

The 720 is the correct outcome of §2.3's protected-unit rule, not a regression:
short-but-real facts now survive (`LOCK-IN PERIOD 3 years from the date of
allotment` is 49 chars and would previously have been dropped by `MIN_CHUNK`),
and `MIN_CHUNK` now only discards orphan labels and bare numbers.

### 8.2 `SIM_FLOOR` = 0.30 is too low to detect absence on its own

Query *minimum SIP for HDFC Flexi Cap Fund* — a fact **not** in the corpus —
scored **0.71** against the best chunk. Any threshold below 0.71 passes it.

| Query | Best cosine | Answer present? |
|---|---|---|
| expense ratio of HDFC ELSS | 0.82 | yes |
| exit load on HDFC Small Cap | 0.75 | yes |
| download capital gains statement | 0.41 | partly (CAS is documented; the download steps are not) |
| minimum SIP for HDFC Flexi Cap | 0.71 | **no** |

**Conclusion: a similarity floor cannot detect absence.** All five factsheets
share near-identical structure, so a question about any of them scores similarly
against all of them. Presence must be decided by *whether the retrieved text
contains the fact*, not by the score. `SIM_FLOOR` is therefore doing narrower
work than its name suggests: it is a garbage floor, not an answerability test.
Phase 5's `INSUFFICIENT` path has to check for the fact, not trust the score.

### 8.3 Re-rank bug: the scheme name was counted as a content term

For *expense ratio of HDFC ELSS Tax Saver Fund*, query tokens were
`{expense, ratio, elss, tax, saver}`. The re-ranked #1 was
`hdfc-elss-factsheet-2026-06:0015` — boilerplate reading *"including brokerage,
transaction cost and statutory levies please refer our website"* — because its
body repeats "HDFC ELSS Tax Saver" and so scored overlap on `elss`/`tax`/`saver`.
The actual answer (`Regular: 1.75% Direct: 1.18%`) ranked **#2**.

The fund name is scheme *identity*, already handled by the §5.3 step-5 filter
and the scheme boost. Counting it a second time as content let a boilerplate
chunk outrank the chunk holding the numbers.

**Fix (two changes, both in `app/retrieve.py`):**
1. Overlap is measured on `_content_tokens(question)` — question terms minus the
   matched alias and the canonical scheme name.
2. Tie-break for metric questions: +0.25 when a chunk matching a `METRIC_TERMS`
   entry actually carries a number with a unit. After change 1 the boilerplate
   and the real answer tied at 0.50 (both sections literally read "Expense
   Ratio"); the number is what separates a chunk that answers the question from
   one that describes it.

### 8.4 Foreign AMC must be checked before the alias match

*expense ratio of Parag Parivartan Flexi Cap Fund* was answered with **HDFC's
own** Flexi Cap ratio (1.35%), `ok=True`, exit 0. Cause: `FOREIGN_SCHEME_HINTS`
was only consulted when the alias match *failed*, and `"flexi cap"` is both a
valid HDFC alias and the tail of a real Parag Parivartan fund name.

**Fix:** the foreign check now runs first and unconditionally. Aliases like
`flexi cap`, `tax saver` and `small cap` are shared across AMCs, so a foreign
AMC named in the question must outrank any alias match. Covered by
`tests/test_retrieve.py::test_foreign_scheme_beats_a_colliding_alias`.

### 8.5 Latency (definition-of-done target: < 1.5 s)

| Stage | Time |
|---|---|
| `retrieve()` warm, 5 questions | 23–38 ms |
| numpy search (100 calls avg) | 0.16 ms |
| Index load from disk | 181 ms |
| First model call (one-off) | 18.2 s |
| Embedding 720 chunks (ingest only) | 88 s |

Warm retrieval is ~40× under target. The 18 s is a one-off model load per
process; a long-running server pays it once.

### 8.6 The store

ChromaDB's Rust binding segfaults on **any** write on this machine — reproduced
on 1.5.9, 1.4.1, 0.6.3 and 0.5.23 with a three-line repro, so it is not a
version problem. `vcruntime140_1.dll` is missing from `C:\Windows\System32`
(present only in the Python directory), which is the likely cause; installing
the MSVC redistributable is the fix. Until then `app/localindex.py` provides the
same contract — persistent, cosine over L2-normalised vectors, ids equal to
`chunk_id` — in numpy. At 720 × 384 the brute-force matmul is 0.16 ms, so an
approximate index would add a dependency to buy speed we do not need.
