# Implementation Plan — Mutual Fund Facts-Only RAG Chatbot

**Guide for:** driving Cursor (or any AI coding agent) phase by phase.
**Source of truth:** [`architecture.md`](architecture.md) → [`PRD.md`](PRD.md) → [`doc/problemstatement.txt`](doc/problemstatement.txt)

---

## 0. How to use this document

Each phase below is a self-contained unit of work with its own gate. Work them **in order** — later phases depend on contracts established earlier.

**The loop, every phase:**

1. Open the phase.
2. Copy its **Cursor prompt** block (grey section) verbatim.
3. Let Cursor implement it, constrained to the files listed in *Touches*.
4. Run the **Verification** commands yourself. Read the output — do not accept Cursor's claim that it works.
5. Check the **Definition of Done**. If any box is unticked, send Cursor the *Fix loop* line at the end of the phase and repeat.
6. Commit. Only then open the next phase.

**Non-negotiables for the whole build** (state these once at the start, or paste them into every prompt — the *Ground rules* block below):

- Touch only the files listed in the phase's *Touches*. No refactors of untouched modules.
- No comments in code. Docstrings only where a function's contract is non-obvious.
- `.env` is never read into a log, never printed, never committed. Add secrets only via `.env.example`.
- Do not add dependencies that are not in `requirements.txt`. If one is genuinely needed, stop and say so.
- Match existing style: `snake_case` functions, `UPPER_SNAKE` config constants, dataclasses for records, type hints on all function signatures.
- Every phase must leave the app runnable. If a phase breaks the app, that phase is not done.

**Ground rules block (paste at the top of your first Cursor chat):**

```
We are building a facts-only RAG chatbot for mutual fund FAQs. Read PRD.md and
architecture.md first — they are the spec; do not deviate from them without telling me.
Implement strictly the phase I give you, touching only the files named in that phase.
No code comments. Type hints on all signatures. No new dependencies without asking.
Never hardcode a file path, model id, or API key — read them from app/config.py.
If anything in the spec is ambiguous, stop and ask rather than guessing.
When you finish, tell me the exact command to verify the phase and what output to expect.
```

---

## Phase map

| Phase | Milestone | Delivers | Gate (one-line proof) |
|-------|-----------|----------|----------------------|
| 0 | — | Project scaffold, config, deps | `python -m app` prints a banner |
| 1 | M1 | Source corpus snapshot + `sources.md` | `data/raw/` has ≥1 file per scheme |
| 2 | M2 | Data inspection + chunking strategy | `docs/chunking_strategy.md` written with real numbers |
| 3 | M3 | Ingestion pipeline | `chunks.txt` + persisted ChromaDB, idempotent |
| 4 | M4 | Retrieval | top-5 chunks for the 6 target questions |
| 5 | M5 | Generation + citation | answers ≤3 sentences, 1 URL, footer |
| 6 | M6 | Guardrails | adversarial set 100%, no false refusals |
| 7 | M7 | Chat UI | demo-ready, 3 example questions |
| 8 | M7 | Eval, samples, README, disclaimer | `scripts/eval.py` reports recall@k |
| 9 | M8 | Demo dry run | ≤3 min end-to-end, no network needed for retrieval |

---

## Phase 0 — Scaffold

**Objective.** Create the skeleton, config layer, and dependency set so every later phase has somewhere to land. Nothing functional.

**Touches.** `requirements.txt`, `.env.example`, `.gitignore`, `app/__init__.py`, `app/__main__.py`, `app/config.py`, `app/chunking.py` (dataclass only), `data/raw/.gitkeep`, `data/processed/.gitkeep`, `data/logs/.gitkeep`, `chroma_db/.gitkeep`, `docs/.gitkeep`, `samples/.gitkeep`, `tests/.gitkeep`, `scripts/.gitkeep`.

**Dependencies** (pin loosely, let the resolver work):

```
chromadb
sentence-transformers
groq
python-dotenv
streamlit
pypdf
beautifulsoup4
httpx
pytest
```

> Check availability before the build. If `streamlit` is unwanted, `gradio` is a drop-in (UI phase is the only consumer). If `pypdf` is a problem, save factsheets as text at fetch time instead.

**`app/config.py` contract.** Load `.env` via `python-dotenv`; expose every constant from architecture.md §7 with its default: `GROQ_API_KEY`, `GROQ_MODEL` (`llama-3.1-8b-instant`), `GROQ_TEMPERATURE` (`0.0`), `CHROMA_PATH` (`./chroma_db`), `CHROMA_COLLECTION` (`mf_facts_{corpus_version}`), `EMBED_MODEL` (`sentence-transformers/all-MiniLM-L6-v2`), `TOP_K` (`5`), `FETCH_K` (`10`), `SIM_FLOOR` (`0.30`), `CHUNK_SIZE` (`900`), `TAIL_OVERLAP` (`150`), `MAX_CONTEXT_CHARS` (`8000`), plus `CORPUS_VERSION`, `SCHEME_ALIASES`, and path constants for `data/raw`, `data/processed`, `data/logs`. Add a `summary()` that prints the config **with the API key masked**.

**`app/chunking.py` contract.** Define the frozen `Chunk` dataclass exactly as in architecture.md §4.3 (all 14 fields) plus a `to_dict()` and a `render_for_txt()` returning the 80-column-block format. No chunking logic yet.

**Verification**

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import app.config as c; print(c.summary())"
python -m app
```

**Expected.** `summary()` prints model ids and constants with the key shown as `GROQ_API_KEY=***`. `python -m app` prints a "scaffold OK, no UI yet" banner and exits 0.

**Definition of done**
- [ ] `.gitignore` contains `.env`, `chroma_db/*` (except `.gitkeep`), `data/logs/*`, `__pycache__/`, `.venv/`
- [ ] `.env.example` has every key with an **empty** value
- [ ] `git status` shows no `.env`
- [ ] `Chunk` dataclass has all fields from architecture.md §4.3
- [ ] `python -m app` exits 0

**Cursor prompt**

```
Phase 0: scaffold. Read architecture.md §3 (repo layout), §4.3 (Chunk contract),
and §7 (configuration) — implement exactly those, nothing more.

Create: requirements.txt, .env.example, .gitignore, app/__init__.py,
app/__main__.py, app/config.py, and app/chunking.py containing ONLY the frozen
Chunk dataclass from architecture.md §4.3 with to_dict() and render_for_txt().

app/config.py must expose every constant listed in architecture.md §7 with those
exact defaults, read from .env via python-dotenv. Add summary() which masks the
API key. Add the four SCHEME_* aliases for the schemes in PRD §4.2 (slug, display
name, category, and search aliases).

app/__main__.py should print a scaffold banner and exit 0 for now.
Do not write any chunking logic yet. Do not create any other files.
```

**Fix loop.** `python -m app` errors → send: *"Run `python -m app` yourself, read the traceback, and fix the cause. Don't change the contracts."*

---

## Phase 1 — Corpus collection (M1)

**Objective.** Assemble the offline snapshot. **This phase is research, not code** — the most common failure mode in this project is building a pipeline over an empty or wrong corpus.

**Touches.** `data/raw/**`, `sources.md`, `scripts/fetch_sources.py`.

**Steps.**
1. For each of the 5 schemes in PRD §4.2, find on the **official HDFC AMC / AMFI / SEBI** domain: the latest factsheet, the scheme page (fees, exit load, minimum SIP), the riskometer note, and the benchmark line.
2. Also grab 2 cross-scheme pages: HDFC's fees/exit-load explainer, and a statement/tax-download guide.
3. Save each as plain text: `data/raw/<scheme-slug>/<source_type>-<as_of>.txt`. Filename **sort order defines chunk order**, so zero-pad dates.
4. For factsheet PDFs, extract text (`pypdf`) and check it reads in order — column-wise PDF extraction produces garbage that will poison every chunk. If extraction is garbled, transcribe the relevant table by hand into a `.txt` file; that is faster and more reliable.
5. Write `sources.md`: one row per source — scheme, source type, title, URL, `as of` date, local file. This is a graded deliverable (PRD §9).
6. Write `scripts/fetch_sources.py` as a **manual, documented** tool (allow-listed domains, `--dry-run` default). It is not imported by the app.

**Target coverage (hard floor, asserted in Phase 3)**

| Scheme | factsheet | fees/exit load | min SIP | lock-in | riskometer | benchmark |
|--------|-----------|---------------|---------|---------|------------|-----------|
| Large Cap | ✓ | ✓ | ✓ | n/a | ✓ | ✓ |
| Flexi Cap | ✓ | ✓ | ✓ | n/a | ✓ | ✓ |
| ELSS | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Small Cap | ✓ | ✓ | ✓ | n/a | ✓ | ✓ |
| Balanced Advantage | ✓ | ✓ | ✓ | n/a | ✓ | ✓ |

**Verification.** Manually open 3 random files in `data/raw/` and confirm the fact you expect is actually there and readable. Run `python scripts/fetch_sources.py --dry-run` and check it lists only allow-listed URLs.

**Definition of done**
- [ ] ≥1 file per scheme; all 5 schemes covered
- [ ] Every ELSS lock-in and every exit-load figure is present in *some* file, findable by Ctrl+F
- [ ] `sources.md` complete, one row per file, all URLs live (open 2 to confirm)
- [ ] No third-party blog content in `data/raw/`

**Cursor prompt**

```
Phase 1: corpus collection. This is research, not code.

From PRD §4.2, for each of the 5 HDFC AMC schemes, identify official public pages
(HDFC AMC / AMFI / SEBI only — no blogs, no aggregators as fact sources): latest
factsheet, scheme fees/exit-load page, minimum SIP, ELSS lock-in, riskometer, benchmark.
Plus 2 cross-scheme pages: HDFC fees explainer and a statement/tax-download guide.

I will save each as data/raw/<scheme-slug>/<source_type>-<as_of>.txt.
Write sources.md as a markdown table: scheme | source type | title | url | as of | local file.

Also write scripts/fetch_sources.py — a manual tool with a domain allow-list,
--dry-run as the default, that is never imported by the app.

Do NOT start building the pipeline. If a fact is missing from the corpus, tell me
exactly which scheme and which fact, instead of writing code around the gap.
```

**Fix loop.** Missing facts → *"A scheme is missing a required fact. List exactly which, and give me the official URL to add. Don't proceed until the corpus table is full."*

---

## Phase 2 — Data inspection + chunking strategy (M2)

**Objective.** Satisfy the brief's explicit requirement: *inspect the data, then propose a strategy with rationale, size, overlap, and per-chunk metadata — before writing any code.* The strategy is an academic deliverable; produce it deliberately.

**Touches.** `scripts/inspect_corpus.py`, `docs/chunking_strategy.md`, `tests/test_chunking.py`.

**Steps.**
1. Build `scripts/inspect_corpus.py`: loads `data/raw/**`, reports per-file char count, average paragraph length, section-heading count, table-row count, and **the 15 longest lines** (which reveal whether table rows survive as single units).
2. Read `data/processed/` samples and note where facts are dense.
3. Choose `CHUNK_SIZE` / `TAIL_OVERLAP` from the **observed** data, not the defaults. A factsheet whose fee table has 6-column rows will need a larger size than a prose FAQ page.
4. Write `docs/chunking_strategy.md` with: the inspection findings (real numbers), the chosen size/overlap, the unit of chunking, the protected-unit rule, the metadata list, the context-prefix format, **and 2 worked chunk examples copied from real output**. Include a short "what we tried and rejected" (e.g. fixed 500-char windows) — graders reward the reasoning.
5. Write the chunking tests now, in the same phase, so Phase 3 implements to a spec rather than inventing one.

**Verification**

```powershell
python scripts/inspect_corpus.py
python -m pytest tests/test_chunking.py -v
```

**Expected.** The inspect report prints per-file stats. The chunking tests **fail** (logic not implemented) — that's correct; they encode the spec. Note which ones fail and carry them into Phase 3.

**Definition of done**
- [ ] `docs/chunking_strategy.md` cites real numbers from the real corpus
- [ ] It states chunk size, overlap, unit, protected units, metadata, prefix format
- [ ] It contains ≥2 real chunk examples
- [ ] `tests/test_chunking.py` exists with ≥6 tests (size cap, no table-row split, stable `chunk_id` across runs, no empty chunks, no nav boilerplate, overlap respected) and currently fails for the right reason

**Cursor prompt**

```
Phase 2: data inspection and chunking strategy. Architecture §4.2 and PRD FR-2.

Step 1 — write scripts/inspect_corpus.py: per-file char count, average paragraph
length, heading count, table-row count, and the 15 longest lines. Run it on data/raw.

Step 2 — from the ACTUAL numbers it prints, write docs/chunking_strategy.md:
inspection findings, chosen chunk size and overlap (with justification tied to those
numbers), chunking unit, protected units (table rows, numbers with units), the metadata
list, the context-prefix format, 2 worked examples from real corpus text, and a
"rejected alternatives" section. Do not copy the defaults from architecture.md §7
without justifying them against what the inspection shows.

Step 3 — write tests/test_chunking.py with at least these 6 tests. They are allowed
to fail now; they are the spec for Phase 3:
  - every chunk <= CHUNK_SIZE
  - no table row / number-with-unit is split across a boundary
  - chunk_id is stable across two runs over the same input
  - no empty or whitespace-only chunks
  - nav/boilerplate lines never appear in chunk text
  - consecutive chunks share TAIL_OVERLAP characters of context

Do not implement chunking logic yet. Do not touch app/ingest.py.
```

**Fix loop.** Strategy lacks real numbers → *"Your strategy section has no measured values. Re-run inspect_corpus.py and rewrite the justification citing its actual output."*

---

## Phase 3 — Ingestion pipeline (M3)

**Objective.** Load → normalise → chunk → embed → store, plus the human-readable chunk dump and idempotency.

**Touches.** `app/chunking.py` (add logic), `app/ingest.py`, `app/normalise.py`, `app/store.py`, `tests/test_chunking.py` (should now pass), `tests/test_ingest.py`.

**Steps.** Implement in this order, verifying each:
1. `app/normalise.py` — strip nav/cookie/footer boilerplate, collapse whitespace, flatten tables to `Label: value`, normalise dashes/quotes, drop zero-width chars. Pure functions, easy to test.
2. `app/chunking.py` — the section-tree walk from architecture.md §4.2. Now make the Phase 2 tests pass.
3. `app/store.py` — Chroma `PersistentClient`, collection with `metadata={"hnsw:space": "cosine"}`, `upsert` by `chunk_id`. **Add the startup assertion** from architecture.md §13.5: verify `collection.metadata` actually reports cosine, or raise — a silent L2 fallback invalidates `SIM_FLOOR`.
4. `app/ingest.py` — CLI (`python -m app.ingest`, flags `--force`, `--dry-run`). Order: load → normalise → chunk → **write `chunks.txt`** → coverage check → embed (batch 64, `normalize_embeddings=True`) → upsert → print a report. Assert `EMBED_MODEL` is unchanged from config.
5. `tests/test_ingest.py` — coverage assertion (all 5 schemes, ≥1 `fees` and ≥1 `guide` source, exits non-zero otherwise); idempotency test (same corpus version ⇒ identical `chunk_id` set and identical collection count).

**Verification**

```powershell
python -m app.ingest
python -m app.ingest                 # re-run: must report "unchanged" or upsert, never duplicate
python -c "from chromadb import PersistentClient; c=PersistentClient(path='chroma_db'); print(c.get_collection('mf_facts_2026-09-28.1').count())"
Get-Content data\processed\chunks.txt -TotalCount 40
python -m pytest tests/ -v
```

**Expected.** A report with document count, chunk count, chunks-per-scheme, and coverage; all tests green; `chunks.txt` shows real HDFC facts with `url` and `as of` on every block.

**Manual gate — do not skip.** Open `chunks.txt` and read 10 chunks from different sources. For each, confirm you could answer "expense ratio of the ELSS plan" from the chunk *alone*. Any chunk that reads as a context-free fragment is a **recall bug discovered before the demo**, not after. Note anything odd in `docs/chunking_strategy.md`.

**Definition of done**
- [ ] All Phase 2 tests now pass
- [ ] `chunks.txt` is human-readable with full metadata headers
- [ ] Re-running ingest does not duplicate or change `chunk_id`s
- [ ] Cosine assertion present and passing
- [ ] Coverage check exits non-zero on an incomplete corpus
- [ ] 10 sampled chunks each answer their question standalone
- [ ] Collection count matches chunk count in `chunks.txt`

**Cursor prompt**

```
Phase 3: ingestion pipeline. Implement architecture §4.1 stages 1-6, following the
strategy in docs/chunking_strategy.md (that file is the spec, not architecture §7 defaults).

Files: app/normalise.py (pure text cleaning), app/chunking.py (the section-tree chunker
from architecture §4.2 — make the Phase 2 tests pass), app/store.py (ChromaDB persistent
collection, cosine space, upsert by chunk_id, PLUS a startup assertion that
collection.metadata actually reports cosine and raise if not), app/ingest.py (CLI).

app/ingest.py order: load -> normalise -> chunk -> write data/processed/chunks.txt ->
coverage check -> embed (batch 64, normalize_embeddings=True) -> upsert -> print report.
The coverage check must exit non-zero unless all 5 schemes and at least one 'fees' and
one 'guide' source are present.

Flags: --force to rebuild, --dry-run to chunk and report without embedding.
Reading a single model instance and reusing it for chunks. Assert EMBED_MODEL is the
configured value; never hardcode a model id.

Then add tests/test_ingest.py: idempotency (same corpus => identical chunk_id set) and
the coverage exit-code test. Run pytest and the ingest CLI, then report actual output.
```

**Fix loop.** Chunk count explodes / table rows split → *"Chunks are splitting table rows and the count is implausibly high. Re-read §4.2's protected-unit rule and fix the chunker; do not tune the size to hide it."*

---

## Phase 4 — Retrieval (M4)

**Objective.** Top-5 relevant chunks, scheme-aware, with a floor and INSUFFICIENT handling. No LLM yet.

**Touches.** `app/retrieve.py`, `app/embedder.py`, `tests/test_retrieve.py`.

**Steps.**
1. `app/embedder.py` — a single cached `SentenceTransformer(config.EMBED_MODEL)` used by both ingest and query. `lru_cache` on the getter so the model loads once per process.
2. `app/retrieve.py` — implement architecture.md §5.3 steps 1–7 exactly, in that order:
   - embed question (normalised) → `collection.query(n=FETCH_K)` → floor at `SIM_FLOOR` → scheme filter (alias match; if the filter empties the set, return INSUFFICIENT, **do not** fall back to other schemes) → re-rank → top `TOP_K`.
   - `Context` dataclass: `chunks: list[Chunk]`, `scores: list[float]`, `ok: bool`, `reason: str`.
   - Truncate to `MAX_CONTEXT_CHARS` as a hard stop.
3. The lexical re-rank: overlap score on question terms, with boosts for a scheme-name hit, a metric term (`expense ratio`, `exit load`, `SIP`, `lock-in`, `riskometer`, `benchmark`), and a source-type match.
4. A `__main__` block: `python -m app.retrieve "expense ratio of HDFC ELSS Tax Saver Fund" --debug` printing each hit's `chunk_id`, section, score, and a 200-char preview.
5. `tests/test_retrieve.py` — a question about a scheme **not** in the corpus returns `ok=False`; a question naming a scheme returns only that scheme's chunks; a nonsense question returns `ok=False`.

**Verification**

```powershell
python -m app.retrieve "expense ratio of HDFC ELSS Tax Saver Fund" --debug
python -m app.retrieve "exit load on HDFC Small Cap Fund" --debug
python -m app.retrieve "how do I download my capital gains statement" --debug
python -m app.retrieve "expense ratio of Parag Parivartan Flexi Cap Fund" --debug
python -m pytest tests/test_retrieve.py -v
```

**Expected.** Queries 1–3 return 3–5 chunks whose text actually contains the answer; the score of the best hit should be clearly above the rest. Query 4 (a **different AMC**) returns `ok=False`. This last one is the floor's whole purpose.

**Tuning checkpoint.** If the best hit for query 1 is not the ELSS expense-ratio chunk, do not lower `SIM_FLOOR` and move on. Diagnose: missing context prefix? term mismatch (`expense ratio` vs `expense ratio (%)`)? competing Large-Cap chunk? Record the finding in `docs/chunking_strategy.md` under *Retrieval observations* — this is the kind of judgement the demo should show.

**Definition of done**
- [ ] All 3 target queries return the correct source in top-5
- [ ] The off-corpus query returns `ok=False`
- [ ] Scheme filter never mixes schemes for a named-scheme question
- [ ] `EMBED_MODEL` identical in ingest and query (one `embedder.py`, not two)
- [ ] Retrieval latency < 1.5 s
- [ ] `--debug` output is legible enough to screenshot in the demo

**Cursor prompt**

```
Phase 4: retrieval. Implement architecture §5.3 steps 1-7 in order, nothing more.
No LLM calls in this phase.

Create app/embedder.py exposing one cached SentenceTransformer(config.EMBED_MODEL)
shared by ingest and query, and app/retrieve.py returning a Context dataclass
(chunks, scores, ok, reason).

Order: embed question with normalize_embeddings=True -> collection.query(n=FETCH_K)
-> drop hits below SIM_FLOOR -> if the question names a scheme, filter to that
scheme ONLY (if the filter empties the set return ok=False, never fall back to other
schemes) -> lexical re-rank (question-term overlap, with boosts for a scheme-name hit,
a metric term from {expense ratio, exit load, SIP, lock-in, riskometer, benchmark}, and
a source-type match) -> take TOP_K -> hard-truncate to MAX_CONTEXT_CHARS.

Add a __main__ so `python -m app.retrieve "<question>" --debug` prints per-hit chunk_id,
section, score and a 200-char preview. Add tests/test_retrieve.py covering: off-corpus
scheme returns ok=False; a named-scheme question returns only that scheme; a nonsense
question returns ok=False.

Run the four verification queries listed in the plan and show me the real output.
If the best hit is wrong, diagnose the cause — do not simply lower SIM_FLOOR.
```

**Fix loop.** Best hit wrong → *"The top hit for this query is not the answer. Compare the query terms against the retrieved chunk text and tell me whether the cause is the context prefix, a term mismatch, or a competing chunk — then fix that specific cause."*

---

## Phase 5 — Generation + citation (M5)

**Objective.** First real answers: ≤3 sentences, exactly one cited URL, mandatory footer.

**Touches.** `app/prompts.py`, `app/generate.py`, `tests/test_generate.py`.

**Steps.**
1. `app/prompts.py` — the system prompt from architecture.md §5.4 **verbatim**, plus `build_context_block(chunks)` and `resolve_as_of(chunks)`.
2. `app/generate.py`:
   - `call_llm(question, context)` — Groq, `temperature=0`, `max_tokens≈220`, 1 retry with backoff on rate-limit/5xx/timeout. Read the key from config; never log it.
   - `enforce(draft, context)` — the post-generation guard from architecture.md §5.5, as five small pure functions: `enforce_url`, `enforce_sentence_cap`, `enforce_footer`, `detect_advice_leak`, `is_degenerate`. Each returns `(text, flags)`.
   - `answer_question(question) -> dict` — the response shape from architecture.md §5.6. `citation` is a **structured field** pulled from chunk metadata, never parsed from prose.
   - `fallback_response(context)` — retrieval-only answer (top chunk's first 2 sentences + its citation) for LLM failure.
3. `tests/test_generate.py` with a mocked LLM returning adversarial strings: two URLs, five sentences, no footer, a recommendation, an empty string. Assert the guard fixes each.

**Verification**

```powershell
python -m app.generate "What is the expense ratio of HDFC Large Cap Fund Direct Growth?"
python -m app.generate "Is there a lock-in for HDFC ELSS Tax Saver Fund?"
python -m app.generate "What is the minimum SIP for HDFC Equity Flexi Cap Fund?"
python -m app.generate "What is the exit load on HDFC Small Cap Fund?"
python -m app.generate "What is the riskometer and benchmark of HDFC Balanced Advantage Fund?"
python -m app.generate "How do I download my capital gains statement?"
python -m pytest tests/test_generate.py -v
```

**Expected.** Each answer: ≤3 sentences, one clickable URL pointing at the real official page, and the exact line `Last updated from sources: <date>`. **Now fact-check 3 answers against the actual source page** — a citation that resolves is not the same as a correct answer.

**Definition of done**
- [ ] All 6 target questions answered, factually correct against the source
- [ ] Every answer ≤3 sentences, 1 URL, footer present
- [ ] `citation` is structured, not parsed from prose
- [ ] The 5 mocked adversarial outputs are all corrected by the guard
- [ ] LLM failure returns the fallback, not an exception
- [ ] API key never appears in any log or traceback

**Cursor prompt**

```
Phase 5: generation and citation. Implement architecture §5.4 and §5.5.

Create app/prompts.py with the system prompt copied verbatim from §5.4, plus
build_context_block(chunks) and resolve_as_of(chunks) (most recent non-empty as_of_date
among retrieved chunks, else the corpus fetched_at date).

Create app/generate.py with:
  - call_llm(question, context): Groq via the groq client, temperature=0, max_tokens~220,
    one retry with backoff on rate limit / 5xx / timeout. Key from config, never logged.
  - enforce(draft, context): the five post-generation checks from §5.5 as separate pure
    functions returning (text, flags) — enforce_url (keep only URLs present in the
    retrieved chunks' source_url set; if none survive substitute the top chunk's URL; if
    >1 survive keep the first), enforce_sentence_cap (truncate after sentence 3),
    enforce_footer (append "Last updated from sources: {as_of}" if missing),
    detect_advice_leak (recommendation verbs -> retry once stricter, then refusal),
    is_degenerate (<15 chars or truncated).
  - answer_question(question) -> dict with exactly the shape in architecture §5.6.
    citation must be a structured field read from chunk metadata, never parsed from prose.
  - fallback_response(context): top chunk's first 2 sentences plus its citation, used when
    the LLM call fails.

Add tests/test_generate.py with a MOCKED LLM covering: two URLs, five sentences, missing
footer, recommendation verbs, empty string. Assert each is corrected.

Then run the six target questions from the plan and show me the output. Do not call
the LLM from a test.
```

**Fix loop.** Answers wrong → *"This answer contradicts the source page. Retrieve the source chunk, quote it, and tell me whether the error is a retrieval miss or the model ignoring the context."*

---

## Phase 6 — Guardrails (M6)

**Objective.** PII, advice refusals, performance redirects, and INSUFFICIENT — all before the LLM.

**Touches.** `app/guardrails.py`, `tests/test_guardrails.py`, `app/generate.py` (wire the checks in).

**Steps.**
1. `app/guardrails.py` in architecture.md §5.2 order, cheapest first:
   - `redact_pii(q) -> (verdict, message)` — PAN `[A-Z]{5}[0-9]{4}[A-Z]`, Aadhaar `\b\d{4}\s?\d{4}\s?\d{4}\b`, email, phone (10-digit, optional separators/`+91`), OTP keywords, account-no keywords. **Never echo the matched value.** Return only a length and a boolean to the logger.
   - `classify_intent(q) -> Intent` — `FACTUAL | ADVICE | PERFORMANCE | UNKNOWN`, an explicit phrase table, no model call.
   - `refusal_response(intent)` — polite facts-only message + one educational link (SEBI investor education or HDFC "Mutual Funds 101"). Copy from PRD §13.4.
   - `performance_response(scheme)` — states that figures live in the official factsheet, links it, computes nothing.
   - `insufficient_response(missing)` — names what isn't covered + general HDFC link. Deterministic, no LLM.
2. Wire into `answer_question` so a refusal short-circuits **before** retrieval and before the LLM.
3. `tests/test_guardrails.py`: one positive per PII type **plus near-miss negatives** (a 12-digit folio, `1234567890` inside a longer number) that must *not* hard-reject; the 7 refusal questions from PRD §13.2; and the 10 factual questions that must **not** be refused.

**Verification**

```powershell
python -m app.generate "Should I buy HDFC Small Cap Fund?"
python -m app.generate "Which of these five funds has the best returns?"
python -m app.generate "My PAN is ABCDE1234F, can you tell me my tax?"
python -m app.generate "Contact me at 9876543210 to explain."
python -m app.generate "How should I split my portfolio between these schemes?"
python -m app.generate "What is the expense ratio of HDFC Large Cap Fund?"
python -m pytest tests/test_guardrails.py -v
```

**Expected.** Questions 1–5: refusal/redirect with an educational or factsheet link, no numbers, no recommendation, and **nothing written to `queries.jsonl` beyond the redacted fields**. Question 6: a real answer, not a refusal. Then `Select-String -Path data\logs\queries.jsonl -Pattern "ABCDE1234F|9876543210"` returns **nothing** — verify PII never hit disk.

**Definition of done**
- [ ] 7/7 PRD refusal questions refused
- [ ] 10/10 factual questions still answered (no false refusals — the metric that matters)
- [ ] PII rejected for all 6 patterns, and near-misses not rejected
- [ ] PII absent from `queries.jsonl` (grep verified)
- [ ] Refusals never call the LLM
- [ ] Every refusal has an educational link

**Cursor prompt**

```
Phase 6: guardrails. Implement architecture §5.2 in order, cheapest first, as an
explicit rule table — no model calls.

Create app/guardrails.py with redact_pii, classify_intent, refusal_response,
performance_response, insufficient_response. Patterns and copy come from §5.2 and
PRD §13.4. Never echo a matched PII value in any message or log field.

Wire the checks into app.answer_question so PII and ADVICE/PERFORMANCE intents
short-circuit BEFORE retrieval and before any LLM call. Confirm by code inspection
that a refusal path cannot reach the Groq client.

tests/test_guardrails.py must include near-miss negatives: a 12-digit folio string and
a phone-like number embedded in a longer number must NOT be rejected. The 7 refusal
questions from PRD §13.2 must all refuse, and 10 factual questions must NOT be refused.

Run the six verification questions. Then grep data/logs/queries.jsonl for the PAN and
the phone number and show me that neither is present.
```

**Fix loop.** False refusal on a factual question → *"A factual question is being refused. Show which rule matched, and narrow that rule rather than deleting it."*

---

## Phase 7 — Chat UI (M7)

**Objective.** The demo surface: welcome line, disclaimer, 3 example questions, answer + citation, refusal bubble, sources expander.

**Touches.** `app/ui.py`, `app/__main__.py`, `samples/disclaimer.txt`.

**Steps.**
1. `samples/disclaimer.txt` — the exact copy from PRD §13.3, one line, no edits.
2. `app/ui.py` — Streamlit, `chat_message` bubbles:
   - On first load: welcome line + disclaimer + **exactly 3** example questions as buttons (pick from PRD §13.1 — expense ratio, ELSS lock-in, how to download the statement; they must exercise three different code paths: FACTUAL, FACTUAL, and `guide` source type).
   - Clicking a button fills the input.
   - Answer bubble: text, then a real link from the structured `citation` field, then the `Last updated from sources:` line.
   - Refusal / INSUFFICIENT / PII bubbles styled distinctly, all with a link.
   - **Sources expander** (`st.expander`): per hit — section, source type, score, 200-char preview. This is the slide that proves retrieval is real.
   - Show latency and the `EMBED_MODEL` in a footer caption. Honest metrics read as engineering; hidden ones read as luck.
3. Startup: if the collection is missing or `corpus_version` mismatches, show a blocking message: `python -m app.ingest`. If `GROQ_API_KEY` is missing, show setup instructions but still let retrieval run.
4. `app/__main__.py` → `streamlit run app/ui.py`.

**Verification.** `streamlit run app/ui.py`, then click all 3 examples plus one advice question and one PII question. Screenshot each state. Kill the network mid-session to confirm the LLM-failure fallback renders rather than a stack trace.

**Definition of done**
- [ ] Welcome line + disclaimer + exactly 3 example questions on load
- [ ] Answer bubble shows a clickable link and the footer
- [ ] Refusal and INSUFFICIENT bubbles are visually distinct and carry a link
- [ ] Sources expander lists the retrieved chunks with scores
- [ ] Missing/stale index shows the ingest instruction, not an error
- [ ] Missing API key shows setup instructions; retrieval still works
- [ ] `python -m app` starts the UI

**Cursor prompt**

```
Phase 7: chat UI. Read PRD §13.3 for the exact disclaimer copy and architecture §5.6
for the response dict your UI consumes.

Create samples/disclaimer.txt containing the disclaimer string exactly as written in
PRD §13.3, one line.

Create app/ui.py (Streamlit):
  - on load: welcome line, the disclaimer, and EXACTLY 3 example questions as buttons
    (use three of the PRD §13.1 questions; make sure they exercise different paths:
    a normal factual lookup, an ELSS lock-in lookup, and a "how do I download" guide)
  - clicking a button fills the input
  - answer bubble: answer text, then a real link rendered from the structured citation
    field, then the Last updated from sources line
  - refusal, insufficient, and pii_rejected bubbles styled distinctly, each with a link
  - a sources expander listing each retrieved chunk: section, source_type, score, and a
    200-char preview
  - a footer caption with latency and the embedding model name

Startup behaviour: if the Chroma collection is missing or corpus_version mismatches,
render a blocking message telling the user to run `python -m app.ingest`. If
GROQ_API_KEY is missing, render setup instructions but still allow retrieval to run.

Point app/__main__.py at `streamlit run app/ui.py`. Use no new dependencies. Then run
it and describe what you see, including the refusal and PII states.
```

**Fix loop.** Crash on first load → *"Run the app, read the traceback, fix the cause. Show me the working output, not a description of it."*

---

## Phase 8 — Eval, samples, README (M7)

**Objective.** The evidence that it works, and the graded deliverables.

**Touches.** `scripts/eval.py`, `samples/qa.md`, `sources.md` (refresh), `README.md`, `tests/test_e2e.py`.

**Steps.**
1. `scripts/eval.py` over a labelled set of 20 queries (`expected_scheme`, `expected_source_type`, `expected_url`) built from the 6 target questions plus 14 variations across schemes and phrasings. Report: recall@5, scheme accuracy, citation correctness, refusal precision/recall, and per-query rows.
2. `samples/qa.md` — 5–10 real questions with the assistant's **actual** answers and links, plus a one-line grade per item. No hand-edited answers.
3. `README.md` — setup (venv, `pip install`, copy `.env.example`, `python -m app.ingest`, `streamlit run app/ui.py`), scope (HDFC AMC + the 5 schemes), architecture summary, **known limits** (corpus snapshot date, aggregator-vs-official conflicts, MiniLM's general-purpose weakness, no performance data by design, one AMC only), and the disclaimer.
4. `tests/test_e2e.py` — a smoke test over the 6 target questions asserting one URL, footer present, ≤3 sentences. Marked `@pytest.mark.slow`; skips without `GROQ_API_KEY`.

**Verification**

```powershell
python scripts/eval.py
python -m pytest tests/ -v
```

**Expected.** A table with recall@5, and a README where every command has been actually run.

**Definition of done**
- [ ] `eval.py` runs end to end and reports all four metrics
- [ ] `samples/qa.md` answers are copy-pasted from real runs
- [ ] `README.md` has setup, scope, known limits, disclaimer; every command verified
- [ ] e2e test passes with a real key, skips cleanly without one
- [ ] `sources.md` lists every file in `data/raw/`

**Cursor prompt**

```
Phase 8: evaluation and deliverables.

Create scripts/eval.py: a labelled set of 20 queries (expected_scheme,
expected_source_type, expected_url) covering the 6 target questions plus 14
variations across the 5 schemes and several phrasings. Run retrieval and generation
per query and report recall@5, scheme accuracy, citation correctness, and refusal
precision/recall, plus a per-query table.

Create samples/qa.md: 5-10 entries with the assistant's ACTUAL answers and links,
generated by running the app. Do not hand-write or edit answers to look better. Add a
one-line grade per entry. No PII in any entry.

Create README.md: setup steps (venv, pip install -r requirements.txt, copy .env.example,
python -m app.ingest, streamlit run app/ui.py), scope (HDFC AMC + the 5 schemes), a short
architecture summary, known limits (corpus snapshot date and its `as of` dates, official
vs aggregator conflicts, MiniLM being general-purpose rather than financial, no
performance data by design, single AMC), and the disclaimer. Every command you list must
be one you have actually run.

Create tests/test_e2e.py: a slow-marked smoke test over the 6 target questions
asserting exactly one URL, the footer, and <=3 sentences; skip when GROQ_API_KEY is unset.

Run eval.py and pytest, and show me the real output.
```

**Fix loop.** Metric looks bad → *"Report the number honestly and then diagnose the worst 3 queries specifically. Fix retrieval causes, not the metric."*

---

## Phase 9 — Demo dry run (M8)

**Objective.** A ≤3-minute, network-tolerant, non-crashing live demo. No new features.

**Touches.** `README.md` (demo section only), `samples/disclaimer.txt` (verify), `demo.mp4` (only if a hosted link is impossible).

**Rehearsal script** (time-boxed; PRD requires ≤3 min):

| Time | Action | Proof on screen |
|------|--------|------------------|
| 0:00–0:20 | Intro: what it is, the disclaimer | Welcome line + "Facts-only. No investment advice." |
| 0:20–0:40 | Show the pipeline, not the chat | `chunks.txt` + the sources expander on a real query |
| 0:40–1:40 | Three example questions | Three answers, each with a clickable official link |
| 1:40–2:20 | **Refusal** demo: "Should I buy HDFC Small Cap?" + "best returns?" | Polite refusal + education/factsheet link |
| 2:20–2:40 | **Guardrail** demo: a fake PAN | Rejection; then show the PAN is absent from `queries.jsonl` |
| 2:40–3:00 | Known limits + Q&A | README limits slide |

**Pre-flight checklist**

- [ ] `python -m app.ingest` run **before** presenting; model pre-warmed
- [ ] `streamlit run app/ui.py` already up, browser tab open
- [ ] All 3 example buttons verified working on the actual demo machine
- [ ] Clicked links open real official pages (not 404s)
- [ ] Revoked-API-key fallback rehearsed (rename `.env` → `.env.bak`, confirm the retrieval-only path, restore)
- [ ] `samples/qa.md` open in a second tab as backup
- [ ] Backup video recorded

**Definition of done**
- [ ] Full path rehearsed twice, under 3 minutes, both times
- [ ] Every link clicked live resolves
- [ ] All four guardrail states demonstrated without fumbling
- [ ] `README.md` documents how to run the demo
- [ ] Backup video recorded if no hosted link exists

**Cursor prompt**

```
Phase 9: demo readiness. No new features.

Verify, by running each command, that: ingest completes and is idempotent; the UI starts;
all 3 example questions answer with a working link; the four guardrail states (advice,
performance, PII, off-corpus) all render correctly; and a missing API key degrades to
retrieval-only without crashing.

Add a "Running the demo" section to README.md with the 3-minute rehearsal script from
the plan, the pre-flight checklist, and what to do if the network fails mid-demo.

Do not change application behaviour. Fix only things that would break during a live demo.
Report the exact sequence of commands to run before presenting.
```

---

## Appendix A — Definition of done (whole project)

Mirrors PRD §8. All boxes must be ticked before submitting.

- [ ] `python -m app.ingest` builds a persisted collection + `chunks.txt`; app restart does not re-embed
- [ ] All 5 schemes in the corpus, each with a working official URL
- [ ] UI shows welcome + disclaimer + 3 examples
- [ ] 6 factual target questions: correct, ≤3 sentences, exactly one working citation
- [ ] "Should I buy…?" → polite refusal + educational link
- [ ] "Best returns?" → factsheet link, no comparison
- [ ] PAN/phone/email rejected and absent from disk
- [ ] `.env` git-ignored, absent from the repo
- [ ] `samples/qa.md`, `sources.md`, `README.md`, `docs/chunking_strategy.md` all present and accurate
- [ ] `eval.py` runs and reports all four metrics

## Appendix B — Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| MiniLM download fails on first run | No cached model, offline | Pre-download once on a connected machine: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"` |
| Collection count ≠ `chunks.txt` count | Partial write, or two corpus versions | Re-run with `--force`; assert `collection.metadata['hnsw:space'] == 'cosine'` |
| Every similarity sits in 0.6–0.8 | L2 distance instead of cosine | Startup assertion from architecture §13.5 |
| Best hit is the wrong scheme | Scheme filter not firing on the alias | Add the alias to `SCHEME_ALIASES` in `config.py` |
| Answers ignore the context | LLM too weak for the format rules | Try the next model up on Groq before touching the prompt |
| Two links in one answer | Post-guard not running | Assert `enforce()` is called on **every** return path in `answer_question` |
| PII in the log file | Logging before the PII check | Move the check above the logger; re-grep |
| Garbled table text in a chunk | PDF column-wise extraction | Re-extract or hand-transcribe the table into `data/raw/` as text |

## Appendix C — Anti-goals

Tell Cursor these explicitly if it starts gold-plating:

- No agents, tool use, or multi-step reasoning loops — this is a single-shot RAG.
- No reranking model, no hybrid BM25, no query rewriting (the `Should` items stay undone).
- No streaming, no chat persistence, no auth, no Docker, no cloud deploy.
- No multi-AMC support, no PDF upload, no web search at query time.
- No unit-test framework beyond `pytest`, no CI config.

> If one of these turns out to be load-bearing (a reranker, say, because recall@5 is genuinely poor), add it as a documented decision in `docs/chunking_strategy.md` with the measured before/after — don't add it silently.
