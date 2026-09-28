# Source List — Mutual Fund Facts-Only RAG Chatbot

**Scope:** HDFC Asset Management Company (HDFC AMC) — 5 schemes
**Corpus version:** `2026-09-28.1`
**Retrieved:** 2026-09-28
**Policy:** Public official sources only — HDFC AMC, AMFI, SEBI. No third-party blogs, no aggregators, no screenshots. (PRD §4.4)

Reproduce with:

```powershell
python scripts\fetch_sources.py                  # dry run, prints the plan
python scripts\fetch_sources.py --fetch --extract # rebuilds data/raw/
```

---

## 1. Distinct source URLs

Six distinct official URLs back all thirteen text files. Every answer in the app cites one of these.

| # | Source document | Publisher | URL | As of |
|---|-----------------|-----------|-----|-------|
| 1 | HDFC Mutual Fund Factsheet — June 2026 (master, 140 pp.) | HDFC AMC | https://files.hdfcfund.com/s3fs-public/2026-07/HDFC%20MF%20Factsheet%20-%20June%202026.pdf | 2026-06-30 |
| 2 | KIM — HDFC Large Cap Fund, dated 21 Nov 2025 | HDFC AMC | https://files.hdfcfund.com/s3fs-public/KIM/2025-11/KIM%20-%20HDFC%20Large%20Cap%20Fund%20dated%20November%2021%2C%202025.pdf | 2025-11-21 |
| 3 | KIM — HDFC Flexi Cap Fund, dated 21 Nov 2025 | HDFC AMC | https://files.hdfcfund.com/s3fs-public/KIM/2025-11/KIM%20-%20HDFC%20Flexi%20Cap%20Fund%20dated%20November%2021%2C%202025.pdf | 2025-11-21 |
| 4 | KIM — HDFC ELSS Tax Saver, dated 21 Nov 2025 | HDFC AMC | https://files.hdfcfund.com/s3fs-public/KIM/2025-11/KIM%20-%20HDFC%20ELSS%20Tax%20Saver%20dated%20November%2021%2C%202025.pdf | 2025-11-21 |
| 5 | KIM — HDFC Small Cap Fund, dated 21 Nov 2025 | HDFC AMC | https://files.hdfcfund.com/s3fs-public/KIM/2025-11/KIM%20-%20HDFC%20Small%20Cap%20Fund%20dated%20November%2021%2C%202025.pdf | 2025-11-21 |
| 6 | KIM — HDFC Balanced Advantage Fund, dated 21 Nov 2025 | HDFC AMC | https://files.hdfcfund.com/s3fs-public/KIM/2025-11/KIM%20-%20HDFC%20Balanced%20Advantage%20Fund%20dated%20November%2021%2C%202025.pdf | 2025-11-21 |
| 7 | SAI with updated addendum, 25 Sep 2025 (extract: investor statements) | HDFC AMC | https://files.hdfcfund.com/s3fs-public/2025-09/SAI%20with%20updated%20Addendum%20dated%20September%2025%2C%202025.pdf | 2025-09-25 |
| 8 | HDFC Mutual Fund — Investor Charter | HDFC AMC | https://files.hdfcfund.com/s3fs-public/2024-05/Investor%20Charter%20-%20MF.pdf | 2024-05-01 |

All URLs are on `files.hdfcfund.com`, HDFC AMC's document CDN. The scheme pages on `www.hdfcfund.com` are listed in the fetcher's allow-list for completeness but are **not** used — see §4.

---

## 2. Corpus files (13 text files, ~443 KB)

| # | Scheme | File | Type | As of | Covers |
|---|--------|------|------|-------|--------|
| 1 | Large Cap | `data/raw/hdfc-large-cap/factsheet-2026-06.txt` | factsheet | 2026-06-30 | expense ratio, benchmark, exit load, riskometer, fund manager, inception, AUM |
| 2 | Flexi Cap | `data/raw/hdfc-flexi-cap/factsheet-2026-06.txt` | factsheet | 2026-06-30 | expense ratio, benchmark, exit load, riskometer, fund manager, inception, AUM |
| 3 | ELSS | `data/raw/hdfc-elss/factsheet-2026-06.txt` | factsheet | 2026-06-30 | expense ratio, benchmark, exit load, **lock-in 3 yrs**, riskometer, fund manager |
| 4 | Small Cap | `data/raw/hdfc-small-cap/factsheet-2026-06.txt` | factsheet | 2026-06-30 | expense ratio, benchmark, exit load, riskometer, fund manager, inception, AUM |
| 5 | Balanced Advantage | `data/raw/hdfc-balanced-advantage/factsheet-2026-06.txt` | factsheet | 2026-06-30 | expense ratio, benchmark, exit load, riskometer, fund manager, inception, AUM |
| 6 | Large Cap | `data/raw/hdfc-large-cap/kim-2025-11.txt` | kim | 2025-11-21 | minimum application amount ₹100, full exit-load rules + exemptions, benchmark, riskometer |
| 7 | Flexi Cap | `data/raw/hdfc-flexi-cap/kim-2025-11.txt` | kim | 2025-11-21 | minimum application amount ₹100, full exit-load rules + exemptions, benchmark, riskometer |
| 8 | ELSS | `data/raw/hdfc-elss/kim-2025-11.txt` | kim | 2025-11-21 | minimum application amount ₹500, **statutory lock-in 3 years**, exit load Nil, benchmark, riskometer |
| 9 | Small Cap | `data/raw/hdfc-small-cap/kim-2025-11.txt` | kim | 2025-11-21 | minimum application amount ₹100, full exit-load rules + exemptions, benchmark, riskometer |
| 10 | Balanced Advantage | `data/raw/hdfc-balanced-advantage/kim-2025-11.txt` | kim | 2025-11-21 | minimum application amount ₹100, exit-load-free redemption limit + slabs, benchmark, riskometer |
| 11 | Cross-scheme | `data/raw/general/riskometer-benchmark-annexure-2026-06.txt` | riskometer | 2026-06-30 | SEBI riskometer level + Tier-1/Tier-2 benchmark for every scheme (factsheet pp. 123–138) |
| 12 | Cross-scheme | `data/raw/general/sai-investor-statements-2025-09.txt` | guide | 2025-09-25 | how investors receive account / consolidated account statements; MFCentral one-stop portal; SCAS vs MF-CAS |
| 13 | Cross-scheme | `data/raw/general/investor-charter-2024-05.txt` | guide | 2024-05-01 | investor service commitments, CAS timelines, redemption pay-out timelines |

Each file begins with a `SOURCE DOCUMENT` header block (`source_url`, `source_type`, `as_of`, `pages`, `retrieved`) so provenance travels with every chunk.

---

## 3. Coverage against PRD §4.2 hard floor

| Scheme | factsheet | fees / exit load | min investment | lock-in | riskometer | benchmark | min SIP |
|--------|-----------|-----------------|----------------|---------|------------|-----------|---------|
| Large Cap | yes | yes | yes (₹100) | n/a | yes | yes | **gap** |
| Flexi Cap | yes | yes | yes (₹100) | n/a | yes | yes | **gap** |
| ELSS | yes | yes | yes (₹500) | yes (3 yrs) | yes | yes | **gap** |
| Small Cap | yes | yes | yes (₹100) | n/a | yes | yes | **gap** |
| Balanced Advantage | yes | yes | yes (₹100) | n/a | yes | yes | **gap** |
| Statement guide | — | — | — | — | — | — | yes (SAI + Charter) |

---

## 4. Known gaps and rejected sources

### 4.1 Minimum SIP amount — not in corpus (open)

The brief lists "minimum SIP" as a target question. **No official fetchable source states it.**

- The master factsheet and all five KIMs carry *Minimum Application Amount* (purchase / additional purchase / redemption) but **not** the SIP instalment minimum. Verified by regex across all five KIMs: zero hits.
- SIP minimums live only on `www.hdfcfund.com/explore/mutual-funds/<scheme>` — the pages named in the brief. That host returns **HTTP 403** to every automated request (verified with browser User-Agent on GET and HEAD, both `Invoke-WebRequest` and direct fetch).

Third-party aggregators (Anand Rathi, Torus Digital, Fisdom, ET, Value Research) all state ₹100 (₹500 for ELSS) and would fill the gap, but PRD §4.4 forbids them as sources. **They are excluded.**

**Consequence for the demo:** the bot must answer "What is the minimum SIP?" with a *bounded, honest* reply — cite the KIM's minimum application amount, state that the SIP instalment minimum is not in the corpus, and link the scheme page for the investor to confirm. This is exactly the `INSUFFICIENT` path from architecture.md §5.3 and is a good thing to demonstrate rather than hide.

**To close the gap (needs a human):** open the five scheme pages in a real browser and transcribe the SIP minimum into `data/raw/<scheme>/scheme-page-<date>.txt` with the URL and access date in the header. That is a 10-minute manual task and is consistent with the PRD's "manual, one-time snapshot" approach.

### 4.2 Lock-in for non-ELSS schemes — stated by absence

Large Cap, Flexi Cap, Small Cap and Balanced Advantage have no lock-in, and the corpus never says "no lock-in" explicitly. A "Is there a lock-in on HDFC Small Cap?" question may therefore retrieve weakly. The ELSS lock-in *is* explicitly present. Mitigation belongs in Phase 2/3: add an explicit `no lock-in` sentence per scheme during normalise, so the bot can answer from the corpus rather than by inference.

### 4.3 Capital-gains statement specifically

The SAI and Investor Charter cover *account* and *consolidated account* statements authoritatively, plus MFCentral as the one-stop portal — but neither says "capital gains statement" in those terms. `camsonline.com` (HDFC's RTA) has a dedicated Capital Gain & Loss Statement page, but it is JavaScript-rendered: the fetched HTML is 11.8 KB and yields **228 characters** of text. Not usable as a corpus source without a headless browser.

The SAI's tax section does discuss capital gains (sections 111A / 112A / 115AD) and is present in the full SAI, but the extract kept for the corpus is scoped to statements, so those tax tables are not in the ingested text.

**Consequence:** "How do I download my capital-gains statement?" gets a facts-based answer about where statements come from (AMC/RTA, CAS timelines, MFCentral) rather than exact button-by-button steps. Honest and cited, not a fabrication.

### 4.4 Sources deliberately rejected

| Source | Why rejected |
|--------|--------------|
| groww.in (the 5 URLs in the brief) | Third-party aggregator, and PRD §4.4 makes official sources authoritative. Also blocked by the fetcher's allow-list. |
| morningstar.in, valueresearchonline.com, economictimes, scripbox, fisdom, anandrathi, advisor khoj, jezzmoney, mintbyte, torusdigital, sharpely | Third-party data vendors and news — banned as fact sources by PRD §4.4. |
| webnotes.in, paytm.com, economictimes, tax2win, basunivesh, finnovate, cleartax guide | Third-party blogs on downloading capital-gains statements — banned. |
| scribd.com copy of the HDFC Large Cap Scheme Summary | Unofficial rehosting of an HDFC document. The authentic SSD lives on the blocked `hdfcfund.com`. |

### 4.5 Feasibility note for the demo

Every fact in the corpus came from `files.hdfcfund.com`, which serves PDFs without bot protection. `www.hdfcfund.com` (403) and `camsonline.com` (JS-rendered) are the two blocks encountered. This is the concrete form of the "fragile scraping" risk in architecture.md §11 — recorded here so the demo can state it rather than discover it.

---

## 5. Verified spot-checks

Extracted directly from `data/raw/`, cross-read against the source PDFs:

| Fact | Value | File |
|------|-------|------|
| Large Cap expense ratio | Regular 1.56% / Direct 1.03% | `hdfc-large-cap/factsheet-2026-06.txt` |
| Large Cap benchmark | NIFTY 100 Total Returns Index (TRI) | `hdfc-large-cap/factsheet-2026-06.txt` |
| Flexi Cap expense ratio | Regular 1.35% / Direct 0.75% | `hdfc-flexi-cap/factsheet-2026-06.txt` |
| Flexi Cap benchmark | NIFTY 500 Index (TRI) | `hdfc-flexi-cap/factsheet-2026-06.txt` |
| ELSS expense ratio | Regular 1.75% / Direct 1.18% | `hdfc-elss/factsheet-2026-06.txt` |
| ELSS benchmark | NIFTY 500 Index (TRI) | `hdfc-elss/factsheet-2026-06.txt` |
| ELSS lock-in | 3 years from the date of allotment of the respective Units | `hdfc-elss/factsheet-2026-06.txt`, `hdfc-elss/kim-2025-11.txt` |
| ELSS exit load | Nil | `hdfc-elss/factsheet-2026-06.txt` |
| ELSS minimum application amount | ₹500 and in multiples of ₹500 thereafter | `hdfc-elss/kim-2025-11.txt` |
| Small Cap expense ratio | Regular 1.55% / Direct 0.76% | `hdfc-small-cap/factsheet-2026-06.txt` |
| Small Cap benchmark | BSE 250 Smallcap Index (TRI) | `hdfc-small-cap/factsheet-2026-06.txt` |
| Balanced Advantage expense ratio | Regular 1.29% / Direct 0.77% | `hdfc-balanced-advantage/factsheet-2026-06.txt` |
| Balanced Advantage benchmark | NIFTY 50 Hybrid Composite Debt 50:50 Index (Total Returns Index) | `hdfc-balanced-advantage/factsheet-2026-06.txt` |
| Balanced Advantage exit load | Up to 15% of units redeemable without exit load from allotment; excess subject to 1.00% within 1 year, FIFO | `hdfc-balanced-advantage/factsheet-2026-06.txt` |
| Minimum application amount, other 4 schemes | ₹100 and any amount thereafter | `<scheme>/kim-2025-11.txt` |
| Exit load, Large / Flexi / Small Cap | 1.00% if redeemed or switched out within 1 year of allotment; nil after 1 year | `<scheme>/factsheet-2026-06.txt` |

Expense ratios are dated 2026-06-30 (factsheet) while minimums and exit-load rules are dated 2025-11-21 (KIM) — a deliberate artefact of using the most recent document for each fact. The `as_of` date travels with each chunk, so the answer footer always states which document the number came from.
