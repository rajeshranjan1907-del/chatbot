"""Corpus inspection for the mutual fund FAQ corpus.

Phase 2, step 1. Measures the real shape of data/raw/ so the chunking strategy in
docs/chunking_strategy.md is chosen from evidence rather than defaults.

Usage:
    python scripts/inspect_corpus.py
    python scripts/inspect_corpus.py --out reports/corpus_inspection.txt
    python scripts/inspect_corpus.py --longest 25
"""

from __future__ import annotations

import argparse
import re
import statistics
from collections import Counter
from pathlib import Path

RAW = Path("data/raw")

# Lines that repeat on nearly every factsheet page and carry no per-scheme fact.
# NOTE: a bare-number rule is deliberately absent. Page numbers in this corpus are
# indistinguishable from legitimate single-cell table values ("4", "13"), so
# stripping `^\d+$` would silently delete real data. See unwrap() instead, which
# only drops a numeric line when it is glued to a page-header line.
BOILERPLATE_PATTERNS = (
    r"^For Product label and Riskometers, refer page no",
    r"^MUTUAL FUND INVESTMENTS ARE SUBJECT TO MARKET RISKS",
    r"^Read all scheme related documents carefully",
    r"^HDFC MF SAI - dated",
    r"^Page \d+ of \d+$",
    r"^\d+\s*\|\s*(January|February|March|April|May|June|July|August|September|October|November|December)",
    r"^Contd on next page$",
    r"^\.{4,}",
    r"^\.\.\.\.Contd",
    r"^\.\.\.\.\.\.",
)

# Structural headings in HDFC factsheets and KIMs. These are the natural chunk
# boundaries and the natural source of the `section` metadata field.
#
# Two families, because the documents disagree on style:
#   TIGHT  - whole line must be exactly this, optionally with a trailing marker
#             glyph ($, #, euro, etc.) that pypdf picks up from the layout.
#   PROSE  - a heading-with-value line, e.g. "Exit Load: Nil" or
#             "INVESTMENT OBJECTIVE: To generate ...". Matched as a PREFIX, but
#             only when the line is short enough to be a heading rather than a
#             sentence that merely starts with the same word.
HEADING_TIGHT = (
    r"CATEGORY OF SCHEME",
    r"DATE OF ALLOTMENT/INCEPTION DATE",
    r"QUANTITATIVE DATA",
    r"EXPENSE RATIO",
    r"BENCHMARK INDEX",
    r"ADDL\. BENCHMARK INDEX",
    r"LOCK-?IN PERIOD",
    r"PORTFOLIO",
    r"NAV",
    r"NAV PER",
    r"NAV PER\s*UNIT.*",
    r"FUND MANAGER",
    r"ASSETS UNDER MANAGEMENT",
    r"SERVICES PROVIDED FOR INVESTORS",
    r"TRANSACTIONS THROUGH MF UTILITY",
    r"ACCOUNT STATEMENTS?/?CONSOLIDATED ACCOUNT STATEMENT",
    r"MF\s?Central",
    r"Minimum Application Amount",
    r"Minimum Investment",
    r"Load Structure",
    r"Tax Status",
    r"Who should invest",
    r"Who should not invest",
    r"DISCLOSURE AND REPORTS",
    r"RIGHTS OF UNITHOLDERS",
)
HEADING_PREFIX = re.compile(
    r"^(INVESTMENT OBJECTIVE|EXIT LOAD|LOCK-?IN PERIOD|EXPENSE RATIO"
    r"|BENCHMARK INDEX|ADDL\. BENCHMARK INDEX|MINIMUM APPLICATION AMOUNT"
    r"|ACCOUNT STATEMENTS?|CONSOLIDATED ACCOUNT STATEMENT|SERVICES PROVIDED FOR INVESTORS)",
    re.IGNORECASE,
)
# A PROSE-heading line must be short and must not read like a wrapped sentence.
PROSE_MAX_LEN = 90
HEADING_TRAILING_GLYPH = r"[\s$#€¥№:.\-–—]*"

# A line that carries a number with a unit is a "protected unit": splitting it
# from its label produces a confidently wrong answer.
NUMERIC_UNIT = re.compile(
    r"(?:\d[\d,]*\.?\d*\s*(?:%|per cent|bps|years?|year|months?|days?))"
    r"|(?:Rs\.?\s*\.?\s*[\d,]+)"
    r"|(?:₹\s*[\d,]+)"
    r"|(?:[\d,]+\s*Cr)",
    re.IGNORECASE,
)
LABEL_VALUE = re.compile(r"^[A-Z][A-Za-z0-9 /&()\-.']{2,60}:\s*\S")
SOURCE_HEADER = re.compile(r"^(title|scheme|source_type|publisher|source_url|doc_date|as_of|pages|retrieved)\s*:")


def strip_source_header(text: str) -> tuple[str, dict[str, str]]:
    """Separate the leading SOURCE DOCUMENT metadata block from the body."""
    meta: dict[str, str] = {}
    lines = text.splitlines()
    start = 0
    if lines and lines[0].strip() == "SOURCE DOCUMENT":
        for i in range(1, len(lines)):
            m = SOURCE_HEADER.match(lines[i].strip())
            if m:
                meta[m.group(1)] = lines[i].split(":", 1)[1].strip()
                start = i + 1
            elif lines[i].startswith("=" * 10):
                start = i + 1
                break
    return "\n".join(lines[start:]), meta


def boilerplate(line: str) -> bool:
    return any(re.match(p, line.strip(), re.IGNORECASE) for p in BOILERPLATE_PATTERNS)


def is_heading(line: str) -> bool:
    """True only for structural headings, not for sentences that start like one.

    Two deliberate restrictions, both learned from false positives in the first
    pass of this report:

    1. `NAV` and `PORTFOLIO` are ordinary English words. A bare `^NAV\\b` or
       `^PORTFOLIO` prefix rule matches "NAV of the Scheme is expected to
       increase..." and "portfolio of the schemes, performance of the
       schemes...". Those words are therefore TIGHT-only (the whole line must be
       the word), never PROSE.
    2. A PROSE heading must be followed by a delimiter (":" or "-"). That is what
       separates "Exit Load: Nil" and "Account Statements:" from a sentence that
       merely opens with the same words.
    """
    s = line.strip()
    if not s:
        return False
    for pat in HEADING_TIGHT:
        if re.fullmatch(pat + HEADING_TRAILING_GLYPH, s, re.IGNORECASE):
            return True
    m = HEADING_PREFIX.match(s)
    if m:
        tail = s[m.end():].lstrip()
        if tail[:1] in {":", "-", "–"} and len(s) <= PROSE_MAX_LEN:
            return True
    return False


PAGE_HEADER = re.compile(
    r"^(\d+)\s*\|\s*(January|February|March|April|May|June|July|August|September|October|November|December)\b",
    re.IGNORECASE,
)


def unwrap(text: str) -> tuple[list[str], int]:
    """Re-join hard-wrapped PDF lines into logical paragraphs.

    pypdf emits one physical line per typeset row, so a sentence arrives as
    several lines. This mirrors what the Phase 3 chunker must do before any
    size-based split, and measures the true atomic unit.

    Returns (paragraphs, pages_dropped).
    """
    lines = [l.rstrip() for l in text.splitlines()]
    paras: list[str] = []
    buf: list[str] = []
    dropped = 0
    for line in lines:
        s = line.strip()
        if not s:
            if buf:
                paras.append(" ".join(buf))
                buf = []
            continue
        pm = PAGE_HEADER.match(s)
        if pm:
            # "<page> | <Month> <year>" is a running page header. If a bare
            # number follows it, that number is the page number, not data.
            if buf and PAGE_HEADER.match(" ".join(buf)):
                pass
            paras.append(s)
            buf = []
            continue
        if re.fullmatch(r"\d{1,3}", s) and paras and PAGE_HEADER.match(paras[-1]):
            dropped += 1
            continue
        if boilerplate(s):
            if buf:
                paras.append(" ".join(buf))
                buf = []
            continue
        if buf and is_heading(buf[-1]) and is_heading(s):
            paras.append(" ".join(buf))
            buf = [s]
            continue
        if is_heading(s) and buf:
            paras.append(" ".join(buf))
            buf = [s]
            continue
        buf.append(s)
    if buf:
        paras.append(" ".join(buf))
    return paras, dropped


def percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return int(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo))


def inspect_file(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    body, meta = strip_source_header(raw)
    lines = [l.strip() for l in body.splitlines()]
    lines = [l for l in lines if l]

    lengths = [len(l) for l in lines]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    para_lens = [len(p) for p in paragraphs]

    headings = [l for l in lines if is_heading(l)]
    boiler = [l for l in lines if boilerplate(l)]
    numeric = [l for l in lines if NUMERIC_UNIT.search(l)]
    label_value = [l for l in lines if LABEL_VALUE.match(l)]

    paras, pages_dropped = unwrap(body)
    unwrap_lens = [len(p) for p in paras]

    return {
        "path": path,
        "meta": meta,
        "chars": len(body),
        "line_count": len(lines),
        "line_lens": lengths,
        "max_line": max(lengths) if lengths else 0,
        "para_count": len(paragraphs),
        "para_lens": para_lens,
        "avg_para": int(statistics.mean(para_lens)) if para_lens else 0,
        "p50_para": percentile(para_lens, 0.50),
        "p90_para": percentile(para_lens, 0.90),
        "p99_para": percentile(para_lens, 0.99),
        "max_para": max(para_lens) if para_lens else 0,
        "unwrap_paras": paras,
        "unwrap_lens": unwrap_lens,
        "unwrap_p50": percentile(unwrap_lens, 0.50),
        "unwrap_p90": percentile(unwrap_lens, 0.90),
        "unwrap_p99": percentile(unwrap_lens, 0.99),
        "unwrap_max": max(unwrap_lens) if unwrap_lens else 0,
        "pages_dropped": pages_dropped,
        "headings": headings,
        "heading_count": len(headings),
        "boiler_count": len(boiler),
        "numeric_count": len(numeric),
        "label_value_count": len(label_value),
        "longest": sorted(lines, key=len, reverse=True),
        "body": body,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="write the report to a file instead of stdout")
    ap.add_argument("--longest", type=int, default=15, help="how many longest lines to show per file")
    args = ap.parse_args()

    files = sorted(RAW.glob("*/*.txt"))
    if not files:
        print("No corpus files found. Run: python scripts/fetch_sources.py --fetch --extract")
        return 1

    out: list[str] = []
    w = out.append

    w("=" * 100)
    w("CORPUS INSPECTION - mutual fund FAQ corpus")
    w("=" * 100)
    w(f"files scanned : {len(files)}")
    w(f"longest lines shown per file: {args.longest}")
    w("")

    results = [inspect_file(p) for p in files]

    # ---------- per-file table ----------
    w("1. PER-FILE SHAPE")
    w("-" * 100)
    w("chars/lines are raw pypdf output. uP* are logical-paragraph stats AFTER de-wrapping.")
    head = (
        f"{'file':<44}{'chars':>7}{'lines':>6}{'uPar':>6}"
        f"{'uP50':>6}{'uP90':>6}{'uPmax':>7}{'maxL':>6}{'head':>5}{'num':>5}"
    )
    w(head)
    w("-" * 100)
    for r in results:
        name = str(r["path"]).replace("data\\raw\\", "").replace("data/raw/", "")
        w(
            f"{name:<44}{r['chars']:>7}{r['line_count']:>6}{len(r['unwrap_paras']):>6}"
            f"{r['unwrap_p50']:>6}{r['unwrap_p90']:>6}{r['unwrap_max']:>7}"
            f"{r['max_line']:>6}{r['heading_count']:>5}{r['numeric_count']:>5}"
        )
    w("")

    # ---------- aggregates ----------
    all_lines: list[int] = []
    all_paras: list[int] = []
    total_chars = 0
    total_lines = 0
    total_boiler = 0
    total_numeric = 0
    total_lv = 0
    total_head = 0
    for r in results:
        all_lines += r["line_lens"]
        all_paras += r["para_lens"]
        total_chars += r["chars"]
        total_lines += r["line_count"]
        total_boiler += r["boiler_count"]
        total_numeric += r["numeric_count"]
        total_lv += r["label_value_count"]
        total_head += r["heading_count"]

    w("2. AGGREGATE LINE-LENGTH DISTRIBUTION  (the atomic unit a chunker must not break)")
    w("-" * 100)
    w(f"total characters        : {total_chars:,}")
    w(f"total lines             : {total_lines:,}")
    w(f"total paragraphs        : {len(all_paras):,}")
    w(f"paragraph p50 / p90 / p99: {percentile(all_paras,.5)} / {percentile(all_paras,.9)} / {percentile(all_paras,.99)}")
    w(f"paragraph max           : {max(all_paras) if all_paras else 0:,}")
    w(f"line    p50 / p90 / p99 / max : "
      f"{percentile(all_lines,.5)} / {percentile(all_lines,.9)} / {percentile(all_lines,.99)} / {max(all_lines) if all_lines else 0}")
    w("")
    w("  NOTE: `paragraph` here means a blank-line-separated block. The corpus has")
    w(f"  only {len(all_paras)} of them across {total_chars:,} chars because pypdf emits")
    w("  one line per typeset row with no blank lines. A `\\n\\n` splitter would yield")
    w(f"  {len(all_paras)} blobs averaging {int(statistics.mean(all_paras)) if all_paras else 0:,} chars each -- unusable.")
    w("  Section 2b measures what de-wrapping actually produces.")
    w("")

    all_unwrap: list[int] = []
    for r in results:
        all_unwrap += r["unwrap_lens"]
    dropped_pages = sum(r["pages_dropped"] for r in results)

    w("2b. AFTER DE-WRAPPING  (the real atomic units the chunker will split on)")
    w("-" * 100)
    w(f"logical paragraphs      : {len(all_unwrap):,}  (from {total_lines:,} physical lines)")
    w(f"bare page numbers dropped: {dropped_pages}  (only when adjacent to a '<page> | <Month>' header)")
    w(f"p50 / p90 / p99 / max   : {percentile(all_unwrap,.5)} / {percentile(all_unwrap,.9)} / {percentile(all_unwrap,.99)} / {max(all_unwrap) if all_unwrap else 0}")
    w(f"mean                    : {int(statistics.mean(all_unwrap)) if all_unwrap else 0}")
    ubuckets = [(0, 200), (200, 400), (400, 600), (600, 900), (900, 1300), (1300, 10**9)]
    w("")
    w("logical-paragraph length histogram:")
    for lo, hi in ubuckets:
        n = sum(1 for v in all_unwrap if lo <= v < hi)
        bar = "#" * max(0, round(n / max(1, len(all_unwrap)) * 60))
        label = f"{lo}-{hi if hi < 10**9 else '+'}"
        w(f"  {label:>10}  {n:>6}  {n/max(1,len(all_unwrap))*100:>5.1f}%  {bar}")
    w("")
    over = lambda t: sum(1 for v in all_unwrap if v > t)
    w(f"  logical paragraphs > 400 chars : {over(400):>6}  ({over(400)/max(1,len(all_unwrap))*100:.1f}%)")
    w(f"  logical paragraphs > 900 chars : {over(900):>6}  ({over(900)/max(1,len(all_unwrap))*100:.1f}%)")
    w(f"  logical paragraphs >1300 chars : {over(1300):>6}  ({over(1300)/max(1,len(all_unwrap))*100:.1f}%)")
    w("")

    buckets = [(0, 40), (40, 80), (80, 120), (120, 180), (180, 260), (260, 400), (400, 10**9)]
    w("")
    w("line-length histogram:")
    for lo, hi in buckets:
        n = sum(1 for v in all_lines if lo <= v < hi)
        bar = "#" * max(0, round(n / max(1, len(all_lines)) * 60))
        label = f"{lo}-{hi if hi < 10**9 else '+'}"
        w(f"  {label:>10}  {n:>6}  {n/max(1,len(all_lines))*100:>5.1f}%  {bar}")
    w("")

    w("3. STRUCTURAL SIGNALS  (drives chunk boundaries and the `section` metadata)")
    w("-" * 100)
    w(f"recognised headings     : {total_head:>7}  ({total_head/max(1,total_lines)*100:.1f}% of lines)")
    w(f"numeric-with-unit lines : {total_numeric:>7}  ({total_numeric/max(1,total_lines)*100:.1f}% of lines)  <- PROTECTED UNITS")
    w(f"'Label: value' lines     : {total_lv:>7}  ({total_lv/max(1,total_lines)*100:.1f}% of lines)")
    w(f"boilerplate lines       : {total_boiler:>7}  ({total_boiler/max(1,total_lines)*100:.1f}% of lines)  <- must be stripped")
    w("")

    w("4. MOST COMMON HEADINGS ACROSS THE CORPUS")
    w("-" * 100)
    hc: Counter[str] = Counter()
    for r in results:
        hc.update(r["headings"])
    for text, n in hc.most_common(30):
        w(f"  {n:>4}x  {text[:88]}")
    w("")

    w("5. MOST COMMON BOILERPLATE (strip candidates)")
    w("-" * 100)
    bc: Counter[str] = Counter()
    for r in results:
        for l in r["longest"]:
            pass
    for p in files:
        body, _ = strip_source_header(p.read_text(encoding="utf-8"))
        for l in {x.strip() for x in body.splitlines() if x.strip()}:
            if boilerplate(l):
                bc[l[:88]] += 1
    for text, n in bc.most_common(15):
        w(f"  {n:>4}x  {text}")
    w("")

    w("6. THE LONGEST LINES PER FILE  (reveals whether table rows survive as single units)")
    w("-" * 100)
    for r in results:
        name = str(r["path"]).replace("data\\raw\\", "").replace("data/raw/", "")
        w("")
        w(f"### {name}")
        for line in r["longest"][: args.longest]:
            flag = "NUM" if NUMERIC_UNIT.search(line) else "   "
            w(f"  [{len(line):>4} {flag}] {line[:150]}")
    w("")

    w("6b. THE LONGEST LOGICAL PARAGRAPHS  (the actual chunking decisions Phase 3 must make)")
    w("-" * 100)
    w("These are the units that will be split. Each needs either to fit in one chunk")
    w("or to be split on a sentence boundary without orphaning its numbers.")
    w("")
    for r in results:
        name = str(r["path"]).replace("data\\raw\\", "").replace("data/raw/", "")
        ranked = sorted(r["unwrap_paras"], key=len, reverse=True)[:5]
        w(f"### {name}")
        for p in ranked:
            flag = "NUM" if NUMERIC_UNIT.search(p) else "   "
            w(f"  [{len(p):>5} {flag}] {p[:190]}")
        w("")

    w("7. SOURCE PROVENANCE")
    w("-" * 100)
    for r in results:
        m = r["meta"]
        name = str(r["path"]).replace("data\\raw\\", "").replace("data/raw/", "")
        w(f"{name}")
        w(f"    type={m.get('source_type','?'):<12} as_of={m.get('as_of','?'):<12} url={m.get('source_url','?')[:90]}")
    w("")

    w("=" * 100)
    w("Derived quantities used by docs/chunking_strategy.md:")
    w(f"  raw physical lines               : {total_lines:,}")
    w(f"  blank-line paragraphs (unusable) : {len(all_paras):,}")
    w(f"  logical paragraphs after de-wrap : {len(all_unwrap):,}")
    w(f"  logical para p50 / p90 / p99     : {percentile(all_unwrap,.5)} / {percentile(all_unwrap,.9)} / {percentile(all_unwrap,.99)}")
    w(f"  logical para max                 : {max(all_unwrap) if all_unwrap else 0}")
    w(f"  max single PHYSICAL line         : {max(all_lines) if all_lines else 0}")
    w(f"  headings recognised              : {total_head}")
    w(f"  numeric-with-unit lines          : {total_numeric}")
    w(f"  'Label: value' lines             : {total_lv}")
    w(f"  boilerplate lines                : {total_boiler}  ({total_boiler/max(1,total_lines)*100:.1f}% of lines)")
    w("=" * 100)

    text = "\n".join(out)
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        print(f"wrote {p} ({len(text):,} chars)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
