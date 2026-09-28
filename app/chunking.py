"""Chunking - architecture.md §4.2, strategy in docs/chunking_strategy.md.

Structure-first, size-bounded. The corpus is hard-wrapped pypdf output, so the
walk is:

    normalise -> de-wrap into logical units -> split units into sentences
    -> group into chunks under the current heading
    -> break at CHUNK_SIZE, carrying TAIL_OVERLAP of context

Two invariants the tests enforce, and the reason this is not a generic splitter:

  1. A sentence containing a number with a unit is never split from its
     conditions. "an Exit Load of 1.00% is payable" without "within 1 year" is a
     confidently wrong answer.
  2. A chunk never crosses a file boundary. All five schemes come from one
     master factsheet PDF, so Large Cap's exit load can otherwise be returned
     for a question about Small Cap.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from app import config
from app.normalise import (
    clean_heading,
    normalise,
    strip_source_header,
)

# A sentence ends at . ! or ? followed by a capital, a bullet or an open paren.
# Requires the terminator to be preceded by a word character so we do not split
# on "Rs." or an initial.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\u2022]|\d+\.?\s)")

# Numbers with units are protected: never split one from its label.
_NUMERIC_UNIT = re.compile(
    r"(?:\d[\d,]*\.?\d*\s*(?:%|per cent|bps))"
    r"|(?:Rs\.?\s*\.?\s*[\d,]+)"
    r"|(?:₹\s*[\d,]+)"
    r"|(?:[\d,]+\s*Cr)"
    r"|(?:\d+\s*(?:years?|months?|days?))",
    re.IGNORECASE,
)

# `Label: value` pairs are atomic.
_LABEL_VALUE = re.compile(r"^[A-Z][A-Za-z0-9 /&()\-.']{2,60}:\s*\S")

# A heading is a short line matching one of these. Split into TIGHT (the whole
# line must be the heading, allowing trailing layout glyphs) and PROSE (a
# heading word followed by ":" or "-"). "NAV" and "PORTFOLIO" are ordinary
# English words and are TIGHT-only, or they match mid-sentence.
_HEADING_TIGHT = (
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
_TRAILING_GLYPH = r"[\s$#€¥£№:.\-–—]*"
_HEADING_PROSE = re.compile(
    r"^(INVESTMENT OBJECTIVE|EXIT LOAD|LOCK-?IN PERIOD|EXPENSE RATIO"
    r"|BENCHMARK INDEX|ADDL\. BENCHMARK INDEX|MINIMUM APPLICATION AMOUNT"
    r"|ACCOUNT STATEMENTS?|CONSOLIDATED ACCOUNT STATEMENT"
    r"|SERVICES PROVIDED FOR INVESTORS)",
    re.IGNORECASE,
)
_PROSE_MAX_LEN = 90

# "EXIT LOAD$$" carries two trailing "$" glyphs that `clean_heading` strips,
# leaving "EXIT LOAD" with no trailing colon. So the PROSE family would accept it
# on the colon/dash rule and reject it on length, while the TIGHT family never
# sees it. The result was that EXIT LOAD and #BENCHMARK INDEX were absorbed into
# the preceding EXPENSE RATIO section and the exit-load chunk disappeared
# entirely. These two are therefore also TIGHT.
_EXTRA_TIGHT = (r"EXIT LOAD", r"BENCHMARK INDEX", r"LOCK-?IN PERIOD")

# A heading fused onto the end of a text line. Only matches when a sizeable run
# of text follows, so it cannot fire on a heading in its natural position.
_FUSED_HEADING = re.compile(
    r"(?<=[a-z0-9\)\.])\s+("
    r"PORTFOLIO|QUANTITATIVE DATA|EXPENSE RATIO|LOCK-?IN PERIOD|FUND MANAGER"
    r"|BENCHMARK INDEX|ADDL\. BENCHMARK INDEX|DATE OF ALLOTMENT/INCEPTION DATE"
    r"|CATEGORY OF SCHEME|ASSETS UNDER MANAGEMENT|SERVICES PROVIDED FOR INVESTORS"
    r")(?=[\s.,:]|$)",
    re.IGNORECASE,
)

# Isolated single letter between spaces is the signature of a dropped glyph in
# pypdf output: "llocation" <- "Allocation", "usiness" <- "business".
_DROPPED_GLYPH = re.compile(r"(?<=[a-z]) [a-z] (?=[a-z])")

# A running page header is often fused onto the end of a text line, e.g.
# "BENCHMARK AND SCHEME RISKOMETERS 123 | June 2026". A line-anchored rule
# misses it, so this is matched anywhere. Only the "<n> | <Month> <year>" part is
# removed; the running title is left, since in the annexure it is also the
# section heading.
_PAGE_HEADER_INLINE = re.compile(
    r"\b\d{1,3}\s*\|\s*(?:January|February|March|April|May|June|July|August"
    r"|September|October|November|December)(?:\s+\d{4})?\b"
)

_BULLETS = "•►▪●‣"


@dataclass(frozen=True)
class Chunk:
    """The unit of retrieval and citation - architecture.md §4.3."""

    chunk_id: str
    text: str
    embed_text: str
    scheme: str
    scheme_slug: str
    amc: str
    category: str
    source_type: str
    source_title: str
    source_url: str
    section: str
    chunk_index: int
    as_of_date: str
    fetched_at: str
    corpus_version: str
    pages: str = ""
    text_hash: str = ""
    text_quality: str = "clean"

    def to_dict(self) -> dict:
        return asdict(self)


def is_heading(line: str) -> bool:
    """True only for structural headings, never for a sentence that starts like one."""
    s = clean_heading(line)
    if not s:
        return False
    for pat in _EXTRA_TIGHT:
        if re.fullmatch(pat + _TRAILING_GLYPH, s, re.IGNORECASE):
            return True
    for pat in _HEADING_TIGHT:
        if re.fullmatch(pat + _TRAILING_GLYPH, s, re.IGNORECASE):
            return True
    m = _HEADING_PROSE.match(s)
    if m:
        tail = s[m.end():].lstrip()
        if tail[:1] in {":", "-"} and len(s) <= _PROSE_MAX_LEN:
            return True
    return False


def classify_quality(text: str) -> str:
    """`degraded` when the extraction dropped a glyph.

    507 of 9,571 corpus lines (5.3%) carry this signature, concentrated in the
    KIMs. A chunk flagged `degraded` must not be quoted for a bare figure.
    """
    return "degraded" if _DROPPED_GLYPH.search(text) else "clean"


def split_sentences(text: str) -> list[str]:
    """Split a logical unit into sentences, gluing short fragments back together.

    The 220-char glue threshold matters: pypdf often leaves table fragments and
    list items without terminators, and a bare split would emit "Nil" as its own
    sentence with no label.
    """
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p and p.strip()]
    out: list[str] = []
    for part in parts:
        if out and len(out[-1]) + 1 + len(part) <= 220:
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return out


def overlap_tail(text: str, overlap: int) -> str:
    """Last `overlap` chars, advanced to the next word boundary.

    A raw `text[-overlap:]` slices mid-word, so the next chunk opens on a
    fragment like "iation / income" - which reads as corruption and, worse,
    embeds as a context-free string. Advancing to the next space makes the
    carried context a whole trailing phrase.
    """
    if not overlap or not text:
        return ""
    if overlap >= len(text):
        return text.strip()
    tail = text[-overlap:]
    space = tail.find(" ")
    if 0 <= space < len(tail) - 1:
        tail = tail[space + 1:]
    return tail.strip()


#: A lowercase opener is the only unambiguous continuation signal in this corpus:
#: "under Regulation 52 (6)..." or "reduction of 0.05% for..." cannot begin a
#: section. A capitalised opener is NOT reliable, because the corpus is full of
#: wrapped table fragments ("Inception Date: 01-02-1994") that start with a
#: capital yet continue a sentence. Those are not marked; the numeric-sentence
#: invariant covers the case that actually matters.
_CONTINUATION_RE = re.compile(r"^[a-z]")

#: Marks a chunk that provably begins mid-sentence, so a reader (and a
#: retriever) can tell a continuation from a genuine document start.
CONTINUATION_MARKER = "... "


def is_continuation(text: str) -> bool:
    """True when a chunk provably begins mid-sentence."""
    stripped = text.lstrip()
    if not stripped:
        return False
    if stripped.startswith(CONTINUATION_MARKER):
        return True
    return bool(_CONTINUATION_RE.match(stripped))


def _is_protected(sentence: str) -> bool:
    """Never split one of these away from what it belongs to."""
    return bool(_NUMERIC_UNIT.search(sentence) or _LABEL_VALUE.match(sentence.strip()))


def split_oversized(text: str, limit: int) -> list[str]:
    """Force-split a unit that has no sentence boundary - a flattened table run.

    Cuts at the last space before the limit, then marks the result by leaving
    the caller to flag it. Cutting mid-number is avoided by preferring a break
    after a token that ends in a non-digit.
    """
    if len(text) <= limit:
        return [text]

    pieces: list[str] = []
    rest = text
    while len(rest) > limit:
        window = rest[:limit]
        cut = window.rfind(" ")
        if cut <= 0:
            cut = limit
        # Nudge the break left off a digit so we do not split a figure.
        while cut > 1 and window[cut - 1].isdigit() and not window[cut - 2].isspace():
            cut -= 1
        pieces.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        pieces.append(rest)
    return [p for p in pieces if p]


def _iter_units(body: str) -> list[tuple[str | None, str]]:
    """Return [(heading_or_None, logical_unit)] over normalised body text.

    De-wraps hard-wrapped lines and starts a new unit at each heading. Returns a
    list rather than yielding from a nested closure, which is what made the
    first version unreadable.
    """
    units: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        # buf holds one entry per physical line, already de-hyphenated per line
        # by app.normalise.normalise. Joining with a single space is safe here.
        joined = re.sub(r"\s+", " ", " ".join(buf)).strip()
        if joined:
            units.append((current_heading, joined))
        buf = []

    for raw in body.splitlines():
        s = _PAGE_HEADER_INLINE.sub(" ", raw.strip())
        s = re.sub(r"\s{2,}", " ", s).strip()
        if not s:
            continue
        if is_heading(s):
            flush()
            current_heading = clean_heading(s)
            continue
        # A heading can also arrive fused onto the end of a preceding text line,
        # e.g. "NAV. Inception Date: 01-02-1994" or
        # "...(since June 22, 2023) PORTFOLIO Face value / allotment ...". When
        # the tail after the heading is substantial, split the line there so the
        # heading starts its own unit and the section label stays accurate.
        m = _FUSED_HEADING.search(s)
        if m:
            head_part = s[: m.start()].strip()
            if head_part:
                buf.append(head_part)
            flush()
            current_heading = clean_heading(m.group(0))
            rest = s[m.end():].strip()
            if rest:
                buf.append(rest)
            continue
        buf.append(s)

    flush()
    return units


def _read_source(path: Path) -> tuple[str, dict[str, str]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    body, meta = strip_source_header(raw)
    return normalise(body), meta


def _derive_category(slug: str) -> str:
    if slug == "general":
        return "general"
    return slug.replace("hdfc-", "")


def load_and_chunk(
    raw_dir: Path | str | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
    corpus_version: str | None = None,
) -> list[Chunk]:
    """Load `data/raw/**`, normalise and chunk it. Pure with respect to disk reads.

    Ordering is by path so `chunk_id`s are reproducible across runs - the
    idempotency guarantee in tests/test_ingest.py depends on it.
    """
    raw_dir = Path(raw_dir or config.RAW_DIR)
    size = chunk_size or config.CHUNK_SIZE
    ov = overlap if overlap is not None else config.TAIL_OVERLAP
    version = corpus_version or config.CORPUS_VERSION
    hard_cap = int(size * (1 + config.FALLBACK_TOLERANCE_FACTOR))
    today = date.today().isoformat()

    chunks: list[Chunk] = []

    for path in sorted(raw_dir.glob("*/*.txt")):
        body, meta = _read_source(path)

        scheme_slug = path.parent.name
        scheme = meta.get("scheme") or config.SCHEMES.get(scheme_slug, scheme_slug)
        source_slug = f"{scheme_slug}-{path.stem}"

        pending: list[tuple[str, str]] = list(_iter_units(body))

        # Group units into chunks, tracking the live section.
        buf: list[str] = []
        buf_section = ""
        buf_heading: str | None = None
        per_source = 0

        def emit(buf_text: str, sec: str) -> None:
            nonlocal per_source
            text = re.sub(r"\s+", " ", buf_text).strip()
            if not text:
                return
            # MIN_CHUNK exists to drop orphans like a bare page number. It must
            # never drop a real fact: "LOCK-IN PERIOD 3 years from the date of
            # allotment" is 49 chars and is one of the most-asked facts in this
            # corpus. So the guard only drops text that is *only* a label or a
            # bare number - never text that carries a number with a unit.
            if len(text) < config.MIN_CHUNK and not _NUMERIC_UNIT.search(text):
                return

            # A chunk that opens mid-sentence is a continuation, whether the
            # cause was a size break, a heading change, or a fused-heading split.
            # Mark it so an auditor can tell a continuation from a genuine
            # document start, and so a retriever does not read "under Regulation
            # 52 (6)" as a complete statement.
            # Idempotent: a chunk already carrying the marker must not gain a
            # second one, which would also push it past the hard cap.
            if not text.startswith(CONTINUATION_MARKER) and is_continuation(text):
                text = CONTINUATION_MARKER + text

            index = per_source
            per_source += 1
            prefix = (
                f"Scheme: {scheme}\n"
                f"Document: {meta.get('title', source_slug)}"
                f" (as of {meta.get('as_of', 'unknown')})\n"
                f"Section: {sec}\n"
                f"Source: {meta.get('source_url', '')}\n"
                f"---\n"
            )
            chunks.append(
                Chunk(
                    chunk_id=f"{version}:{source_slug}:{index:04d}",
                    text=text,
                    embed_text=prefix + text,
                    scheme=scheme,
                    scheme_slug=scheme_slug,
                    amc=config.AMC_NAME,
                    category=_derive_category(scheme_slug),
                    source_type=meta.get("source_type", "unknown"),
                    source_title=meta.get("title", source_slug),
                    source_url=meta.get("source_url", ""),
                    section=sec,
                    chunk_index=index,
                    as_of_date=meta.get("as_of", ""),
                    fetched_at=meta.get("retrieved", today),
                    corpus_version=version,
                    pages=meta.get("pages", ""),
                    text_hash="sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                    text_quality=classify_quality(text),
                )
            )

        def current_len() -> int:
            return len(" ".join(buf))

        for heading, unit in pending:
            sec = heading or "General"

            # A heading change always starts a new chunk (strategy 2.1), even if
            # the open chunk is far below CHUNK_SIZE. EXPENSE RATIO and
            # #BENCHMARK INDEX answer different questions; merging them blurs
            # retrieval and mislabels the citation.
            if buf and heading != buf_heading:
                emit(" ".join(buf), buf_section or sec)
                buf = []

            buf_heading = heading
            buf_section = sec

            for sent in split_sentences(unit):
                if not sent:
                    continue
                for piece in split_oversized(sent, hard_cap):
                    if buf and current_len() + 1 + len(piece) > size:
                        joined = " ".join(buf)
                        emit(joined, buf_section)
                        # Carry the tail so a split clause stays interpretable.
                        # Trim it to whatever room the next piece leaves, so the
                        # overlap itself cannot push a chunk past the hard cap.
                        # The ellipsis marks this as carried context from the
                        # previous chunk. Without it a continuation chunk is
                        # indistinguishable from a mid-sentence fragment, and an
                        # auditor reading chunks.txt cannot tell which.
                        #
                        # The marker is counted in the budget, not added on top:
                        # the emitted text is the tail joined to the piece, so the
                        # final whitespace-collapse step must not be able to push
                        # it past the hard cap.
                        tail = overlap_tail(joined, ov)
                        room = hard_cap - len(piece) - len(CONTINUATION_MARKER) - 1
                        if tail and room > 0:
                            tail = CONTINUATION_MARKER + tail[-room:].lstrip()
                        else:
                            tail = ""
                        buf = [tail, piece] if tail else [piece]
                    else:
                        buf.append(piece)

        emit(" ".join(buf), buf_section)

    return chunks
