"""Text normalisation - architecture.md §4.1 stage 2.

Pure functions, no I/O, no global state. Every function returns a new string and
is safe to call in any order, which is what makes the Phase 2 measurements
reproducible.

The corpus in `data/raw/` is pypdf output, so the work here is specific to that:
re-join hard-wrapped lines, strip running headers and footers, and remove the
layout glyphs pypdf leaves on headings.
"""

from __future__ import annotations

import re
import unicodedata

# Zero-width and invisible formatting characters, plus the soft hyphen.
_ZERO_WIDTH = re.compile(r"[‌‍‎‏﻿­]")

# Layout glyphs pypdf leaves behind when it reconstructs a heading from a PDF
# text layer. "EXIT LOAD$$" and "#BENCHMARK INDEX" are real headings; the glyphs
# are not part of them and must not reach a citation.
_TRAILING_GLYPHS = re.compile(r"[\s$#€¥£№]+$")
_HEADING_PREFIX_NUM = re.compile(r"^#{1,6}\s+")
_BULLET_GLYPHS = "•►▪●‣·"

# Running header/footer text that repeats on every page and carries no fact.
_BOILERPLATE = (
    re.compile(r"^For Product label and Riskometers, refer page no", re.IGNORECASE),
    re.compile(r"^MUTUAL FUND INVESTMENTS ARE SUBJECT TO MARKET RISKS", re.IGNORECASE),
    re.compile(r"^Read all scheme related documents carefully", re.IGNORECASE),
    re.compile(r"^HDFC MF SAI\s*-\s*dated", re.IGNORECASE),
    re.compile(r"^Page \d+ of \d+$", re.IGNORECASE),
    re.compile(r"^\.{3,}\s*Contd", re.IGNORECASE),
    re.compile(r"^Contd on (the )?next page$", re.IGNORECASE),
    re.compile(r"^\d{1,3}\s*\|\s*(January|February|March|April|May|June|July|August"
               r"|September|October|November|December)\b", re.IGNORECASE),
)

# A word split mid-token by pypdf, e.g. "mitigat e" -> "mitigate".
#
# IMPORTANT: this must only ever be applied WITHIN a single physical line. The
# corpus is hard-wrapped, so joining lines first and then de-hyphenating merges
# ordinary word pairs across the wrap ("the" + "functioning" -> "thefunctioning").
_MIDWORD_SPACE = re.compile(r"\b([a-z]{1,4}) ([a-z]{2,})\b")

# Short English words that are never a broken word. Without this list the rule
# eats "to the", "of the", "and the" and produces fused non-words that embed
# badly and read as corruption in chunks.txt.
_NOT_A_FRAGMENT = frozenset(
    """a an and are as at be by for from has have he her his in is it its of on or
    she that the their them then there these they this to was were will with you
    your all any but can could do does each if into may more no not our out so
    such than too us we when who whom whose which while who""".split()
)

# Capitalised proper nouns that a midword rule could corrupt. Cheap insurance.
_PROTECT = re.compile(
    r"\b(Annual|Market|Net|Asset|Value|Total|Expense|Ratio|Exit|Load|"
    r"Benchmark|Index|Account|Statement|Plan|Option|Units|Unit|Class)\b"
)

_PAGE_HEADER = re.compile(
    r"^(\d{1,3})\s*\|\s*(January|February|March|April|May|June|July|August"
    r"|September|October|November|December)\b",
    re.IGNORECASE,
)


def is_boilerplate(line: str) -> bool:
    """True for running headers/footers that repeat on every page."""
    s = line.strip()
    if not s:
        return False
    return any(p.match(s) for p in _BOILERPLATE)


def normalise_dashes_and_quotes(text: str) -> str:
    """Fold typographic variants to ASCII so matching and display agree."""
    for src, dst in (
        ("\u2018", "'"), ("\u2019", "'"), ("\u201a", "'"), ("\u201b", "'"),
        ("\u201c", '"'), ("\u201d", '"'), ("\u201e", '"'), ("\u201f", '"'),
        ("\u2013", "-"), ("\u2014", "-"), ("\u2012", "-"), ("\u2015", "-"),
        ("\u2212", "-"), ("\u00a0", " "), ("\u2044", "/"),
    ):
        text = text.replace(src, dst)
    return text


def strip_zero_width(text: str) -> str:
    """Remove zero-width and soft-hyphen characters."""
    return _ZERO_WIDTH.sub("", text)


def clean_heading(text: str) -> str:
    """Remove layout glyphs and stray markdown from a heading string.

    "EXIT LOAD$$" -> "EXIT LOAD"; "##ADDL. BENCHMARK INDEX" -> "ADDL. BENCHMARK INDEX"
    """
    text = _HEADING_PREFIX_NUM.sub("", text.strip())
    text = _TRAILING_GLYPHS.sub("", text)
    # A heading may carry a bullet or marker glyph in front of it.
    text = text.lstrip(_BULLET_GLYPHS).strip()
    return _TRAILING_GLYPHS.sub("", text)


def dehyphenate_line(line: str) -> str:
    """Re-join a word pypdf split with a space mid-token, WITHIN one line.

    Safe to call per line; never call it on text whose lines have already been
    joined, or it will fuse ordinary word pairs across a line break.

    Conservative by design: joins only when the first fragment is short, is not
    an English word in its own right, and the pair is not a protected phrase.
    """
    def repl(m: re.Match[str]) -> str:
        left, right = m.group(1), m.group(2)
        if left in _NOT_A_FRAGMENT:
            return m.group(0)
        if _PROTECT.search(m.group(0)):
            return m.group(0)
        # "of the" and friends survive via _NOT_A_FRAGMENT; this catches the
        # case where a real word is a prefix of a broken one ("the" + "se").
        if left + right[:1] in _NOT_A_FRAGMENT and left in _NOT_A_FRAGMENT:
            return m.group(0)
        return f"{left}{right}"

    return _MIDWORD_SPACE.sub(repl, line)


def collapse_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs, keep intentional newlines."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def flatten_table_row(cells: list[str]) -> str:
    """Render a table row as a single `Label: value` line.

    The extraction flattens tables to whitespace-separated tokens, so the row
    boundary has to be recovered from the text itself. This helper keeps the
    rendering in one place so the chunker and the tests agree.
    """
    parts = [c.strip() for c in cells if c and c.strip()]
    if len(parts) == 1:
        return parts[0]
    return " ".join(parts)


def strip_source_header(text: str) -> tuple[str, dict[str, str]]:
    """Split the leading `SOURCE DOCUMENT` metadata block off the body.

    Returns (body, metadata). Metadata is re-attached per chunk rather than
    being chunked, so provenance never competes with facts for chunk space.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "SOURCE DOCUMENT":
        return text, {}

    meta: dict[str, str] = {}
    key_re = re.compile(r"^(title|scheme|source_type|publisher|source_url|doc_date|as_of|pages|retrieved)\s*:")

    i = 1
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("=" * 10):
            i += 1
            break
        m = key_re.match(stripped)
        if m:
            meta[m.group(1)] = stripped.split(":", 1)[1].strip()
        i += 1

    return "\n".join(lines[i:]).lstrip("\n"), meta


def drop_boilerplate_lines(text: str) -> str:
    """Remove running headers/footers.

    A bare numeric line is removed ONLY when it directly follows a `<page> |
    <Month>` running header - in this corpus a lone "4" or "13" is far more
    often a table value than a page number, so a blanket `^\\d+$` rule would
    silently delete real data.
    """
    kept: list[str] = []
    prev_was_page_header = False
    for line in text.splitlines():
        s = line.strip()
        if not s:
            kept.append("")
            prev_was_page_header = False
            continue
        if _PAGE_HEADER.match(s):
            kept.append(line)
            prev_was_page_header = True
            continue
        if re.fullmatch(r"\d{1,3}", s) and prev_was_page_header:
            prev_was_page_header = False
            continue
        prev_was_page_header = False
        if is_boilerplate(s):
            continue
        kept.append(line)
    return "\n".join(kept)


def normalise(text: str) -> str:
    """Full pipeline: raw extracted text -> clean text.

    Order matters. Boilerplate is dropped before whitespace is collapsed, since
    a running header is a whole line and collapse would fuse it to its
    neighbour.
    """
    text = unicodedata.normalize("NFC", text)
    text = strip_zero_width(text)
    text = normalise_dashes_and_quotes(text)
    text = text.replace("\ufffd", "")          # pypdf replacement char
    text = drop_boilerplate_lines(text)
    # De-hyphenation is per line, before anything joins lines together.
    text = "\n".join(dehyphenate_line(line) for line in text.split("\n"))
    text = collapse_whitespace(text)
    return text
