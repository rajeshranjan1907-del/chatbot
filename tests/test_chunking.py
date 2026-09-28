"""Phase 2 -> Phase 3 chunking contract.

These six tests encode the rules chosen in docs/chunking_strategy.md. They were
written BEFORE the chunker existed and were expected to fail; they are the
specification, not a report card. Do not weaken them to make a run go green.

    pytest tests/test_chunking.py -v

Access note: chunks are `app.chunking.Chunk` dataclasses per architecture §4.3,
so these use attribute access. An earlier draft used dict access; the dataclass
is the contract, so the tests were updated rather than the other way round.
"""

from __future__ import annotations

import re
import sys
from dataclasses import fields
from pathlib import Path

import pytest

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MISSING_CHUNKER = (
    "Phase 3 deliverable not written yet: app/chunking.py must expose "
    "load_and_chunk(raw_dir, chunk_size, overlap) -> list[Chunk]"
)


@pytest.fixture(scope="module")
def chunks() -> list:
    """Load the real corpus through the Phase 3 chunker.

    Fails - rather than skips - while app/chunking.py is absent, so Phase 3 has
    a measurable target: 6 collected, 6 failing -> 6 collected, 6 passing.
    A skip would report 0 collected, which is indistinguishable from "no tests
    exist" and would let the gate pass by accident.
    """
    try:
        from app.chunking import load_and_chunk
    except ImportError as exc:  # pragma: no cover - Phase 3 not started
        raise AssertionError(f"{MISSING_CHUNKER} (import error: {exc})") from exc

    return load_and_chunk(RAW, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)


def _find(chunks: list, scheme: str, section_contains: str) -> list:
    return [
        c
        for c in chunks
        if c.scheme_slug == scheme
        and section_contains.lower() in (c.section or "").lower()
    ]


# ---------------------------------------------------------------- test 1
def test_every_chunk_carries_all_nine_metadata_fields(chunks):
    """PRD requires one official citation per answer, so provenance is per chunk."""
    required = {
        "chunk_id",
        "scheme",
        "scheme_slug",
        "source_type",
        "source_url",
        "as_of_date",
        "section",
        "text",
    }
    assert chunks, "chunker produced no chunks"

    present = {f.name for f in fields(chunks[0])}
    missing = required - present
    assert not missing, f"Chunk dataclass is missing {sorted(missing)}"

    for c in chunks:
        assert c.source_url.startswith("https://"), (
            f"{c.chunk_id} source_url is not an https official URL: {c.source_url!r}"
        )
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", c.as_of_date), (
            f"{c.chunk_id} as_of_date is not ISO-8601: {c.as_of_date!r}"
        )
        assert c.text.strip(), f"{c.chunk_id} has empty text"
        assert c.embed_text.strip(), f"{c.chunk_id} has empty embed_text"
        # The context prefix must reach the embedding but not leak into the
        # stored text a citation renders (architecture §4.2).
        assert len(c.embed_text) > len(c.text), (
            f"{c.chunk_id} embed_text is not longer than text: no context prefix"
        )
        assert c.embed_text.endswith(c.text), (
            f"{c.chunk_id} embed_text does not end with its stored text; "
            "the prefix should be prepended, not interleaved"
        )
        assert "Scheme: " in c.embed_text.split("---")[0], (
            f"{c.chunk_id} embed_text is missing the scheme context prefix"
        )


# ---------------------------------------------------------------- test 2
def test_chunks_never_split_a_numeric_sentence(chunks):
    """Strategy 2.3(1): a number must never be separated from its conditions.

    Guards the failure that motivates the whole rule: emitting "an Exit Load of
    1.00% is payable" without "within 1 year from the date of allotment" is a
    confidently wrong answer.
    """
    # Invariant 1: a chunk that provably opens mid-sentence must be MARKED as a
    # continuation. A lowercase opener can only be a continuation, so it must
    # carry the "... " marker - otherwise a reader (or a retriever) reads its
    # numbers as complete statements.
    unmarked = [
        c.chunk_id
        for c in chunks
        if c.text[:1].islower() and not c.text.startswith("... ")
    ]
    assert not unmarked, (
        f"{len(unmarked)} chunks open mid-sentence without the continuation "
        f"marker: {unmarked[:10]}"
    )

    # Invariant 2 (the one that actually matters): a numeric rule must never be
    # separated from its qualifying conditions. "an Exit Load of 1.00% is
    # payable" without "within 1 year from the date of allotment" is a
    # confidently wrong answer, so both clauses must land in the same chunk.
    for slug in ("hdfc-large-cap", "hdfc-flexi-cap", "hdfc-small-cap"):
        found = _find(chunks, slug, "EXIT LOAD")
        assert found, f"{slug}: no EXIT LOAD chunk produced"
        joined = " ".join(c.text for c in found)
        assert "1.00%" in joined, f"{slug}: exit load percentage missing from chunks"
        assert re.search(r"within\s*1\s*year", joined), (
            f"{slug}: exit load lost its 'within 1 year' condition - "
            "the chunk would answer with an unconditional load"
        )
        # The two exit-load clauses must not be split apart either.
        both = any("within 1 year" in c.text and "No Exit Load" in c.text
                   for c in found)
        assert both, (
            f"{slug}: the two exit-load clauses landed in different chunks; a "
            "retrieval could return the 1.00% rule without the nil-after-1-year "
            "exemption"
        )

    # The ELSS is the mirror case: lock-in and a nil exit load must co-occur.
    elss = _find(chunks, "hdfc-elss", "LOCK")
    assert elss, "hdfc-elss: no LOCK-IN chunk produced"
    assert any("3 years" in c.text for c in elss), (
        "hdfc-elss: lock-in chunk lost its 3-year term"
    )

    # And the real corpus must retain the intact exit-load rule.
    for slug in ("hdfc-large-cap", "hdfc-flexi-cap", "hdfc-small-cap"):
        found = _find(chunks, slug, "EXIT LOAD")
        assert found, f"{slug}: no EXIT LOAD chunk produced"
        joined = " ".join(c.text for c in found)
        assert "1.00%" in joined, f"{slug}: exit load percentage missing from chunks"
        assert re.search(r"within\s*1\s*year", joined), (
            f"{slug}: exit load lost its 'within 1 year' condition - "
            "the chunk would answer with an unconditional load"
        )


# ---------------------------------------------------------------- test 3
def test_label_value_pairs_stay_together(chunks):
    """Strategy 2.3(2): `Regular: 1.56% Direct: 1.03%` is one indivisible unit."""
    expected = {
        "hdfc-large-cap": ("1.56", "1.03"),
        "hdfc-flexi-cap": ("1.35", "0.75"),
        "hdfc-elss": ("1.75", "1.18"),
        "hdfc-small-cap": ("1.55", "0.76"),
        "hdfc-balanced-advantage": ("1.29", "0.77"),
    }
    for slug, (regular, direct) in expected.items():
        found = _find(chunks, slug, "EXPENSE RATIO")
        assert found, f"{slug}: no EXPENSE RATIO chunk produced"
        # Both plan figures must appear in a single chunk. Splitting them lets
        # the model quote one plan's ratio while labelling the other.
        hits = [c for c in found if regular in c.text and direct in c.text]
        assert hits, (
            f"{slug}: expected Regular {regular}% and Direct {direct}% in the SAME "
            f"EXPENSE RATIO chunk, found {len(found)} chunk(s); text was: "
            f"{[c.text[:120] for c in found]}"
        )


# ---------------------------------------------------------------- test 4
def test_boilerplate_is_stripped_and_no_layout_glyphs_leak(chunks):
    """Strategy 2.4 steps 2 and 5.

    `EXIT LOAD$$` in a citation looks like a bug to a marker, and
    `....Contd on next page` is pure noise that dilutes the embedding.
    """
    banned = [
        "....Contd",
        "refer page no: 123-138",
        "MUTUAL FUND INVESTMENTS ARE SUBJECT TO MARKET RISKS",
    ]
    for c in chunks:
        for phrase in banned:
            assert phrase not in c.text, (
                f"{c.chunk_id} still contains boilerplate {phrase!r}"
            )
        assert "$$" not in c.section, f"{c.chunk_id} section kept '$$': {c.section!r}"
        assert "##" not in c.section, f"{c.chunk_id} section kept '##': {c.section!r}"

    # Headings must be cleaned, not deleted: EXIT LOAD must still be findable.
    for slug in ("hdfc-large-cap", "hdfc-elss"):
        assert _find(chunks, slug, "EXIT LOAD"), (
            f"{slug}: stripping removed the EXIT LOAD heading entirely"
        )


# ---------------------------------------------------------------- test 5
def test_chunks_never_cross_a_scheme_boundary(chunks):
    """All five schemes come from ONE master factsheet PDF.

    Without an explicit boundary, a retriever can return HDFC Large Cap's exit
    load for a question about Small Cap. `chunk_id` embeds the scheme slug so
    this stays true after the corpus is regenerated.
    """
    slugs = {
        "hdfc-large-cap",
        "hdfc-flexi-cap",
        "hdfc-elss",
        "hdfc-small-cap",
        "hdfc-balanced-advantage",
        "general",
    }
    for c in chunks:
        assert c.scheme_slug in slugs, (
            f"{c.chunk_id} has unexpected scheme_slug {c.scheme_slug!r}"
        )
        assert c.scheme_slug in c.chunk_id, (
            f"chunk_id {c.chunk_id!r} does not encode its scheme_slug "
            f"{c.scheme_slug!r}, so a citation cannot identify the scheme"
        )
        if c.scheme_slug == "general":
            # Cross-scheme material is not a fund, so it carries a descriptive
            # name rather than one of the five scheme names.
            assert c.scheme.strip(), f"{c.chunk_id} has an empty scheme name"
        else:
            # Slugs hyphenate, display names use spaces: "balanced-advantage" vs
            # "Balanced Advantage". Compare on a normalised form.
            slug_words = c.scheme_slug.replace("hdfc-", "").replace("-", " ")
            assert slug_words in re.sub(r"[^a-z ]", " ", c.scheme.lower()), (
                f"{c.chunk_id} scheme {c.scheme!r} disagrees with slug "
                f"{c.scheme_slug!r}"
            )

    # And the ELSS-only fact must not leak into a non-ELSS scheme.
    for slug in ("hdfc-large-cap", "hdfc-flexi-cap", "hdfc-small-cap",
                 "hdfc-balanced-advantage"):
        for c in _find(chunks, slug, "LOCK"):
            assert "3 year" not in c.text.lower(), (
                f"{slug} inherited the ELSS 3-year lock-in: {c.text[:120]!r}"
            )


# ---------------------------------------------------------------- test 6
def test_degraded_chunks_are_flagged_and_small_enough_to_index(chunks):
    """Strategy 5 and 2.2: the glyph-loss guard and the size contract.

    507 of 9,571 corpus lines (5.3%) have a dropped character. A chunk carrying
    that signature must be flagged, and no chunk may exceed the size limit by
    more than the documented fallback tolerance - otherwise a single flattened
    table run can blow the embedding budget.
    """
    dropped_glyph = re.compile(r"(?<=[a-z]) [a-z] (?=[a-z])")
    fallback_tolerance = int(CHUNK_SIZE * 0.6)  # 900 * 0.6 = 540 -> hard cap 1440

    flagged = 0
    for c in chunks:
        if not hasattr(c, "text_quality"):
            raise AssertionError(
                f"{c.chunk_id} has no 'text_quality' field; the dropped-glyph "
                "guard from strategy section 5 is not implemented"
            )
        if dropped_glyph.search(c.text):
            assert c.text_quality == "degraded", (
                f"{c.chunk_id} contains a dropped-glyph pattern but is marked "
                f"{c.text_quality!r}"
            )
            flagged += 1
        else:
            assert c.text_quality == "clean", (
                f"{c.chunk_id} marked {c.text_quality!r} but shows no "
                "dropped-glyph pattern"
            )
        assert len(c.text) <= CHUNK_SIZE + fallback_tolerance, (
            f"{c.chunk_id} is {len(c.text)} chars, over the {CHUNK_SIZE} "
            f"limit plus {fallback_tolerance} tolerance"
        )
        assert c.text.strip(), f"{c.chunk_id} is whitespace-only"

    # The flag must actually discriminate, not be a constant.
    assert flagged, "no chunk was flagged degraded; the glyph guard is inert"
    assert flagged < len(chunks), "every chunk is degraded; the guard is useless"
