"""Retrieval tests - implementation.md Phase 4 step 5.

The three contracts that matter, in order of how badly a violation would hurt:

  1. A scheme that is NOT in this corpus must return ok=False. This is the one
     that produces a confidently wrong answer if it breaks, so it is asserted
     against several real funds, not just one.
  2. A named-scheme question must return only that scheme's chunks. Without this
     the bot answers "expense ratio of HDFC Small Cap" from the Large Cap sheet.
  3. Nonsense must return ok=False rather than the least-bad chunk.

Plus: SIM_FLOOR is applied to similarity not distance, INSUFFICIENT reasons are
distinct, and the CLI runs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app import config
from app.localindex import LocalIndex
from app.retrieve import (
    INSUFFICIENT_BELOW_FLOOR,
    INSUFFICIENT_FACT_ABSENT,
    INSUFFICIENT_NO_HITS,
    INSUFFICIENT_SCHEME_MISMATCH,
    INSUFFICIENT_UNKNOWN_SCHEME,
    Context,
    detect_foreign_scheme,
    detect_scheme,
    retrieve,
)

pytestmark = pytest.mark.skipif(
    not LocalIndex.exists(),
    reason="no vector index; run: python scripts/build_index.py",
)


@pytest.fixture(scope="module")
def index() -> LocalIndex:
    return LocalIndex.load()


# ------------------------------------------------- 1. off-corpus schemes refuse
@pytest.mark.parametrize(
    "question",
    [
        "expense ratio of Parag Parivartan Flexi Cap Fund",
        "expense ratio of SBI Bluechip Fund",
        "what is the exit load of Axis Small Cap Fund?",
        "tell me about Kotak Flexicap Fund",
        "is there a lock-in on Parag Parivartan ELSS?",
    ],
)
def test_off_corpus_scheme_is_refused(question, index):
    """A different AMC must never be answered with a lookalike HDFC fund."""
    ctx = retrieve(question, index)
    assert not ctx.ok, (
        f"{question!r} was answered with chunks from this corpus. "
        f"Got: {[c.chunk_id for c in ctx.chunks]}"
    )
    # The refusal reason depends on which gate fired first, and step 4 (the
    # floor) legitimately runs before step 5 (the foreign-scheme check). A
    # "Parag Parivartan ELSS" question is lexically far from every HDFC chunk,
    # so the floor catches it before the AMC is even considered. Either way the
    # answer is a refusal, which is what this test protects.
    assert ctx.reason in (
        INSUFFICIENT_UNKNOWN_SCHEME,
        INSUFFICIENT_BELOW_FLOOR,
        INSUFFICIENT_NO_HITS,
    ), f"unexpected reason {ctx.reason!r}"


def test_foreign_scheme_beats_a_colliding_alias(index):
    """"flexi cap" is an HDFC alias AND the tail of Parag Parivartan Flexi Cap.

    The alias must not win. This is the regression that returned HDFC's own
    1.35% for a Parag Parivartan question.
    """
    assert detect_scheme("expense ratio of Parag Parivartan Flexi Cap Fund")[0] == "hdfc-flexi-cap"
    assert detect_foreign_scheme("expense ratio of Parag Parivartan Flexi Cap Fund") == "parag parivartan"
    assert not retrieve("expense ratio of Parag Parivartan Flexi Cap Fund", index).ok


# ------------------------------------------- 2. named-scheme filter stays pure
@pytest.mark.parametrize(
    "question,slug",
    [
        ("expense ratio of HDFC ELSS Tax Saver Fund", "hdfc-elss"),
        ("exit load on HDFC Small Cap Fund", "hdfc-small-cap"),
        ("what is the expense ratio of HDFC Large Cap Fund?", "hdfc-large-cap"),
        ("benchmark of HDFC Balanced Advantage Fund", "hdfc-balanced-advantage"),
        ("expense ratio of HDFC Flexi Cap Fund", "hdfc-flexi-cap"),
    ],
)
def test_named_scheme_returns_only_that_scheme(question, slug, index):
    ctx = retrieve(question, index)
    assert ctx.ok, f"{question!r} returned ok=False ({ctx.reason})"
    assert ctx.chunks, f"{question!r} returned ok=True with no chunks"
    wrong = {c.scheme_slug for c in ctx.chunks} - {slug}
    assert not wrong, (
        f"{question!r} leaked other schemes into the context: {sorted(wrong)}"
    )


def test_scheme_alias_forms_all_resolve(index):
    """Users do not type the canonical name, so the aliases must all work."""
    for phrase in ("hdfc elss", "elss tax saver", "tax saver", "hdfc smallcap", "flexicap"):
        slug, _ = detect_scheme(f"expense ratio of {phrase}")
        assert slug is not None, f"alias {phrase!r} did not resolve"


def test_elss_expense_ratio_returns_the_real_numbers(index):
    """The top hit must actually contain the ratio, not just the word."""
    ctx = retrieve("expense ratio of HDFC ELSS Tax Saver Fund", index)
    assert ctx.ok
    top = ctx.chunks[0]
    joined = " ".join(c.text for c in ctx.chunks[:3])
    assert "1.75%" in joined, "ELSS Regular expense ratio missing from top-3 context"
    assert "1.18%" in joined, "ELSS Direct expense ratio missing from top-3 context"
    assert "HDFC ELSS" in top.text or "expense" in top.section.lower()


def test_exit_load_keeps_its_qualifying_condition(index):
    """1.00% without "within 1 year" is a wrong answer, not a terse one."""
    ctx = retrieve("exit load on HDFC Small Cap Fund", index)
    assert ctx.ok
    joined = " ".join(c.text for c in ctx.chunks)
    assert "1.00%" in joined
    assert "within 1 year" in joined.replace("1 yearfrom", "1 year from")


# ------------------------------------------------------- 3. nonsense refuses
@pytest.mark.parametrize(
    "question",
    [
        "asdkjh qwe zxcv",
        "what is the meaning of life",
        "who won the 1998 world cup",
        "zzzz qqqq",
    ],
)
def test_nonsense_returns_insufficient(question, index):
    ctx = retrieve(question, index)
    assert not ctx.ok, (
        f"{question!r} was answered with {[c.chunk_id for c in ctx.chunks]}. "
        "A nonsense question must not return the least-bad chunk."
    )


def test_empty_question_is_insufficient(index):
    ctx = retrieve("   ", index)
    assert not ctx.ok
    assert ctx.reason == INSUFFICIENT_NO_HITS


# ------------------------------------------------------------- floor behaviour
def test_floor_is_applied_to_similarity_not_distance(index):
    """Regression guard: a distance/similarity mix-up inverts the filter."""
    ctx = retrieve("asdkjh qwe zxcv", index)
    assert not ctx.ok
    assert ctx.reason in (INSUFFICIENT_BELOW_FLOOR, INSUFFICIENT_NO_HITS)

    ok_ctx = retrieve("expense ratio of HDFC Large Cap Fund", index)
    assert all(s >= config.SIM_FLOOR for s in ok_ctx.scores), (
        f"a chunk below SIM_FLOOR survived: "
        f"{[round(s, 4) for s in ok_ctx.scores]}"
    )


def test_scheme_mismatch_is_distinct_from_below_floor(index):
    """The two INSUFFICIENT reasons need different user-facing copy."""
    assert INSUFFICIENT_SCHEME_MISMATCH != INSUFFICIENT_BELOW_FLOOR
    assert INSUFFICIENT_UNKNOWN_SCHEME != INSUFFICIENT_BELOW_FLOOR


# ------------------------------------------------------------------- Context
def test_context_scores_align_with_chunks(index):
    ctx = retrieve("expense ratio of HDFC ELSS Tax Saver Fund", index)
    assert len(ctx.scores) == len(ctx.chunks)
    assert ctx.ok is True
    assert ctx.reason == "ok"


def test_top_k_is_respected(index):
    ctx = retrieve("what is the expense ratio of HDFC Large Cap Fund?", index)
    assert len(ctx.chunks) <= config.TOP_K


def test_context_chars_never_exceed_the_cap(index):
    """MAX_CONTEXT_CHARS is a hard stop, not a target."""
    ctx = retrieve("performance of HDFC Large Cap Fund over 5 years", index)
    total = sum(len(c.text) for c in ctx.chunks)
    assert total <= config.MAX_CONTEXT_CHARS, f"{total} chars exceeds the cap"


def test_context_is_falsey_when_insufficient(index):
    ctx = retrieve("asdkjh qwe zxcv", index)
    assert not ctx
    assert ctx.is_insufficient


def test_default_context_is_not_ok():
    """A Context built by accident must not look like a successful retrieval."""
    assert not Context().ok
    assert not Context().chunks


# ---------------------------------------------------------------------- CLI
def test_cli_debug_runs_and_exits_zero_for_a_real_question():
    proc = subprocess.run(
        [sys.executable, "-m", "app.retrieve",
         "expense ratio of HDFC ELSS Tax Saver Fund", "--debug"],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "chunk_id" not in proc.stdout or "score" in proc.stdout
    assert "hdfc-elss" in proc.stdout


def test_cli_exits_nonzero_for_a_foreign_scheme():
    proc = subprocess.run(
        [sys.executable, "-m", "app.retrieve",
         "expense ratio of Parag Parivartan Flexi Cap Fund", "--debug"],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 1, (
        f"a foreign scheme should exit non-zero.\n{proc.stdout}\n{proc.stderr}"
    )
    assert "INSUFFICIENT" in proc.stdout


# ------------------------------------------------- known-absence (§8.2)
# A missing fact still scores ~0.71 against the best chunk, because every HDFC
# factsheet has the same layout. These lock in the term-presence check that
# replaces the similarity floor as the answerability test.


def test_minimum_sip_is_refused_because_the_corpus_lacks_the_figure():
    ctx = retrieve("What is the minimum SIP for HDFC Equity Flexi Cap Fund?")
    assert not ctx.ok
    assert ctx.reason == INSUFFICIENT_FACT_ABSENT
    assert ctx.missing == "sip"


@pytest.mark.parametrize(
    "question",
    [
        "What is the minimum SIP for HDFC Equity Flexi Cap Fund?",
        "What is the SIP of HDFC Large Cap Fund?",
        "Tell me about the SIP for HDFC Small Cap Fund",
    ],
)
def test_every_sip_phrasing_is_refused(question):
    """The question-side trigger is broad; the corpus-side probe stays narrow.

    A bare "sip" question used to slip through, and 24 chunks do contain a
    standalone "SIP" as a NAV field name, so the corpus probe must demand
    value-bearing wording rather than the bare word.
    """
    ctx = retrieve(question)
    assert not ctx.ok, f"{question!r} should be refused, not answered"
    assert ctx.reason == INSUFFICIENT_FACT_ABSENT


@pytest.mark.parametrize(
    "question",
    [
        "expense ratio of HDFC ELSS Tax Saver Fund",
        "exit load HDFC Equity Flexi Cap Fund",
        "benchmark of HDFC Balanced Advantage Fund",
        "lock in period HDFC ELSS Tax Saver Fund",
    ],
)
def test_the_sip_probe_does_not_fire_on_answerable_questions(question):
    ctx = retrieve(question)
    assert ctx.ok, f"{question!r} should be answerable, got {ctx.reason}"
