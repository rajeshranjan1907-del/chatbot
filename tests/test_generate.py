"""Generation guard tests - implementation.md Phase 5 step 3.

The LLM is MOCKED everywhere. These tests are about the guard, not the model:
given an adversarial draft, the post-generation checks must produce a compliant
answer. A test that called Groq would be testing the network, and would fail
for reasons that have nothing to do with the code under test.

The five adversarial cases the plan names: two URLs, five sentences, no footer,
a recommendation, an empty string.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app import config, generate
from app.chunking import Chunk
from app.generate import (
    FOOTER_PREFIX,
    LLMError,
    NO_ANSWER,
    _URL_RE,
    answer_question,
    build_messages,
    call_llm,
    detect_advice_leak,
    enforce,
    enforce_footer,
    enforce_sentence_cap,
    enforce_url,
    fallback_response,
    is_transient,
    insufficient_response,
    is_degenerate,
)
from app.prompts import SYSTEM_PROMPT, build_context_block, resolve_as_of
from app.retrieve import (
    INSUFFICIENT_BELOW_FLOOR,
    INSUFFICIENT_UNKNOWN_SCHEME,
    Context,
)

ELSS_URL = "https://files.hdfcfund.com/s3fs-public/2026-07/HDFC%20MF%20Factsheet%20-%20June%202026.pdf"
LARGE_CAP_URL = "https://files.hdfcfund.com/s3fs-public/2026-07/HDFC%20MF%20Factsheet%20-%20June%202026.pdf"


def make_chunk(**overrides) -> Chunk:
    base = dict(
        chunk_id="2026-09-28.1:hdfc-elss-factsheet-2026-06:0007",
        text="Regular: 1.75% Direct: 1.18% including statutory levies on expenses.",
        embed_text="embed",
        scheme="HDFC ELSS Tax Saver",
        scheme_slug="hdfc-elss",
        amc="HDFC AMC",
        category="equity",
        source_type="factsheet",
        source_title="HDFC MF Factsheet - June 2026",
        source_url=ELSS_URL,
        section="Expense Ratio",
        chunk_index=7,
        as_of_date="2026-06-30",
        fetched_at="2026-09-28",
        corpus_version="2026-09-28.1",
    )
    base.update(overrides)
    return Chunk(**base)


@pytest.fixture
def context() -> Context:
    return Context(
        chunks=[
            make_chunk(),
            make_chunk(
                chunk_id="2026-09-28.1:hdfc-elss-factsheet-2026-06:0008",
                text="Benchmark: NIFTY 500 Index (TRI).",
                section="Benchmark Index",
                source_url=LARGE_CAP_URL,
                as_of_date="2026-06-30",
            ),
        ],
        scores=[0.82, 0.71],
        ok=True,
        reason="ok",
    )


@pytest.fixture
def context_absent() -> Context:
    """What retrieve() returns when the question asks for a fact we lack.

    The chunks exist and are on-topic - that is the whole point. Only the
    specific figure is missing.
    """
    return Context(
        chunks=[make_chunk()],
        scores=[0.71],
        ok=False,
        reason="fact_absent_from_corpus",
        missing="sip",
    )


# ------------------------------------------------------------------- prompts
def test_system_prompt_is_architecture_verbatim():
    """Diff the prompt against architecture.md itself, not against a checklist.

    A phrase checklist only proves the prompt still mentions the rules; it cannot
    catch an edited rule. §5.4 is the contract, so the contract is the fixture.
    """
    arch = (REPO / "architecture.md").read_text(encoding="utf-8")
    match = re.search(r"^SYSTEM\n(.*?)\nUSER$", arch, re.S | re.M)
    assert match, "architecture.md no longer contains a SYSTEM ... USER block"

    expected = match.group(1).rstrip()
    assert SYSTEM_PROMPT.strip() == expected, (
        "SYSTEM_PROMPT has drifted from architecture.md §5.4.\n"
        f"architecture: {expected!r}\n"
        f"prompts.py  : {SYSTEM_PROMPT.strip()!r}"
    )


def test_system_prompt_still_contains_the_hard_rules():
    """Belt and braces: the rules that the post-guard enforces are named."""
    for line in (
        "at most 3 sentences",
        "EXACTLY ONE source URL",
        "Last updated from sources: {as_of_date}",
        "Never give investment advice",
        "Never state, compute, or imply returns, CAGR, or performance",
        "I couldn't find that in my sources.",
    ):
        assert line in SYSTEM_PROMPT, f"missing from the prompt: {line!r}"


def test_resolve_as_of_takes_the_most_recent(context):
    assert resolve_as_of(context.chunks) == "2026-06-30"


def test_resolve_as_of_uses_the_max_date_not_the_last_chunk():
    """Rank order and date order disagree, and the footer must follow the date.

    Regression: a KIM dated 2025-11-21 ranked 5th put its date in the footer for
    an answer taken from the June 2026 factsheet, understating source freshness.
    """
    chunks = [
        make_chunk(as_of_date="2026-06-30", chunk_id="newest"),
        make_chunk(as_of_date="2025-11-21", source_url=LARGE_CAP_URL),
        make_chunk(as_of_date="2024-05-01", source_url=LARGE_CAP_URL),
    ]
    assert resolve_as_of(chunks) == "2026-06-30"
    # Same set, reversed: the answer must not change.
    assert resolve_as_of(list(reversed(chunks))) == "2026-06-30"


def test_resolve_as_of_falls_back_when_absent():
    chunk = make_chunk(as_of_date="")
    assert resolve_as_of([chunk]) == "2026-09-28"


def test_resolve_as_of_with_no_chunks_is_stable():
    assert resolve_as_of([]) == config.CORPUS_VERSION


def test_context_block_carries_the_url_per_chunk(context):
    block = build_context_block(context.chunks)
    assert "[1]" in block and "[2]" in block
    assert ELSS_URL in block
    assert "Expense Ratio" in block


def test_build_messages_formats_the_footer_date(context):
    messages = build_messages("expense ratio?", context)
    assert messages[0]["role"] == "system"
    assert "Last updated from sources: 2026-06-30" in messages[0]["content"]
    assert "{as_of_date}" not in messages[0]["content"], "placeholder left unformatted"
    assert "expense ratio?" in messages[1]["content"]


# ------------------------------------------------------ 1. two URLs in a draft
def test_two_urls_are_trimmed_to_one(context):
    """Two *different* retrieved URLs -> keep the first, drop the second."""
    other = "https://files.hdfcfund.com/s3fs-public/2026-07/HDFC%20MF%20Factsheet%20-%20July%202026.pdf"
    draft = (
        "The regular plan expense ratio is 1.75% and the direct plan is 1.18%. "
        f"Source: {ELSS_URL} and also {other}."
    )
    text, flags = enforce_url(draft, context)
    urls = _URL_RE.findall(text)
    assert len(urls) == 1, f"expected exactly one URL, got {urls}"
    assert "url_dropped" in flags
    assert urls[0] in {ELSS_URL, other}


def test_repeating_the_same_url_is_deduplicated(context):
    """The same URL twice is not two citations; it must collapse to one."""
    draft = f"The ratio is 1.75%. See {ELSS_URL}. Also {ELSS_URL}."
    text, flags = enforce_url(draft, context)
    assert _URL_RE.findall(text) == [ELSS_URL]
    assert "url_dropped" in flags


def test_a_url_the_model_invented_is_dropped(context):
    """A hallucinated URL must not survive, even when a real one is present."""
    draft = f"The ratio is 1.75%. See {ELSS_URL} and https://example.com/fake-fund"
    text, flags = enforce_url(draft, context)
    assert "example.com" not in text, "a fabricated URL was not removed"
    assert ELSS_URL in text
    assert "url_dropped" in flags


def test_no_url_at_all_is_substituted_from_the_top_chunk(context):
    draft = "The regular plan expense ratio is 1.75% and direct is 1.18%."
    text, flags = enforce_url(draft, context)
    assert ELSS_URL in text, "the guard did not add a citation"
    assert "url_substituted" in flags


def test_all_urls_fabricated_falls_back_to_the_real_one(context):
    draft = "The ratio is 1.75%. See https://example.com/a and https://example.com/b"
    text, flags = enforce_url(draft, context)
    assert "example.com" not in text
    assert ELSS_URL in text
    assert "url_substituted" in flags


# ------------------------------------------------- 2. five sentences in a draft
def test_five_sentences_are_capped_at_three(context):
    draft = (
        "Sentence one about the ratio. Sentence two about the direct plan. "
        "Sentence three about the benchmark. Sentence four about the riskometer. "
        "Sentence five about the fund manager."
    )
    text, flags = enforce_sentence_cap(draft, context)
    body = " ".join(ln for ln in text.splitlines() if not _URL_RE.search(ln))
    assert len([s for s in body.split(". ") if s.strip()]) <= 3, body
    assert "sentences_truncated" in flags
    assert "Sentence four" not in text
    assert "Sentence five" not in text


def test_three_sentences_pass_through_untouched(context):
    draft = "One thing. Two things. Three things."
    text, flags = enforce_sentence_cap(draft, context)
    assert "sentences_truncated" not in flags
    assert text.strip() == draft.strip()


def test_sentence_cap_keeps_the_url_line(context):
    draft = "A one. B two. C three. D four."
    text, _ = enforce_sentence_cap(f"{draft}\n{ELSS_URL}", context)
    assert ELSS_URL in text, "the URL was lost while trimming sentences"
    assert "D four" not in text


# ---------------------------------------------------- 3. missing footer in draft
def test_missing_footer_is_appended(context):
    draft = f"The regular plan expense ratio is 1.75% and direct is 1.18%.\n{ELSS_URL}"
    text, flags = enforce_footer(draft, context)
    assert text.strip().endswith(f"{FOOTER_PREFIX} 2026-06-30"), text
    assert "footer_appended" in flags


def test_footer_date_cannot_be_overridden_by_the_model(context):
    """A model-chosen date must lose to the date in the chunk metadata."""
    draft = (
        f"The ratio is 1.75%.\n{ELSS_URL}\n"
        f"{FOOTER_PREFIX} 1999-01-01"
    )
    text, flags = enforce_footer(draft, context)
    assert "1999-01-01" not in text, "the model's own date survived"
    assert text.strip().endswith(f"{FOOTER_PREFIX} 2026-06-30")
    assert "footer_normalised" in flags


def test_footer_is_not_duplicated(context):
    draft = f"Ratio is 1.75%.\n{ELSS_URL}\n{FOOTER_PREFIX} 2026-06-30"
    text, _ = enforce_footer(draft, context)
    assert text.count(FOOTER_PREFIX) == 1, text


# --------------------------------------------------- 4. advice leak in the draft
def test_recommendation_is_detected(context):
    hit, flags = detect_advice_leak("You should buy this fund for long-term growth.")
    assert hit and "advice_leak" in flags


@pytest.mark.parametrize(
    "text",
    [
        "You should invest in the ELSS for a tax benefit.",
        "I recommend buying the direct plan.",
        "This is the best fund for growth.",
        "It is advisable to hold for 3 years.",
    ],
)
def test_all_recommendation_phrasings_are_caught(text, context):
    hit, _ = detect_advice_leak(text)
    assert hit, f"missed advice leak: {text!r}"


def test_advice_leak_never_reaches_the_final_answer(context, monkeypatch):
    monkeypatch.setattr(
        "app.generate.call_llm",
        lambda q, c: "You should buy the ELSS Tax Saver for tax savings.",
    )
    result = answer_question("should I buy the ELSS?", context)
    assert "should buy" not in result["answer"].lower()
    assert result["reason"] in ("advice_leak", "llm_failed")
    assert "fallback" in result["flags"]


def test_factual_wording_is_not_mistaken_for_advice(context):
    """Redemption language is the corpus's own vocabulary, not a recommendation."""
    for text in (
        "An Exit Load of 1.00% is payable if units are redeemed within 1 year.",
        "No Exit Load is payable if units are redeemed after 1 year.",
        "Units may be switched-out after the lock-in period of 3 years.",
    ):
        hit, _ = detect_advice_leak(text)
        assert not hit, f"false advice positive on: {text!r}"


# ------------------------------------------------ 5. empty / degenerate output
def test_empty_string_is_degenerate():
    hit, flags = is_degenerate("")
    assert hit
    assert "empty" in flags or "too_short" in flags


def test_short_string_is_degenerate():
    hit, flags = is_degenerate("Yes.")
    assert hit and "too_short" in flags


def test_truncated_mid_sentence_is_degenerate():
    hit, flags = is_degenerate("The expense ratio of this fund is 1.75 and the dire")
    assert hit and "truncated" in flags


def test_complete_sentence_is_not_degenerate():
    hit, _ = is_degenerate("The expense ratio is 1.75% for the regular plan.")
    assert not hit


def test_empty_llm_output_falls_back(context, monkeypatch):
    monkeypatch.setattr("app.generate.call_llm", lambda q, c: "")
    result = answer_question("expense ratio?", context)
    assert "fallback" in result["flags"] or not result["ok"]


# ----------------------------------------------------------------- the guard
def test_enforce_produces_a_compliant_answer(context):
    """Everything wrong at once, fixed by the guard."""
    draft = (
        "You should buy this fund. "
        "The regular expense ratio is 1.75%. "
        "The direct plan is 1.18%. "
        "The benchmark is NIFTY 500 TRI. "
        "The manager has twenty years experience. "
        f"See {ELSS_URL} and https://example.com/invented"
    )
    text, flags = enforce(draft, context)

    assert _URL_RE.findall(text) == [ELSS_URL], _URL_RE.findall(text)

    # The footer is a required line, not prose, so it is excluded before
    # counting sentences. The rule is "at most 3 sentences" of answer text.
    prose_lines = [
        ln for ln in text.splitlines()
        if not _URL_RE.search(ln) and FOOTER_PREFIX not in ln
    ]
    prose = " ".join(prose_lines)
    assert len(re.findall(r"[.!?](?:\s|$)", prose)) <= 3, prose
    assert "The manager has twenty years" not in text, "the 4th sentence survived"
    assert "The benchmark is NIFTY" not in text, "the 3rd+ sentence was not trimmed"

    assert text.strip().endswith(f"{FOOTER_PREFIX} 2026-06-30")
    assert {"advice_leak", "sentences_truncated", "url_dropped"} & set(flags)


def test_enforce_never_invents_a_url_not_in_context(context):
    text, _ = enforce("Ratio is 1.75% per https://example.com/x", context)
    for url in _URL_RE.findall(text):
        assert url in {ELSS_URL, LARGE_CAP_URL}


# --------------------------------------------------------------- INSUFFICIENT
def test_insufficient_is_deterministic_and_llm_free():
    ctx = Context(ok=False, reason=INSUFFICIENT_BELOW_FLOOR)
    result = insufficient_response(ctx)
    assert result["ok"] is False
    assert result["citation"] is None
    assert NO_ANSWER in result["answer"]


def test_unknown_scheme_explains_the_absence():
    ctx = Context(ok=False, reason=INSUFFICIENT_UNKNOWN_SCHEME)
    result = insufficient_response(ctx)
    assert "different AMC" in result["detail"]


def test_insufficient_paths_never_call_the_llm(monkeypatch):
    def explode(*a, **k):
        raise AssertionError("the LLM must not be called on the INSUFFICIENT path")

    monkeypatch.setattr("app.generate.call_llm", explode)
    for reason in (INSUFFICIENT_BELOW_FLOOR, INSUFFICIENT_UNKNOWN_SCHEME, "no_hits"):
        result = answer_question("anything", Context(ok=False, reason=reason))
        assert result["ok"] is False
        assert result["citation"] is None


# ------------------------------------- the real draft that broke the guard (live)
# Captured verbatim from qwen/qwen3.8-27b via Groq. Both trailing lines are
# mandated by §5.5, and neither ends in punctuation - a date has no full stop.
# The guard judged the whole string, called it "truncated", then counted the
# footer as sentence 1 and truncated the correct answer away, leaving a
# citation with no claim attached to it.
LIVE_DRAFT = (
    "The expense ratio for the HDFC Large Cap Fund Direct plan is 1.03%.\n"
    f"{ELSS_URL}\n"
    "Last updated from sources: 2026-06-30"
)


def test_live_draft_survives_the_guard(context):
    text, flags = enforce(LIVE_DRAFT, context)
    assert "1.03%" in text, "the correct figure was deleted by the guard"
    assert text.splitlines()[0].endswith("1.03%."), (
        f"the answer line must come first, got {text!r}"
    )
    assert "truncated" not in flags, flags


def test_a_date_footed_draft_is_not_called_truncated(context):
    """A bare date legitimately has no terminator."""
    degenerate, flags = is_degenerate(LIVE_DRAFT)
    assert degenerate is False, flags
    assert "truncated" not in flags


def test_a_genuinely_truncated_draft_is_still_caught(context):
    """Stripping the footer must not blind the check to a real cut-off."""
    degenerate, flags = is_degenerate(
        "The expense ratio for the HDFC Large Cap Fund Direct plan is 1"
    )
    assert degenerate is True
    assert "truncated" in flags


def test_the_footer_does_not_consume_a_sentence_of_the_cap(context):
    """One sentence of prose plus a footer is still one sentence."""
    text, flags = enforce_sentence_cap(LIVE_DRAFT, context)
    prose = " ".join(
        ln for ln in text.splitlines()
        if FOOTER_PREFIX not in ln and not _URL_RE.search(ln)
    )
    assert len(re.findall(r"[.!?](?:\s|$)", prose)) <= 3
    assert "1.03%" in prose, prose
    # The footer must still be present, and last.
    assert text.strip().endswith(f"{FOOTER_PREFIX} 2026-06-30")


def test_the_footer_ends_up_last_even_if_the_model_put_it_first(context):
    """Order in the draft must not decide where the footer lands."""
    draft = (
        "Last updated from sources: 1999-01-01\n"
        "The expense ratio for the Direct plan is 1.03%.\n"
        f"{ELSS_URL}"
    )
    text, flags = enforce(draft, context)
    assert text.strip().endswith(f"{FOOTER_PREFIX} 2026-06-30"), text
    assert "1999-01-01" not in text, "a model-chosen date must not survive"
    assert "1.03%" in text


def test_a_url_only_draft_is_not_an_answer(context):
    """A draft with no prose at all is degenerate, not a citation."""
    degenerate, flags = is_degenerate(f"{ELSS_URL}\n{FOOTER_PREFIX} 2026-06-30")
    assert degenerate is True
    assert "empty" in flags


# ------------------------------------------------- model declines (§5.4 rule 6)
# Captured verbatim from qwen/qwen3.8-27b for the capital-gains question. The
# corpus genuinely has no download steps, so declining is correct - but the
# guard used to strip the refusal line and emit a URL and a date with no claim
# behind them, which reads as a sourced answer to a question that has none.
DECLINING_DRAFT = (
    "I couldn't find that in my sources.\n"
    "Last updated from sources: 2025-11-21"
)


def test_a_declining_draft_becomes_a_deterministic_refusal(context, monkeypatch):
    monkeypatch.setattr(generate, "call_llm", lambda q, c: DECLINING_DRAFT)
    result = answer_question("How do I download my capital gains statement?", context)
    assert result["ok"] is False
    assert result["citation"] is None, "a refusal must not carry a citation"
    assert result["reason"] == "model_declined_no_answer"
    assert result["answer"].startswith(NO_ANSWER)
    # FR-3.3 requires the general HDFC link on every refusal.
    assert config.HDFC_MF_LINK in result["detail"]


def test_a_decline_never_leaves_a_citation_with_no_claim(context, monkeypatch):
    monkeypatch.setattr(generate, "call_llm", lambda q, c: DECLINING_DRAFT)
    result = answer_question("q", context)
    body = [
        ln for ln in result["answer"].splitlines()
        if not _URL_RE.search(ln) and FOOTER_PREFIX not in ln
    ]
    assert body, "the refusal sentence itself must survive"
    assert len(_URL_RE.findall(result["answer"])) == 0


def test_an_answer_mentioning_the_refusal_line_is_not_treated_as_a_decline(
    context, monkeypatch
):
    """A real answer that quotes the phrase must still be returned as an answer."""
    draft = (
        "I couldn't find that in my sources for the minimum SIP, but the HDFC ELSS "
        f"Tax Saver Fund lock-in is 3 years.\n{ELSS_URL}\n{FOOTER_PREFIX} 2026-06-30"
    )
    monkeypatch.setattr(generate, "call_llm", lambda q, c: draft)
    result = answer_question("lock-in?", context)
    assert result["ok"] is True, result
    assert "3 years" in result["answer"]
    assert result["citation"] is not None


# --------------------------------------------------------------- insufficient
def test_minimum_sip_names_the_absent_fact(context_absent):
    result = insufficient_response(context_absent)
    assert result["ok"] is False
    assert result["citation"] is None
    assert result["reason"] == "fact_absent_from_corpus"
    assert "minimum SIP" in result["answer"]
    # The chunks WERE retrieved, so "couldn't find anything close enough" would
    # be false here.
    assert "close enough" not in result["answer"]
    # The canned no-answer line and the headline must not both say "my sources",
    # which produced a stutter before.
    assert result["answer"].count("my sources") <= 1
    assert result["answer"].startswith(NO_ANSWER)


def test_no_llm_is_called_for_an_insufficient_answer(context_absent, monkeypatch):
    def boom(*a, **k):  # pragma: no cover
        raise AssertionError("the insufficient path must not call the LLM")

    monkeypatch.setattr(generate, "call_llm", boom)
    result = answer_question("What is the minimum SIP for HDFC Equity Flexi Cap Fund?")
    assert result["ok"] is False
    assert result["reason"] == "fact_absent_from_corpus"
    assert result["citation"] is None


# ------------------------------------------------------------------ fallback
def test_fallback_uses_the_top_chunk(context):
    result = fallback_response(context)
    assert result["ok"] is True
    assert result["citation"]["chunk_id"] == context.chunks[0].chunk_id
    assert result["citation"]["url"] == ELSS_URL
    assert FOOTER_PREFIX in result["answer"]


def test_fallback_with_no_chunks_is_insufficient():
    result = fallback_response(Context(ok=False, reason="no_hits"))
    assert result["ok"] is False
    assert result["citation"] is None


def test_llm_failure_returns_the_fallback_not_an_exception(context, monkeypatch):
    def boom(q, c):
        raise LLMError("rate limited")

    monkeypatch.setattr("app.generate.call_llm", boom)
    result = answer_question("expense ratio?", context)
    assert result["ok"] is True
    assert "fallback" in result["flags"]
    assert result["citation"]["url"] == ELSS_URL


# ------------------------------------------------------------------- secrets
def test_api_key_never_appears_in_error_messages(monkeypatch):
    """A traceback out of call_llm must not carry the credential."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk_supersecretkey123")

    class Boom:
        def __init__(self, *a, **k):
            pass

        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("401 unauthorized for key gsk_supersecretkey123")

    import sys as _sys
    fake = type(_sys)("groq")
    fake.Groq = Boom
    monkeypatch.setitem(_sys.modules, "groq", fake)

    with pytest.raises(LLMError) as exc:
        call_llm("q", Context(chunks=[make_chunk()], ok=True))
    assert "gsk_supersecretkey123" not in str(exc.value), str(exc.value)


def test_call_llm_without_a_key_raises_a_clear_error(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    with pytest.raises(LLMError) as exc:
        call_llm("q", Context(chunks=[make_chunk()], ok=True))
    assert "GROQ_API_KEY" in str(exc.value)


# ------------------------------------------------- retry policy (§5.5)
class _Status(Exception):
    def __init__(self, code: int):
        super().__init__(f"http {code}")
        self.status_code = code


def _named(name: str) -> type[BaseException]:
    return type(name, (Exception,), {})


@pytest.mark.parametrize(
    "exc,expected",
    [
        (_Status(429), True),   # rate limited
        (_Status(500), True),   # server error
        (_Status(503), True),
        (_Status(401), False),  # bad key: a retry cannot help
        (_Status(403), False),
        (_Status(400), False),  # malformed request: a retry cannot help
        (_Status(404), False),
        (_named("RateLimitError")(), True),
        (_named("APITimeoutError")(), True),
        (_named("InternalServerError")(), True),
        (_named("AuthenticationError")(), False),
        (_named("BadRequestError")(), False),
        (ValueError("something local went wrong"), False),
    ],
)
def test_is_transient_classifies_correctly(exc, expected):
    assert is_transient(exc) is expected


def test_is_transient_walks_the_cause_chain():
    """The clients wrap the real failure, so the status is one level down."""
    inner = _Status(500)
    outer = RuntimeError("request failed")
    outer.__cause__ = inner
    assert is_transient(outer) is True

    denied = RuntimeError("request failed")
    denied.__cause__ = _Status(401)
    assert is_transient(denied) is False


def test_is_transient_survives_a_cyclic_cause_chain():
    """Self-referential __context__ must not hang the classifier."""
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__context__ = b
    b.__context__ = a
    assert is_transient(a) is False


def test_a_permanent_error_is_not_retried(monkeypatch):
    """A 401 must cost one call, not two, and must not sleep."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk_key")
    slept = []
    monkeypatch.setattr(generate.time, "sleep", lambda s: slept.append(s))

    calls = []

    class Boom:
        def __init__(self, *a, **k):
            pass

        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):
                    calls.append(1)
                    raise _Status(401)

    import sys as _sys
    fake = type(_sys)("groq")
    fake.Groq = Boom
    monkeypatch.setitem(_sys.modules, "groq", fake)

    with pytest.raises(LLMError):
        call_llm("q", Context(chunks=[make_chunk()], ok=True))
    assert len(calls) == 1, "a permanent error should not be retried"
    assert slept == [], "a permanent error should not sleep before giving up"


def test_a_transient_error_is_retried_exactly_once(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk_key")
    monkeypatch.setattr(generate.time, "sleep", lambda s: None)

    calls = []

    class Flaky:
        def __init__(self, *a, **k):
            pass

        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):
                    calls.append(1)
                    raise _Status(429)

    import sys as _sys
    fake = type(_sys)("groq")
    fake.Groq = Flaky
    monkeypatch.setitem(_sys.modules, "groq", fake)

    with pytest.raises(LLMError) as exc:
        call_llm("q", Context(chunks=[make_chunk()], ok=True))
    assert len(calls) == 2, "a transient error gets exactly one retry"
    assert "after 2 attempts" in str(exc.value)


# ---------------------------------------------------------------- shape (§5.6)
def test_response_shape(context, monkeypatch):
    monkeypatch.setattr(
        "app.generate.call_llm",
        lambda q, c: f"The regular plan expense ratio is 1.75%.\n{ELSS_URL}",
    )
    result = answer_question("expense ratio?", context)
    for key in ("answer", "citation", "as_of", "ok", "reason", "flags"):
        assert key in result, f"missing key {key!r} from the §5.6 shape"
    assert set(result["citation"]) >= {"label", "url"}


def test_citation_is_structured_not_parsed_from_prose(context, monkeypatch):
    """The citation object comes from chunk metadata even if the model cites nothing."""
    monkeypatch.setattr(
        "app.generate.call_llm",
        lambda q, c: "The expense ratio is 1.75% for the regular plan and 1.18% direct.",
    )
    result = answer_question("expense ratio?", context)
    assert result["citation"]["url"] == ELSS_URL
    assert result["citation"]["chunk_id"] == context.chunks[0].chunk_id
