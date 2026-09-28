"""Generation + citation - architecture.md §5.4, §5.5, §5.6.

    python -m app.generate "What is the expense ratio of HDFC Large Cap Fund Direct Growth?"

The post-generation guard is five small pure functions, each returning
(text, flags). Splitting them matters: the demo has to be able to say *which*
check fired, and a single 80-line `enforce()` could not. The order is
degenerate -> advice -> URL -> sentences -> footer, so a response that is going
to be thrown away is not also silently rewritten.

The citation is a structured field read from chunk metadata, never parsed out
of prose. A model that invents a URL is a bug we can catch; a model that
invents a *citation object* is not, so the model never gets to write one.

`call_llm` never logs the key and never puts it in an exception message: the
Groq client is constructed here and the key is read from config, so a
traceback out of this module carries no credential.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from typing import Iterable

from app import config
from app.chunking import Chunk
from app.prompts import SYSTEM_PROMPT, build_context_block, resolve_as_of
from app.retrieve import Context, retrieve

FOOTER_PREFIX = "Last updated from sources:"

#: A URL, tolerant of the punctuation a model adds around one.
_URL_RE = re.compile(r"https?://[^\s<>\"'\)\]]+[^\s<>\"'\)\].,;:!?]")

#: Recommendation verbs. architecture.md §5.5 calls these out explicitly; this
#: list is deliberately closed so a false positive cannot silently refuse a
#: factual question.
_ADVICE_RE = re.compile(
    r"\b(you should|we recommend|i recommend|recommend(?:ed|s|ing)?\s+(?:buying|investing|"
    r"switching|holding)|buy|sell|hold|invest in|advisable|suitable for you|"
    r"best (?:fund|scheme|option|choice)|worth (?:buying|investing)|ideal (?:fund|scheme)|"
    r"opt (?:for|into)|prefer)\b",
    re.IGNORECASE,
)

#: Advice words that also occur inside legitimate factual answers about a fund's
#: own characteristics. These are excluded from the advice scan, because
#: "hold" and "sell" appear in "units redeemed or switched-out" style text and
#: in the KIM's own vocabulary.
_ADVICE_ALLOW = (
    "units are redeemed",
    "redeemed or switched",
    "switched-out",
    "switched-out after",
)

NO_ANSWER = "I couldn't find that in my sources."


class LLMError(RuntimeError):
    """Raised when the LLM could not be used. Never carries the API key."""


# --------------------------------------------------------------------- prompt
def build_messages(question: str, context: Context) -> list[dict]:
    as_of = resolve_as_of(context.chunks)
    user = (
        f"QUESTION\n{question}\n\n"
        f"CONTEXT\n{build_context_block(context.chunks)}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(as_of_date=as_of)},
        {"role": "user", "content": user},
    ]


# ------------------------------------------------------------------- LLM call
#: groq exception class names worth one more attempt. Matching on the name keeps
#: `groq` an optional dependency: importing it at module scope would make the
#: whole module unimportable in the fallback-only environment, which is exactly
#: where the tests run.
_TRANSIENT_NAMES = frozenset({
    "RateLimitError",
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",
    "Timeout",
    "ConnectionError",
})


def is_transient(exc: BaseException) -> bool:
    """True if a retry could plausibly succeed. §5.5: rate limit, 5xx, timeout.

    A 401 or a 400 will fail identically on the second attempt, so retrying only
    doubles the latency of a request that is never going to work. Classification
    is by status code first, then by class name, then by the nested `__cause__`
    chain that the OpenAI-compatible clients wrap connection errors in.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))

        status = getattr(current, "status_code", None)
        if isinstance(status, int):
            if status == 429 or status >= 500:
                return True
            if 400 <= status < 500:
                return False

        if type(current).__name__ in _TRANSIENT_NAMES:
            return True

        current = current.__cause__ or current.__context__

    return False


def call_llm(question: str, context: Context) -> str:
    """Groq, temperature 0, max_tokens 220, one retry on transient failure."""
    if not config.GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY is not set")

    try:
        from groq import Groq
    except ImportError as exc:
        raise LLMError("the groq package is not installed") from exc

    client = Groq(api_key=config.GROQ_API_KEY)
    messages = build_messages(question, context)

    last: Exception | None = None
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=messages,
                temperature=config.GROQ_TEMPERATURE,
                max_tokens=220,
            )
            text = response.choices[0].message.content
            if not text or not text.strip():
                raise LLMError("the model returned an empty completion")
            return text.strip()
        except LLMError:
            raise
        except Exception as exc:
            # str(exc) on a client error can echo request headers, which is
            # where the key would be. Log the type, not the payload.
            last = exc
            if not is_transient(exc):
                raise LLMError(
                    f"LLM call failed (not retryable): {type(exc).__name__}"
                ) from None
            if attempt == 0:
                time.sleep(1.0)
    raise LLMError(
        f"LLM call failed after 2 attempts: {type(last).__name__}"
    ) from None


# ------------------------------------------------------------------ the guard
def prose_only(text: str) -> str:
    """The answer text with the URL line and the footer removed.

    The guard has to reason about the *answer*, and the two mandated trailing
    lines are not part of it. Measured cause of a deleted correct answer: the
    draft ended "...is 1.03%.\nhttps://...\nLast updated from sources: 2026-06-30",
    so the truncation check saw a date with no terminal punctuation and flagged
    it, and the sentence cap then counted the footer as sentence 1 and truncated
    the real answer away. A date legitimately has no full stop.
    """
    kept = []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        if _URL_RE.search(line):
            continue
        if FOOTER_PREFIX in line:
            continue
        kept.append(line.strip())
    return " ".join(kept).strip()


def is_degenerate(text: str) -> tuple[bool, list[str]]:
    """Too short, or cut off mid-token.

    Judged on the prose only - see prose_only for why the footer is excluded.
    """
    flags: list[str] = []
    stripped = (text or "").strip()
    prose = prose_only(text)
    if len(stripped) < 15:
        flags.append("too_short")
    # The prose must end on a terminator. A bare date does not, and that is fine.
    if prose and not re.search(r"[.!?)\]\"']\s*$", prose):
        flags.append("truncated")
    if not prose:
        flags.append("empty")
    return bool(flags), flags


def detect_advice_leak(text: str) -> tuple[bool, list[str]]:
    """Recommendation or opinion language - architecture.md §5.5."""
    lowered = text.lower()
    for allow in _ADVICE_ALLOW:
        lowered = lowered.replace(allow, "")
    hit = _ADVICE_RE.search(lowered)
    return (bool(hit), ["advice_leak"]) if hit else (False, [])


def enforce_url(text: str, context: Context) -> tuple[str, list[str]]:
    """Exactly one URL, and it must be one the retrieved chunks declare."""
    flags: list[str] = []
    allowed = {c.source_url for c in context.chunks if c.source_url}
    found = _URL_RE.findall(text or "")

    if not found:
        if context.chunks:
            text = (text or "").rstrip()
            return f"{text}\n{context.chunks[0].source_url}", flags + ["url_substituted"]
        return text, flags + ["url_absent"]

    # Keep only URLs that are character-for-character in the retrieved set.
    kept: list[str] = []
    for url in found:
        if url in allowed and url not in kept:
            kept.append(url)
        else:
            flags.append("url_dropped")

    if not kept:
        if context.chunks:
            kept = [context.chunks[0].source_url]
            flags.append("url_substituted")
        else:
            return _URL_RE.sub("", text).strip(), flags + ["url_absent"]

    if len(kept) > 1:
        flags.append("url_trimmed")

    body = _URL_RE.sub("", text).strip()
    return f"{body}\n{kept[0]}".strip(), flags


def enforce_sentence_cap(text: str, context: Context) -> tuple[str, list[str]]:
    """At most 3 sentences, counted on the prose.

    Neither the URL line nor the footer counts toward the cap. The footer is not
    prose - counting it deleted a correct one-sentence answer outright.
    """
    flags: list[str] = []
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    body_lines, trailer = [], []
    for line in lines:
        if _URL_RE.search(line) or FOOTER_PREFIX in line:
            trailer.append(line.strip())
        else:
            body_lines.append(line.strip())

    body = " ".join(body_lines).strip()
    if not body:
        return "\n".join(lines).strip(), flags

    # Split on a terminator followed by whitespace and a capital/digit/bullet.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\u2022])", body)
    if len(parts) > 3:
        flags.append("sentences_truncated")
        body = " ".join(parts[:3])

    # The footer is re-appended last, after the URL, so it is always the final
    # line regardless of the order the model emitted them in.
    url_lines = [ln for ln in trailer if _URL_RE.search(ln)]
    footers = [ln for ln in trailer if FOOTER_PREFIX in ln]
    out = body
    for line in url_lines + footers:
        out += "\n" + line
    return out, flags


def _is_bare_refusal(line: str) -> bool:
    """True if the line is the §5.4 rule-6 refusal and nothing else."""
    normalized = " ".join((line or "").split()).strip().rstrip(".").lower()
    return normalized == NO_ANSWER.rstrip(".").lower()


def enforce_footer(text: str, context: Context) -> tuple[str, list[str]]:
    """The footer is mandatory and must be last."""
    flags: list[str] = []
    as_of = resolve_as_of(context.chunks)
    footer = f"{FOOTER_PREFIX} {as_of}"

    body = text or ""
    if FOOTER_PREFIX in body:
        flags.append("footer_normalised")
        # Drop the existing footer line(s), then re-append the canonical one, so
        # a model-chosen date cannot disagree with the retrieved metadata.
        #
        # Only a line that is *nothing but* the bare refusal is removed. Anchoring
        # on the whole line matters: a real answer may legitimately contain the
        # phrase ("I couldn't find that in my sources for the SIP, but the
        # lock-in is 3 years"), and matching a prefix deleted the entire claim.
        # The refusal is handled upstream in answer_question via _declined().
        kept = [
            ln for ln in body.splitlines()
            if FOOTER_PREFIX not in ln
            and not _is_bare_refusal(ln)
        ]
        body = "\n".join(kept).strip()
    else:
        flags.append("footer_appended")

    if body:
        return f"{body}\n{footer}", flags
    return f"{NO_ANSWER}\n{footer}", flags


# ------------------------------------------------------------------- fallback
def _sentences(text: str, limit: int) -> str:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\u2022])", text.strip())
    return " ".join(parts[:limit])


def fallback_response(context: Context, reason: str = "llm_unavailable") -> dict:
    """Retrieval-only answer: top chunk's first 2 sentences plus its citation."""
    if not context.chunks:
        return {
            "answer": NO_ANSWER,
            "citation": None,
            "as_of": config.CORPUS_VERSION,
            "ok": False,
            "reason": reason,
            "flags": [],
        }

    top = context.chunks[0]
    as_of = resolve_as_of(context.chunks)
    body = _sentences(top.text, 2)
    return {
        "answer": f"{body}\n{top.source_url}\n{FOOTER_PREFIX} {as_of}",
        "citation": {
            "label": f"{top.amc} - {top.source_title}"
            + (f" ({top.as_of_date})" if top.as_of_date else ""),
            "url": top.source_url,
            "chunk_id": top.chunk_id,
        },
        "as_of": as_of,
        "ok": True,
        "reason": reason,
        "flags": ["fallback"],
    }


def insufficient_response(context: Context) -> dict:
    """Deterministic refusal. No LLM call - the model must not shape this."""
    messages = {
        "below_similarity_floor": (
            "I couldn't find anything close enough in my sources to answer that.",
            "No chunk in the corpus scored high enough similarity to your question.",
        ),
        "no_hits": (
            "I couldn't find anything in my sources for that question.",
            "The search returned no matching content at all.",
        ),
        "scheme_not_in_corpus": (
            "I don't have documents for that scheme.",
            "I hold HDFC AMC documents only, and none of them are for the scheme named.",
        ),
        "unknown_scheme": (
            "I only have documents for HDFC AMC's mutual fund schemes.",
            "The fund named in your question belongs to a different AMC, so I have no "
            "official source for it and won't answer from memory.",
        ),
        "model_declined_no_answer": (
            "Those documents don't cover that.",
            "The model read the retrieved chunks and found no answer to your question, "
            "so it declined rather than guessing. The closest source is "
            f"{config.HDFC_MF_LINK}",
        ),
    }
    headline, detail = messages.get(
        context.reason, ("I couldn't find that in my sources.", "")
    )

    # The fact-absent case needs its own wording: the chunks are on-topic and were
    # returned, so "I couldn't find anything close enough" would be false, and
    # restating "my sources" after NO_ANSWER reads as a stutter. The honest
    # statement is that this specific figure is absent from the documents.
    if context.reason == "fact_absent_from_corpus":
        label = {"sip": "minimum SIP"}.get(context.missing, context.missing or "that detail")
        headline = (
            f"The {label} for this scheme isn't stated in those documents, "
            "so I can't give you the figure."
        )
        detail = (
            "The retrieved chunks are about this scheme but contain no SIP amount. "
            "HDFC's factsheets list 'MINIMUM LUMP SUM INVESTMENT' but not a minimum "
            "SIP instalment, so the number has to come from the AMC or the fund page."
        )

    return {
        "answer": f"{NO_ANSWER} {headline}",
        "citation": None,
        "as_of": config.CORPUS_VERSION,
        "ok": False,
        "reason": context.reason,
        "flags": ["insufficient"],
        "detail": detail,
    }


# ------------------------------------------------------------------ top level
def enforce(draft: str, context: Context) -> tuple[str, list[str]]:
    """Run all five checks in order, collecting flags.

    Order matters and is not the order they are listed in the plan. Measured
    cause: running the sentence cap before the URL check counted a URL as a
    sentence, so "See https://..." was trimmed as prose and the answer ended
    "See  and Last updated from sources: ...". The URL is removed first, then
    the remaining prose is capped, then the URL is re-appended.
    """
    flags: list[str] = []
    text = draft or ""

    degenerate, f = is_degenerate(text)
    flags.extend(f)
    if degenerate and not text.strip():
        return text, flags

    advice, f = detect_advice_leak(text)
    flags.extend(f)

    # 1. Pull the URL out and validate it. After this, `text` is prose-only plus
    #    a trailing URL line, and enforce_sentence_cap will not eat the URL.
    text, f = enforce_url(text, context)
    flags.extend(f)

    # 2. Cap the prose at three sentences.
    text, f = enforce_sentence_cap(text, context)
    flags.extend(f)

    # 3. The footer is mandatory and goes last.
    text, f = enforce_footer(text, context)
    flags.extend(f)

    return text, flags


def _declined(draft: str) -> bool:
    """True if the model used the §5.4 rule-6 refusal verbatim.

    Matched loosely on purpose: models paraphrase the instructed line, and the
    check only has to separate "declined" from "answered". A draft that carries
    any real prose alongside the phrase is not a refusal.
    """
    prose = prose_only(draft)
    if not prose:
        return False
    return NO_ANSWER.lower() in prose.lower() and len(prose) < len(NO_ANSWER) + 40


def answer_question(question: str, context: Context | None = None) -> dict:
    """architecture.md §5.6 response shape."""
    if context is None:
        context = retrieve(question)

    if not context.ok:
        return insufficient_response(context)

    try:
        draft = call_llm(question, context)
    except LLMError as exc:
        result = fallback_response(context, reason="llm_failed")
        result["detail"] = str(exc)
        return result

    text, flags = enforce(draft, context)

    degenerate, _ = is_degenerate(text)
    if degenerate and len(text.strip()) < 15:
        return fallback_response(context, reason="degenerate_output")

    if "advice_leak" in flags:
        return fallback_response(context, reason="advice_leak")

    # The model declining is a real outcome, not a formatting error. §5.4 rule 6
    # tells it to reply exactly "I couldn't find that in my sources." when the
    # context has no answer, and the corpus genuinely lacks some things (the
    # capital-gains download steps, for one). Measured cause of a silent
    # failure: enforce_footer strips that line as footer-adjacent, leaving a URL
    # and a date attached to no claim at all - a citation with nothing behind it.
    # Route it to the deterministic refusal so it is honest about what is missing
    # and names the general HDFC MF link, per §5.3.
    if _declined(draft):
        return insufficient_response(
            Context(
                chunks=context.chunks,
                scores=context.scores,
                ok=False,
                reason="model_declined_no_answer",
            )
        )

    # The citation comes from chunk metadata, never from the model's prose.
    top = context.chunks[0]
    return {
        "answer": text,
        "citation": {
            "label": f"{top.amc} - {top.source_title}"
            + (f" ({top.as_of_date})" if top.as_of_date else ""),
            "url": top.source_url,
            "chunk_id": top.chunk_id,
        },
        "as_of": resolve_as_of(context.chunks),
        "ok": True,
        "reason": "ok",
        "flags": flags,
    }


# ----------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Answer a question from the corpus.")
    ap.add_argument("question", nargs="+", help="the question to answer")
    ap.add_argument("--json", action="store_true", help="print the raw response dict")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover
                pass

    question = " ".join(args.question)
    result = answer_question(question)

    if args.json:
        import json

        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["ok"] else 1

    print("=" * 100)
    print(f"Q: {question}")
    print("=" * 100)
    print(result["answer"])
    print()
    if result.get("citation"):
        print(f"citation : {result['citation']['label']}")
        print(f"  url    : {result['citation']['url']}")
    else:
        print("citation : none")
    print(f"as_of    : {result['as_of']}")
    print(f"reason   : {result['reason']}")
    if result.get("flags"):
        print(f"flags    : {', '.join(result['flags'])}")
    if result.get("detail"):
        print(f"detail   : {result['detail']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
