"""Retrieval - architecture.md §5.3, steps 1-7 in order.

    python -m app.retrieve "expense ratio of HDFC ELSS Tax Saver Fund" --debug

The seven steps, and nothing more (Phase 4 has no LLM in it):

    1. embed the question, L2-normalised, with the same model ingest used
    2. over-fetch FETCH_K hits from the store
    3. no hits at all            -> INSUFFICIENT
    4. drop hits below SIM_FLOOR (cosine similarity)
    5. if the question names a scheme, keep ONLY that scheme
    6. an empty set after 4 or 5 -> INSUFFICIENT, never a fallback to another scheme
    7. lexical re-rank, then take TOP_K

Step 5 is the one that carries the most weight. "Expense ratio of HDFC Small
Cap" must never be answered with the Large Cap factsheet just because that
chunk scored higher, so when the filter empties the set we say so rather than
quietly substituting a different scheme's numbers. A confident wrong number is
worse than an admission.

Step 7 is deliberately simple: term overlap plus explicit boosts, no second
model. The store returns `distance` (cosine distance) and we convert once to
`score` (cosine similarity) at the boundary, so SIM_FLOOR is compared against a
similarity everywhere in this module and never against a distance by accident.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from app import config
from app.chunking import Chunk
from app.embedder import embed_query
from app.localindex import IndexError_, LocalIndex, ensure_index

RULE = "=" * 100
THIN = "-" * 100

#: Words that carry no retrieval signal. Dropping them stops "what is the" from
#: contributing overlap to every chunk equally.
_STOPWORDS = frozenset(
    """a an the of in on at to for from by is are was were be been being and or
    do does did what which who whom how why when where can could should would
    will shall may might must please tell me my i you your it its this that
    these those there here about into over under with without if then than
    fund funds scheme schemes hdfc mutual""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+")

#: A number carrying a unit - the mark of a chunk that states a value rather
#: than describing one.
_NUMERIC = re.compile(
    r"(?:\d[\d,]*\.?\d*\s*(?:%|per cent|bps))"
    r"|(?:Rs\.?\s*\.?\s*[\d,]+)"
    r"|(?:₹\s*[\d,]+)"
    r"|(?:[\d,]+\s*Cr)"
    r"|(?:\d+\s*(?:years?|months?|days?))",
    re.IGNORECASE,
)

#: Which source_type most likely holds the answer, per architecture.md §5.3.
#: Deliberately keyed on the *question's* intent, not the chunk's.
_SOURCE_HINT = {
    "expense ratio": "factsheet",
    "exit load": "factsheet",
    "lock-in": "factsheet",
    "lock in": "factsheet",
    "benchmark": "factsheet",
    "riskometer": "riskometer",
    "download": "guide",
    "statement": "guide",
    "cas": "guide",
    "how do i": "guide",
    "how to": "guide",
    "redemption": "guide",
    "capital gains": "guide",
    "nav": "factsheet",
    "aum": "factsheet",
    "inception": "factsheet",
    "sip": "kim",
    "minimum": "kim",
}

#: Why retrieval returned nothing, for the INSUFFICIENT path. Phase 5 renders
#: this; keeping it machine-readable means the message cannot drift from the
#: rule that produced it.
INSUFFICIENT_NO_HITS = "no_hits"
INSUFFICIENT_BELOW_FLOOR = "below_similarity_floor"
INSUFFICIENT_SCHEME_MISMATCH = "scheme_not_in_corpus"
INSUFFICIENT_UNKNOWN_SCHEME = "unknown_scheme"
#: The question asked for something the corpus does not carry. Distinguished from
#: the reasons above because the retrieval *worked* - the chunks are on-topic, they
#: just do not contain the answer. See MISSING_FACT_PROBES.
INSUFFICIENT_FACT_ABSENT = "fact_absent_from_corpus"

#: Facts the corpus does not carry. §8.2 measured that a missing fact still
#: scores ~0.71 here, far above SIM_FLOOR, because all five factsheets share the
#: same structure. So the floor cannot be the answerability test: the presence of
#: the *specific* thing asked about has to be checked directly in the retrieved
#: text.
#:
#: Two lists per fact, because they must not be the same list:
#:
#:   "question" - what to look for in the *user's* wording. Broad, so "What is
#:     the SIP of HDFC Large Cap?" is caught, not just "minimum SIP".
#:   "corpus"   - what would have to appear in the *retrieved chunks* for us to
#:     believe the figure exists. Deliberately narrow and value-bearing. 24
#:     chunks contain a bare "SIP" (NAV/plan field names, "SIPSTART"), so a bare
#:     \bsip\b probe would match noise and let the question through anyway.
MISSING_FACT_PROBES: dict[str, dict[str, tuple[str, ...]]] = {
    "sip": {
        "question": ("sip",),
        "corpus": (
            "minimum sip",
            "sip instalment",
            "sip amount",
            "sip of rs",
            "monthly sip",
            "sip - rs",
        ),
    },
}


@dataclass
class Context:
    """What retrieval hands to the generator - architecture.md §5.3."""

    chunks: list[Chunk] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    ok: bool = False
    reason: str = ""
    #: Which fact was asked for but not found, when reason is
    #: INSUFFICIENT_FACT_ABSENT. Named so the generator can say what it does not
    #: know instead of guessing which fact was meant.
    missing: str = ""

    def __bool__(self) -> bool:
        return self.ok

    @property
    def is_insufficient(self) -> bool:
        return not self.ok


# --------------------------------------------------------------------- helpers
def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


def detect_scheme(question: str) -> tuple[str | None, str | None]:
    """Return (slug, matched_alias).

    Longest alias first, so "hdfc large cap fund" wins over "large cap" and the
    user cannot land on the wrong scheme by naming it two ways at once.
    """
    q = question.lower()
    best: tuple[str | None, str | None] = (None, None)
    best_len = 0
    for alias, slug in config.SCHEME_ALIASES.items():
        if alias in q and len(alias) > best_len:
            best, best_len = (slug, alias), len(alias)
    return best


def detect_foreign_scheme(question: str) -> str | None:
    """Name of a competing AMC mentioned in the question, if any."""
    q = question.lower()
    for hint in config.FOREIGN_SCHEME_HINTS:
        if hint in q:
            return hint
    return None


def _content_tokens(question: str) -> set[str]:
    """Question terms with the scheme name removed.

    Measured cause: for "expense ratio of HDFC ELSS Tax Saver Fund" the tokens
    were {expense, ratio, elss, tax, saver}. The last three are scheme identity,
    already handled by the step-5 filter and the scheme boost below. Counting
    them a second time as content let a chunk that merely parrots the fund name
    ("HDFC ELSS Tax Saver An openended equity linked savings scheme...") outrank
    the chunk actually holding "Regular: 1.75% Direct: 1.18%". So overlap is
    measured on the part of the question that is actually asking for something.
    """
    tokens = set(_tokens(question))
    slug, alias = detect_scheme(question)
    if alias:
        tokens -= set(_tokens(alias))
    if slug:
        tokens -= set(_tokens(config.SCHEMES.get(slug, "")))
    return tokens


def lexical_score(question: str, chunk: Chunk) -> float:
    """Lexical re-rank - architecture.md §5.3, "Re-rank (FR-3.4, Should)".

    Term overlap sets the base; the boosts encode what we know about where each
    kind of fact actually lives. Kept as an explicit, inspectable sum rather than
    a learned model because the demo has to be able to explain its ordering.
    """
    q_tokens = _content_tokens(question)
    if not q_tokens:
        # The question was only a scheme name, so there is no content term to
        # rank on. Fall back to the whole question rather than scoring zero and
        # handing the order back to cosine alone.
        q_tokens = set(_tokens(question))
    if not q_tokens:
        return 0.0

    haystack = f"{chunk.text} {chunk.section}".lower()
    c_tokens = set(_tokens(haystack))
    overlap = q_tokens & c_tokens
    if not overlap:
        return 0.0

    # Normalise by query length so a long question is not automatically favoured.
    score = len(overlap) / len(q_tokens)

    q_lower = question.lower()

    # A scheme-name hit: the chunk is about the fund that was asked about.
    if chunk.scheme.lower() in q_lower:
        score += 0.30

    # A metric term: the question is asking for a named metric, and the chunk
    # contains that same metric string.
    for term in config.METRIC_TERMS:
        if term in q_lower and term in haystack:
            score += 0.20
            break

    # Source-type affinity: "how do I download" wants a guide, "SIP" wants a KIM.
    for phrase, wanted in _SOURCE_HINT.items():
        if phrase in q_lower and chunk.source_type == wanted:
            score += 0.15
            break

    # A metric question is answered by a chunk that carries a value. Measured
    # cause: after removing the scheme name from overlap, the boilerplate chunk
    # and the real answer both matched "expense ratio" (it is in both sections)
    # and tied at 0.50. The tie-breaker has to be the number itself - "Regular:
    # 1.75% Direct: 1.18%" answers the question, "please refer our website"
    # does not.
    if any(term in q_lower for term in config.METRIC_TERMS) and _NUMERIC.search(chunk.text):
        score += 0.25

    return score


def missing_fact(question: str) -> str | None:
    """Name of a fact the question asks for that the corpus does not carry.

    Returns the fact name when the question asks for it, or None. Whether the
    corpus actually has it is a separate check, because a bare word match in a
    fact sheet is not evidence a figure exists.
    """
    q = question.lower()
    for fact, spec in MISSING_FACT_PROBES.items():
        if any(probe in q for probe in spec["question"]):
            return fact
    return None


def corpus_has_fact(fact: str, chunks: list[Chunk]) -> bool:
    """True if the retrieved chunks state this fact in value-bearing wording."""
    joined = " ".join(c.text for c in chunks).lower()
    return any(p in joined for p in MISSING_FACT_PROBES[fact]["corpus"])


def rerank(question: str, hits: list[dict]) -> list[dict]:
    """Order hits by lexical score, breaking ties on cosine similarity."""
    for hit in hits:
        hit["lexical"] = lexical_score(question, hit["chunk_obj"])
    return sorted(hits, key=lambda h: (h["lexical"], h["score"]), reverse=True)


def _as_chunk(record: dict) -> Chunk:
    """Rebuild a Chunk from a stored record."""
    fields = {f for f in Chunk.__dataclass_fields__}
    return Chunk(**{k: v for k, v in record.items() if k in fields})


def _truncate(chunks: list[Chunk], scores: list[float], limit: int) -> tuple[list[Chunk], list[float]]:
    """Hard stop at MAX_CONTEXT_CHARS. Highest ranked chunks are kept first."""
    kept_c: list[Chunk] = []
    kept_s: list[float] = []
    total = 0
    for chunk, score in zip(chunks, scores):
        size = len(chunk.text)
        if total + size > limit:
            break
        kept_c.append(chunk)
        kept_s.append(score)
        total += size
    return kept_c, kept_s


# -------------------------------------------------------------------- retrieve
def retrieve(question: str, index: LocalIndex | None = None) -> Context:
    """Run architecture.md §5.3 steps 1-7 and return a Context."""
    if not question or not question.strip():
        return Context(ok=False, reason=INSUFFICIENT_NO_HITS)

    index = index or ensure_index()

    # Step 1: embed the question exactly as ingest embedded the chunks.
    q_vec = embed_query(question)

    # Step 2: over-fetch, so the re-rank has something to work with.
    hits = index.query(q_vec, n=config.FETCH_K)

    # Step 3: nothing at all.
    if not hits:
        return Context(ok=False, reason=INSUFFICIENT_NO_HITS)

    for hit in hits:
        hit["chunk_obj"] = _as_chunk(hit["chunk"])

    # Step 4: cosine floor. Applied to similarity, not distance.
    kept = [h for h in hits if h["score"] >= config.SIM_FLOOR]
    if not kept:
        return Context(ok=False, reason=INSUFFICIENT_BELOW_FLOOR)

    # Step 5: scheme filter. A named scheme that is not ours is a refusal case,
    # not a reason to answer with someone else's fund.
    #
    # The foreign check runs FIRST and unconditionally. Measured cause: "expense
    # ratio of Parag Parivartan Flexi Cap Fund" contains the alias "flexi cap",
    # so the alias match fired and the answer came back with HDFC Flexi Cap's own
    # 1.35% - confidently correct for the wrong fund. Aliases like "flexi cap",
    # "tax saver" and "small cap" are shared across AMCs, so a foreign AMC named
    # in the question outranks any alias match. Returning INSUFFICIENT is the
    # only safe answer: we hold no Parag Parivartan document.
    foreign = detect_foreign_scheme(question)
    if foreign:
        return Context(ok=False, reason=INSUFFICIENT_UNKNOWN_SCHEME)

    slug, _alias = detect_scheme(question)
    if slug is not None:
        kept = [h for h in kept if h["chunk_obj"].scheme_slug == slug]

    # Step 6: empty after filtering -> INSUFFICIENT. Never fall back.
    if not kept:
        reason = (
            INSUFFICIENT_UNKNOWN_SCHEME
            if slug is None
            else INSUFFICIENT_SCHEME_MISMATCH
        )
        return Context(ok=False, reason=reason)

    # Step 7: re-rank, then TOP_K, then the hard character stop.
    ranked = rerank(question, kept)[: config.TOP_K]
    chunks, scores = _truncate(
        [h["chunk_obj"] for h in ranked], [h["score"] for h in ranked], config.MAX_CONTEXT_CHARS
    )

    # Known-absence check. The floor passed, so these chunks are on-topic - but
    # "on topic" is not "contains the answer". A question about a fact the corpus
    # does not carry must not be handed to the generator, which would have to
    # either invent the number or pad the answer with whatever the top chunk
    # happened to say. Returns INSUFFICIENT with the fact named.
    fact = missing_fact(question)
    if fact and not corpus_has_fact(fact, chunks):
        return Context(ok=False, reason=INSUFFICIENT_FACT_ABSENT, missing=fact)

    return Context(chunks=chunks, scores=scores, ok=bool(chunks), reason="ok")


# ------------------------------------------------------------------------ CLI
def _debug(question: str, ctx: Context) -> None:
    print(RULE)
    print(f"QUERY: {question}")
    print(RULE)

    slug, alias = detect_scheme(question)
    foreign = detect_foreign_scheme(question)
    print(f"  scheme named   : {slug or '(none)'}   alias: {alias or '-'}")
    if foreign:
        print(f"  foreign scheme : {foreign!r} -> will refuse")
    print(f"  sim_floor      : {config.SIM_FLOOR}")
    print(f"  top_k/fetch_k  : {config.TOP_K} / {config.FETCH_K}")
    print()

    if not ctx.ok:
        print(f"  RESULT: INSUFFICIENT ({ctx.reason})")
        print(THIN)
        return

    for rank, (chunk, score) in enumerate(zip(ctx.chunks, ctx.scores), start=1):
        lexical = lexical_score(question, chunk)
        print(THIN)
        # Show both scores. Rank is driven by lexical with cosine as tiebreak
        # (architecture.md §5.3 step 7), so printing cosine alone made a correct
        # ranking look scrambled - e.g. rank 2 at 0.7347 above rank 3 at 0.7592.
        # Lexical is what explains the order; cosine explains the floor.
        print(f"  [{rank}] lexical {lexical:.4f}   cosine {score:.4f}   {chunk.chunk_id}")
        print(f"      scheme     : {chunk.scheme}  ({chunk.source_type})")
        print(f"      section    : {chunk.section}")
        print(f"      as_of      : {chunk.as_of_date or 'n/a'}")
        print(f"      quality    : {chunk.text_quality}")
        print(f"      url        : {chunk.source_url}")
        preview = " ".join(chunk.text.split())[:200]
        print(f"      preview    : {preview}{'...' if len(chunk.text) > 200 else ''}")
    print()
    print(f"  {len(ctx.chunks)} chunks, "
          f"{sum(len(c.text) for c in ctx.chunks)} chars "
          f"(cap {config.MAX_CONTEXT_CHARS})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Retrieve context for a question (no LLM).")
    ap.add_argument("question", nargs="+", help="the question to retrieve for")
    ap.add_argument("--debug", action="store_true", help="print per-hit detail")
    args = ap.parse_args(argv)

    # The corpus carries bullets and other non-cp1252 glyphs. Windows consoles
    # default to a legacy code page, so printing a preview would raise
    # UnicodeEncodeError partway through the output and lose the report.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover
                pass

    question = " ".join(args.question)

    # The index is a build artifact and a fresh checkout has none, so the CLI
    # builds it rather than refusing to run - the same recovery the UI does.
    try:
        index = ensure_index()
    except IndexError_ as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        ctx = retrieve(question, index)
    except Exception as exc:
        print(f"ERROR: retrieval failed: {exc}", file=sys.stderr)
        return 2

    if args.debug:
        _debug(question, ctx)
    else:
        if not ctx.ok:
            print(f"INSUFFICIENT ({ctx.reason})")
        else:
            for chunk, score in zip(ctx.chunks, ctx.scores):
                print(f"{score:.4f}  {chunk.chunk_id}")
    return 0 if ctx.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
