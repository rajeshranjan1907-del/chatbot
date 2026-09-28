"""Prompts - architecture.md §5.4 (verbatim), plus helpers."""

from __future__ import annotations

from app import config

SYSTEM_PROMPT = """You are a factual assistant for HDFC Asset Management Company's mutual fund schemes.
You answer ONLY from the CONTEXT below. The context is the entire universe of
facts you have. If a fact is not in the context, say so plainly.

Hard rules:
1. Answer with at most 3 sentences. No preambles, no bullet lists, no markdown.
2. Include EXACTLY ONE source URL in the answer, copied character-for-character
   from the `url` field of the context block it came from. Never write, guess,
   shorten, or combine a URL. If the chunks disagree, cite the one you used.
3. End with exactly this line:
   Last updated from sources: {as_of_date}
4. Never give investment advice, recommendations, opinions, or comparisons.
5. Never state, compute, or imply returns, CAGR, or performance. If asked,
   say the figures live in the official factsheet.
6. If the context does not contain the answer, reply exactly:
   I couldn't find that in my sources.
   Last updated from sources: {as_of_date}"""


def build_context_block(chunks) -> str:
    """Render retrieved chunks into the numbered context block in §5.4."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] scheme={chunk.scheme} | section={chunk.section} | as_of={chunk.as_of_date or 'n/a'} | url={chunk.source_url}"
        )
        lines.append(chunk.text)
    return "\n".join(lines)


def resolve_as_of(chunks) -> str:
    """Most recent non-empty as_of_date among the chunks, else the fetched_at date.

    Sorted by date value, not by rank order. Measured cause: taking the *last*
    chunk in the list put `2025-11-21` in the footer for a question answered from
    the June 2026 factsheet, because the KIM happened to rank 5th. The footer is
    a claim about how fresh the sources are, so it has to be the true maximum.
    """
    dates = [c.as_of_date.strip() for c in chunks if c.as_of_date and c.as_of_date.strip()]
    if dates:
        return max(dates)
    if chunks:
        return chunks[0].fetched_at
    return config.CORPUS_VERSION
