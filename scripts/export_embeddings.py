"""Export chunks and their embeddings to a readable txt file.

    python scripts/export_embeddings.py                 # 20 chunks + vectors
    python scripts/export_embeddings.py --all           # every chunk
    python scripts/export_embeddings.py --limit 60      # custom sample
    python scripts/export_embeddings.py --output data/processed/embeddings.txt

This exists because ChromaDB's write path segfaults on this machine (see
docs/embedding_export notes in the Phase 3 report). Embedding itself works
fine, so the vectors can be inspected and verified here without a store.

Why the numbers mean something: every vector is L2-normalised, so cosine
similarity between two vectors is just their dot product, and cosine distance
is 1 - dot product. That is what makes config.SIM_FLOOR a real threshold
rather than an arbitrary number.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app import config
from app.chunking import load_and_chunk
from app.embedder import embed_query, embed_texts

RULE = "=" * 100
THIN = "-" * 100

#: Queries chosen to exercise the three cases the bot must handle: a fact that
#: is present, a fact spread across two chunks, and a fact that is absent.
DEMO_QUERIES: tuple[str, ...] = (
    "what is the expense ratio of HDFC Large Cap Fund?",
    "is there an exit load on HDFC Small Cap Fund?",
    "what is the minimum SIP amount for HDFC Flexi Cap Fund?",
)


def fmt_vector(vec: list[float], per_line: int = 8, indent: str = "      ") -> str:
    out = []
    for i in range(0, len(vec), per_line):
        out.append(indent + "  ".join(f"{v:+.6f}" for v in vec[i : i + per_line]))
    return "\n".join(out)


def cosine(a: list[float], b: list[float]) -> float:
    """Both inputs are already unit vectors, so the dot product is the cosine."""
    return sum(x * y for x, y in zip(a, b))


def chunk_block(index: int, chunk, vector: list[float], total: int) -> list[str]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [
        THIN,
        f"CHUNK {index + 1} of {total}",
        THIN,
        f"  chunk_id     : {chunk.chunk_id}",
        f"  scheme       : {chunk.scheme}  ({chunk.scheme_slug})",
        f"  source_type  : {chunk.source_type}",
        f"  source_title : {chunk.source_title}",
        f"  section      : {chunk.section}",
        f"  as_of_date   : {chunk.as_of_date}   pages: {chunk.pages or 'n/a'}",
        f"  text_quality : {chunk.text_quality}",
        f"  text_hash    : {chunk.text_hash}",
        f"  chunk_chars  : {len(chunk.text)}",
        f"  vector_dim   : {len(vector)}",
        f"  L2_norm      : {norm:.9f}  (must be 1.0 - this is what makes cosine usable)",
        f"  source_url   : {chunk.source_url}",
        "",
        "  TEXT SENT TO THE MODEL (header + body, as embedded):",
    ] + [
        f"    {line}" for line in chunk.embed_text.splitlines()
    ] + [
        "",
        f"  EMBEDDING VECTOR ({len(vector)} floats):",
        fmt_vector(vector),
        "",
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Dump chunks and embeddings to a txt file.")
    ap.add_argument("--output", type=Path, default=REPO / "data" / "processed" / "embeddings.txt")
    ap.add_argument("--limit", type=int, default=20, help="how many chunk records to include")
    ap.add_argument("--all", action="store_true", help="include every chunk (large file)")
    args = ap.parse_args(argv)

    if not config.RAW_DIR.exists():
        print(f"ERROR: {config.RAW_DIR} does not exist. Run Phase 1 first.", file=sys.stderr)
        return 2

    print(f"loading {config.RAW_DIR} ...")
    chunks = load_and_chunk()
    print(f"  -> {len(chunks)} chunks")

    show = chunks if args.all else chunks[: args.limit]
    total_to_embed = len(show)

    # Part 1 ranks queries against the WHOLE corpus, not just the sample shown in
    # Part 2. Embedding only the sample would rank every query against a dozen
    # chunks from one alphabetically-first file and produce confident nonsense.
    print(f"embedding all {len(chunks)} chunks with {config.EMBED_MODEL} ...")
    all_vectors = embed_texts([c.embed_text for c in chunks])
    print(f"  -> {len(all_vectors)} vectors, dim {len(all_vectors[0])}")

    if total_to_embed < len(chunks):
        index = {c.chunk_id: i for i, c in enumerate(chunks)}
        vectors = [all_vectors[index[c.chunk_id]] for c in show]
    else:
        vectors = all_vectors
    print(f"showing {total_to_embed} chunk records in Part 2")

    lengths = [len(c.text) for c in chunks]
    norms = [math.sqrt(sum(v * v for v in vec)) for vec in vectors]

    out: list[str] = []
    out.append(RULE)
    out.append("HDFC MUTUAL FUND RAG - CHUNKS AND EMBEDDINGS".center(100))
    out.append(RULE)
    out.append(f"generated_at        : {datetime.now(timezone.utc).isoformat()}")
    out.append(f"corpus_version      : {config.CORPUS_VERSION}")
    out.append(f"embed_model         : {config.EMBED_MODEL}")
    out.append(f"embedding_dim       : {len(vectors[0])}")
    out.append("normalisation       : L2 (unit vectors)")
    out.append("cosine similarity   : equal to the dot product of two unit vectors")
    out.append("cosine distance     : 1 - dot product  (this is what Chroma stores)")
    out.append(f"sim_floor           : {config.SIM_FLOOR}  (retrieval discards anything below)")
    out.append(f"chunks in corpus    : {len(chunks)}")
    out.append(f"chunks shown here   : {total_to_embed}")
    out.append(f"chunk length range  : {min(lengths)} - {max(lengths)} chars (target {config.CHUNK_SIZE})")
    out.append(f"mean L2 norm        : {sum(norms) / len(norms):.9f}")
    out.append("")
    out.append(RULE)
    out.append("PART 1 - WORKED EXAMPLES: query embedding vs. nearest chunks")
    out.append(RULE)
    out.append("")
    out.append("The bot embeds a question exactly the same way it embeds a chunk, then")
    out.append("ranks all %d chunks by dot product. Below are the top 5 matches for three" % len(chunks))
    out.append("queries, including one whose answer is NOT in the corpus - that is the case")
    out.append("the bot must answer with INSUFFICIENT rather than inventing a number.")
    out.append("")

    for question in DEMO_QUERIES:
        qvec = embed_query(question)
        scored = sorted(
            ((cosine(qvec, v), c) for v, c in zip(all_vectors, chunks)),
            key=lambda p: p[0],
            reverse=True,
        )
        out.append(THIN)
        out.append(f"QUERY: {question}")
        out.append(THIN)
        out.append(f"  query vector dim : {len(qvec)}")
        out.append(f"  query L2 norm    : {math.sqrt(sum(v * v for v in qvec)):.9f}")
        out.append("  query vector (first 32 of %d):" % len(qvec))
        out.append(fmt_vector(qvec[:32], per_line=8, indent="      "))
        out.append("")
        out.append("  rank  cosine  distance  chunk_id")
        for rank, (sim, chunk) in enumerate(scored[:5], start=1):
            flag = "  <-- below SIM_FLOOR" if sim < config.SIM_FLOOR else ""
            out.append(
                f"  {rank:>4}  {sim:>6.4f}  {1 - sim:>9.4f}  {chunk.chunk_id}{flag}"
            )
        out.append("")

    out.append(RULE)
    out.append("PART 2 - CHUNKS WITH THEIR EMBEDDINGS")
    out.append(RULE)
    out.append("")

    for i, (chunk, vector) in enumerate(zip(show, vectors)):
        out.extend(chunk_block(i, chunk, vector, total_to_embed))

    out.append(RULE)
    out.append("END")
    out.append(RULE)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out) + "\n", encoding="utf-8")

    size = args.output.stat().st_size
    print(f"\nwrote {args.output}  ({size:,} bytes, {total_to_embed} chunks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
