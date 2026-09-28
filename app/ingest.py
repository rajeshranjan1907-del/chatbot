"""Ingestion pipeline - architecture.md §4.1, stages 1-6.

    python -m app.ingest                  # normal run
    python -m app.ingest --dry-run        # chunk + report, no embedding
    python -m app.ingest --force          # rebuild from scratch

Order is deliberate: chunk, WRITE chunks.txt, coverage check, embed, upsert,
report. The human-readable dump and the coverage gate both happen BEFORE
anything is written to the vector store, so a broken run cannot leave a
half-populated index behind.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from app import config
from app.chunking import load_and_chunk
from app.store import (
    StoreError,
    assert_cosine_space,
    collection_count,
    get_client,
    get_collection,
    upsert,
)

SEPARATOR = "=" * 80
RULE = "-" * 80


def write_chunks_txt(chunks: list, path: Path | None = None) -> Path:
    """Dump every chunk with full metadata so a human can audit before the demo.

    Deterministic: chunk order is the chunker's, so re-running produces a
    byte-identical file for the same corpus.
    """
    path = path or config.CHUNKS_TXT
    path.parent.mkdir(parents=True, exist_ok=True)

    out: list[str] = []
    for c in chunks:
        out.append(SEPARATOR)
        out.append(f"chunk_id : {c.chunk_id}")
        out.append(f"scheme   : {c.scheme}")
        out.append(f"type     : {c.source_type:<14} section: {c.section}")
        out.append(f"as_of    : {c.as_of_date}    pages: {c.pages or 'n/a'}")
        out.append(f"quality  : {c.text_quality}")
        out.append(f"url      : {c.source_url}")
        out.append(RULE)
        out.append("")
        out.append(c.text)
        out.append("")

    path.write_text("\n".join(out), encoding="utf-8")
    return path


def coverage_report(chunks: list) -> dict:
    per_scheme = Counter(c.scheme_slug for c in chunks)
    per_type = Counter(c.source_type for c in chunks)
    degraded = sum(1 for c in chunks if c.text_quality == "degraded")
    return {
        "total_chunks": len(chunks),
        "per_scheme": dict(per_scheme),
        "per_source_type": dict(per_type),
        "degraded": degraded,
        "clean": len(chunks) - degraded,
    }


def check_coverage(chunks: list) -> list[str]:
    """Return a list of problems. Empty list means the corpus is complete.

    Required: all five schemes, at least one fee-bearing source and one guide.
    The PRD's hard floor is a facts corpus; a run that silently lost the KIMs
    would otherwise produce a confident bot that cannot answer "what is the
    minimum lump sum" or "what are the charges".
    """
    problems: list[str] = []

    per_scheme = Counter(c.scheme_slug for c in chunks)
    for slug, name in config.SCHEMES.items():
        if per_scheme.get(slug, 0) == 0:
            problems.append(f"no chunks for scheme {slug} ({name})")

    per_type = Counter(c.source_type for c in chunks)
    for required in config.FEE_BEARING_SOURCE_TYPES:
        if per_type.get(required, 0) == 0:
            problems.append(
                f"no fee-bearing source: source_type={required!r} is absent, so "
                f"fee questions (expense ratio, charges, minimums) are "
                f"unanswerable (present: {sorted(per_type)})"
            )
    for required in config.GUIDE_SOURCE_TYPES:
        if per_type.get(required, 0) == 0:
            problems.append(
                f"no guide source: source_type={required!r} is absent, so "
                f"process questions (CAS, redemption, statements) are "
                f"unanswerable (present: {sorted(per_type)})"
            )

    if not chunks:
        problems.append("corpus produced zero chunks")

    return problems


def print_report(chunks: list, cov: dict, dry_run: bool, out_path: Path | None) -> None:
    print()
    print(SEPARATOR)
    print("INGEST REPORT".center(80))
    print(SEPARATOR)
    print(f"corpus_version    : {config.CORPUS_VERSION}")
    print(f"embed_model       : {config.EMBED_MODEL}")
    print(f"chunk_size        : {config.CHUNK_SIZE}   overlap: {config.TAIL_OVERLAP}")
    print(f"chunks            : {cov['total_chunks']}")
    print(f"  clean           : {cov['clean']}")
    print(f"  degraded        : {cov['degraded']}  (dropped-glyph regions)")
    print()
    print("chunks per scheme")
    for slug in sorted(cov["per_scheme"]):
        name = config.SCHEMES.get(slug, slug)
        print(f"  {slug:<26}{cov['per_scheme'][slug]:>5}   {name}")
    print()
    print("chunks per source_type")
    for st in sorted(cov["per_source_type"]):
        print(f"  {st:<26}{cov['per_source_type'][st]:>5}")
    print()
    if out_path:
        print(f"chunks.txt        : {out_path}")
    if dry_run:
        print("mode              : DRY RUN (nothing embedded or stored)")
    print(SEPARATOR)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the vector store from data/raw/.")
    ap.add_argument("--force", action="store_true", help="rebuild the collection from scratch")
    ap.add_argument("--dry-run", action="store_true", help="chunk and report without embedding")
    args = ap.parse_args(argv)

    if not config.RAW_DIR.exists():
        print(f"ERROR: {config.RAW_DIR} does not exist.", file=sys.stderr)
        print("Run Phase 1 first: python scripts/fetch_sources.py --fetch --extract", file=sys.stderr)
        return 2

    print(f"loading {config.RAW_DIR} ...")
    chunks = load_and_chunk()
    if not chunks:
        print("ERROR: chunker produced zero chunks.", file=sys.stderr)
        return 2
    print(f"  -> {len(chunks)} chunks")

    # Stage 4: dump BEFORE embedding, so a human can audit the corpus.
    out_path = write_chunks_txt(chunks)
    print(f"  -> wrote {out_path}")

    cov = coverage_report(chunks)
    print_report(chunks, cov, args.dry_run, out_path)

    # Coverage gate - non-zero exit on an incomplete corpus.
    problems = check_coverage(chunks)
    if problems:
        print("\nCOVERAGE CHECK FAILED", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print("\nNothing was embedded. Fix the corpus and re-run.", file=sys.stderr)
        return 1
    print("coverage check    : OK (5 schemes, fees + guide present)")

    if args.dry_run:
        return 0

    # Assert we are embedding the model the whole app is configured for.
    assert config.EMBED_MODEL == config._env(
        "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    ), "EMBED_MODEL drifted from the configured value"

    from app.embedder import embed_texts

    try:
        client = get_client()
        collection = get_collection(client)
        assert_cosine_space(collection)
        print(f"\ncollection        : {collection.name}")
        print(f"hnsw:space        : {collection.metadata.get('hnsw:space')}  (asserted)")
    except StoreError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2

    before = collection_count(collection)
    print(f"existing vectors  : {before}")

    if args.force:
        collection.delete(where={})
        before = 0
        print("forced rebuild    : collection cleared")

    print(f"\nembedding {len(chunks)} chunks with {config.EMBED_MODEL} (batch {config.EMBED_BATCH}) ...")
    vectors = embed_texts([c.embed_text for c in chunks])
    print(f"  -> {len(vectors)} vectors, dim {len(vectors[0])}")

    upsert(collection, chunks, vectors)
    after = collection_count(collection)
    print(f"vectors in store  : {after}")
    print(f"delta             : {after - before:+d}  "
          f"({'idempotent: no change' if after == before else 'new/updated chunks'})")

    report = {
        "corpus_version": config.CORPUS_VERSION,
        "embed_model": config.EMBED_MODEL,
        "collection": config.CHROMA_COLLECTION,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "chunks": len(chunks),
        "vectors": after,
        "coverage": cov,
    }
    config.INGEST_REPORT.parent.mkdir(parents=True, exist_ok=True)
    config.INGEST_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {config.INGEST_REPORT}")
    print(SEPARATOR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
