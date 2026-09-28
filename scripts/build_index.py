"""Build the local vector index from data/raw/.

    python scripts/build_index.py

ChromaDB's write path segfaults on this machine (see app/localindex.py), so the
searchable index is a numpy matrix on disk. Everything upstream is unchanged:
chunks come from app.chunking, vectors from app.embedder, ids are chunk_ids, and
the space is cosine over L2-normalised vectors - the same contract Chroma gives.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app import config
from app.chunking import load_and_chunk
from app.embedder import embed_texts
from app.ingest import check_coverage
from app.localindex import build_index


def main() -> int:
    if not config.RAW_DIR.exists():
        print(f"ERROR: {config.RAW_DIR} does not exist. Run Phase 1 first.", file=sys.stderr)
        return 2

    print(f"loading {config.RAW_DIR} ...")
    chunks = load_and_chunk()
    if not chunks:
        print("ERROR: chunker produced zero chunks.", file=sys.stderr)
        return 2
    print(f"  -> {len(chunks)} chunks")

    problems = check_coverage(chunks)
    if problems:
        print("\nCOVERAGE CHECK FAILED", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    print(f"embedding {len(chunks)} chunks with {config.EMBED_MODEL} ...")
    vectors = embed_texts([c.embed_text for c in chunks])
    print(f"  -> {len(vectors)} vectors, dim {len(vectors[0])}")

    index = build_index(chunks, vectors)
    print(f"\nwrote index to {index.root}")
    print(f"  count       : {index.manifest['count']}")
    print(f"  dim         : {index.manifest['dim']}")
    print(f"  space       : {index.manifest['space']}")
    print(f"  corpus      : {index.manifest['corpus_version']}")

    # Prove it is searchable rather than asserting it.
    probe = index.query(vectors[0], n=1)
    top = probe[0]
    ok = top["chunk_id"] == chunks[0].chunk_id
    print(f"\nself-query check: {'PASS' if ok else 'FAIL'} "
          f"(top hit {top['chunk_id']} at cosine {top['score']:.4f})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
