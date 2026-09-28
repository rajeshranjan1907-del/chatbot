"""Local vector index - a pure-numpy stand-in for ChromaDB.

Why this exists: ChromaDB's Rust binding segfaults on any write on this
machine (reproduced on 1.5.9, 1.4.1, 0.6.3 and 0.5.23 with a three-line
repro, so it is not a version problem). Embedding works fine, and Phase 4
needs to search, not to depend on a particular store.

So this module keeps the same contract Chroma provides - persistent on disk,
cosine over L2-normalised vectors, ids equal to `chunk_id`, upsert keyed on
`chunk_id` - in about 200 lines of numpy. `app/retrieve.py` treats it as a
drop-in: `query(q_vec, n)` returns hits in the same shape Chroma would.

Brute force is the right algorithm here. 720 chunks x 384 dims is a 276K-element
matmul, which is sub-millisecond. An approximate index would add a dependency
and a tuning problem to buy speed we do not need.

Layout on disk:
    <dir>/vectors.npy     float32, shape (n_chunks, 384), C order
    <dir>/chunks.jsonl   one JSON object per line, aligned to the row order
    <dir>/manifest.json  corpus_version, model, dim, count, built_at
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app import config

VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"


class IndexError_(RuntimeError):
    """Raised when the index is missing, stale, or unreadable."""


def default_index_dir() -> Path:
    return config.PROCESSED_DIR / "vector_index"


class LocalIndex:
    """Brute-force cosine search over an on-disk float32 matrix."""

    def __init__(self, root: Path, vectors, records: list[dict], manifest: dict):
        self.root = root
        self.vectors = vectors  # numpy array (n, dim)
        self.records = records
        self.manifest = manifest

    # ---------------------------------------------------------------- build
    @classmethod
    def build(cls, chunks: list, vectors: list[list[float]], root: Path | None = None) -> "LocalIndex":
        """Write chunks + vectors to disk. Deterministic: same input, same bytes."""
        import numpy as np

        root = root or default_index_dir()
        root.mkdir(parents=True, exist_ok=True)

        if not chunks:
            raise IndexError_("cannot build an index from zero chunks")
        if len(chunks) != len(vectors):
            raise IndexError_(
                f"chunk/vector count mismatch: {len(chunks)} chunks, {len(vectors)} vectors"
            )

        matrix = np.asarray(vectors, dtype="float32")
        if matrix.ndim != 2:
            raise IndexError_(f"expected a 2-D matrix, got shape {matrix.shape}")

        # Re-normalise defensively. SIM_FLOOR is only meaningful if these are
        # unit vectors, and float32 round-trip alone moves the norm by ~1e-7.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise IndexError_("zero-norm vector in corpus; cannot normalise")
        matrix = matrix / norms

        np.save(root / VECTORS_FILE, matrix, allow_pickle=False)
        with (root / CHUNKS_FILE).open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")

        manifest = {
            "corpus_version": config.CORPUS_VERSION,
            "embed_model": config.EMBED_MODEL,
            "dim": int(matrix.shape[1]),
            "count": int(matrix.shape[0]),
            "space": "cosine",
            "normalised": True,
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        (root / MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return cls(root, matrix, [c.to_dict() for c in chunks], manifest)

    # ----------------------------------------------------------------- load
    @classmethod
    def load(cls, root: Path | None = None) -> "LocalIndex":
        import numpy as np

        root = root or default_index_dir()
        if not (root / MANIFEST_FILE).exists():
            raise IndexError_(
                f"no vector index at {root}. Build it with:\n"
                f"  python scripts/build_index.py"
            )
        manifest = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
        vectors = np.load(root / VECTORS_FILE, allow_pickle=False)
        records = [
            json.loads(line)
            for line in (root / CHUNKS_FILE).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if vectors.shape[0] != len(records):
            raise IndexError_(
                f"index is corrupt: {vectors.shape[0]} vectors vs {len(records)} records"
            )
        return cls(root, vectors, records, manifest)

    @classmethod
    def exists(cls, root: Path | None = None) -> bool:
        root = root or default_index_dir()
        return (root / MANIFEST_FILE).exists()

    # ---------------------------------------------------------------- query
    def query(self, q_vec: list[float], n: int) -> list[dict]:
        """Return up to `n` hits, most similar first.

        Each hit is {chunk_id, distance, score, chunk}. `distance` is cosine
        distance (1 - similarity), matching what Chroma reports, and `score` is
        the cosine similarity, which is what SIM_FLOOR is compared against.
        """
        import numpy as np

        if self.manifest.get("embed_model") != config.EMBED_MODEL:
            raise IndexError_(
                f"index was built with {self.manifest.get('embed_model')!r} but "
                f"config.EMBED_MODEL is {config.EMBED_MODEL!r}. Rebuild, or the "
                f"similarities are meaningless."
            )

        query = np.asarray([q_vec], dtype="float32")
        if query.shape[1] != self.vectors.shape[1]:
            raise IndexError_(
                f"query dim {query.shape[1]} != index dim {self.vectors.shape[1]}"
            )
        norm = float(np.linalg.norm(query))
        if norm == 0:
            return []
        query = query / norm

        # (1, dim) @ (dim, n) -> (1, n) cosine similarities.
        sims = (query @ self.vectors.T)[0]
        take = min(n, sims.shape[0])
        # argpartition finds the top-k without sorting all 720.
        top = np.argpartition(-sims, take - 1)[:take]
        top = top[np.argsort(-sims[top])]

        hits = []
        for idx in top:
            sim = float(sims[idx])
            hits.append(
                {
                    "chunk_id": self.records[idx]["chunk_id"],
                    "distance": 1.0 - sim,
                    "score": sim,
                    "chunk": self.records[idx],
                }
            )
        return hits


def build_index(chunks: list, vectors: list[list[float]], root: Path | None = None) -> LocalIndex:
    return LocalIndex.build(chunks, vectors, root)
