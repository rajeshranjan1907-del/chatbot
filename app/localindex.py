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

`ensure_index()` is the single entry point every caller should use. It is the
answer to the one failure that a fresh deploy always hits: the index above is a
gitignored build artifact, so a Render instance - whose filesystem is ephemeral
and starts empty - has `data/raw` and no vectors, and the app could not answer a
single question until an index existed. `ensure_index` loads a usable index when
there is one and otherwise builds it from `data/raw` in the same process, so the
app recovers without anyone remembering a build step.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from app import config

VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"


class IndexError_(RuntimeError):
    """Raised when the index is missing, stale, or unreadable."""


def _wanted_dtype() -> str:
    return "float16" if config.EMBED_FP16 else "float32"


def _dtype_mismatch(manifest: dict) -> str | None:
    """Why the index's vectors are not the ones this process will compare against.

    `vectors.npy` is float32 in both cases - half precision halves the *resident
    model*, but the vectors are widened on the way in so `SIM_FLOOR` does not move
    between builds. That means a dtype change is invisible in the file itself and
    can only be caught from the manifest, so it is recorded and compared here.

    An index written before `embed_dtype` existed has no such key. Treated as a
    match when the process wants fp32 - which is the old behaviour, and the only
    combination that can legitimately be present on disk.
    """
    built = manifest.get("embed_dtype")
    if built is None:
        return None if _wanted_dtype() == "float32" else (
            "index predates EMBED_FP16 and was built in float32, but this process "
            "wants float16. Rebuild, or every similarity is computed against a "
            "different vector space."
        )
    wanted = _wanted_dtype()
    if built != wanted:
        return (
            f"index was built with embed_dtype={built!r} but this process is "
            f"configured for {wanted!r}. Rebuild, or the similarities compare "
            f"vectors from two different spaces."
        )
    return None


def _check_dtype(manifest: dict) -> None:
    reason = _dtype_mismatch(manifest)
    if reason:
        raise IndexError_(reason)


#: Set only when the configured index dir turned out to be unwritable and we fell
#: back to a temp dir. Sticky for the process so that a later `load()` with no
#: argument reads the directory `ensure_index()` actually wrote, rather than
#: looking in a place that can never hold an index.
_ACTIVE_ROOT: Path | None = None


def default_index_dir() -> Path:
    if _ACTIVE_ROOT is not None:
        return _ACTIVE_ROOT
    return config.INDEX_DIR


class LocalIndex:
    """Brute-force cosine search over an on-disk float32 matrix."""

    def __init__(self, root: Path, vectors, records: list[dict], manifest: dict):
        self.root = root
        self.vectors = vectors  # numpy array (n, dim)
        self.records = records
        self.manifest = manifest

    # ---------------------------------------------------------------- build
    @classmethod
    def build(cls, chunks: list, vectors, root: Path | None = None) -> "LocalIndex":
        """Write chunks + vectors to disk. Deterministic: same input, same bytes.

        `vectors` may be a list of lists or a numpy array. The array form exists so
        the build can hand over the matrix it already has instead of round-tripping
        276,480 Python floats through `np.asarray` while both copies are resident.
        """
        import numpy as np

        root = root or default_index_dir()
        root.mkdir(parents=True, exist_ok=True)

        if not chunks:
            raise IndexError_("cannot build an index from zero chunks")
        if len(chunks) != len(vectors):
            raise IndexError_(
                f"chunk/vector count mismatch: {len(chunks)} chunks, {len(vectors)} vectors"
            )

        # asarray on an existing float32 array is a no-op rather than a copy, which
        # is the whole point of accepting it.
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
            # Recorded so an index built with EMBED_FP16=1 is recognised as a
            # different vector space from an fp32 one and rebuilt rather than
            # silently searched with mismatched similarities. The vectors are
            # stored as float32 either way, so dtype alone cannot reveal the
            # mismatch - only the manifest can.
            "embed_dtype": "float16" if config.EMBED_FP16 else "float32",
            "embed_batch": config.EMBED_BATCH,
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
                f"  python scripts/build_index.py\n"
                f"or call ensure_index(), which builds it on demand."
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
        _check_dtype(self.manifest)

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


def build_index(chunks: list, vectors, root: Path | None = None) -> LocalIndex:
    return LocalIndex.build(chunks, vectors, root)


# ------------------------------------------------------------------- bootstrap
def unusable_reason(root: Path) -> str | None:
    """Why the index at `root` cannot be used, or None if it is fine.

    Separate from `load()` so a caller can tell "nothing there yet" (build it)
    from "something is there but wrong" (rebuild it, and say so). The two
    produce different deploy logs, and a silent rebuild of a corrupt index is
    how a real disk problem goes unnoticed.
    """
    if not (root / MANIFEST_FILE).exists():
        return f"no index at {root}"

    try:
        manifest = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"{root / MANIFEST_FILE} is unreadable ({exc})"

    model = manifest.get("embed_model")
    if model != config.EMBED_MODEL:
        return (
            f"index was built with embed_model={model!r} but config.EMBED_MODEL is "
            f"{config.EMBED_MODEL!r}"
        )

    dtype = _dtype_mismatch(manifest)
    if dtype:
        return dtype

    version = manifest.get("corpus_version")
    if version != config.CORPUS_VERSION:
        return (
            f"index corpus_version is {version!r} but config.CORPUS_VERSION is "
            f"{config.CORPUS_VERSION!r}"
        )

    missing = [n for n in (VECTORS_FILE, CHUNKS_FILE) if not (root / n).exists()]
    if missing:
        return f"index at {root} is incomplete: missing {', '.join(missing)}"

    return None


def _writable(root: Path) -> bool:
    """True if `root` exists (or can be created) and accepts a write."""
    probe = root / ".write-probe"
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _resolve_build_root(preferred: Path, log: Callable[[str], None]) -> Path:
    """A directory we can actually write the index into.

    The configured path is the normal answer. A read-only checkout is the
    exception - a container image or a mount that does not allow writes under
    the source tree - and there the only sane thing is to build somewhere
    writable rather than to fail a deploy over it. The choice is recorded in
    `_ACTIVE_ROOT` so `load()` looks in the same place.
    """
    global _ACTIVE_ROOT
    if _writable(preferred):
        return preferred

    fallback = Path(tempfile.gettempdir()) / "hdfc_mf_facts_vector_index"
    log(f"{preferred} is not writable; building the index in {fallback} instead")
    if not _writable(fallback):
        raise IndexError_(
            f"cannot write a vector index: neither {preferred} nor {fallback} is "
            f"writable. Set INDEX_DIR to a writable directory."
        )
    _ACTIVE_ROOT = fallback
    return fallback


def build_from_raw(
    root: Path | None = None,
    log: Callable[[str], None] = print,
) -> LocalIndex:
    """Chunk `data/raw`, embed it and write the index. Raises on any failure.

    The same sequence `scripts/build_index.py` runs, kept here so the deploy
    path and the CLI path cannot drift apart: a prebuilt index and a
    self-healed one have to be byte-compatible or search results depend on who
    started the app.
    """
    # Imported here, not at module scope: the chunker and the embedder pull in
    # torch, and a caller that only wants to *read* an existing index should not
    # pay ~90 MB of model load to do it.
    import numpy as np

    from app.chunking import load_and_chunk
    from app.embedder import embed_texts_np
    from app.ingest import check_coverage

    if not config.RAW_DIR.exists():
        raise IndexError_(
            f"{config.RAW_DIR} does not exist, so there is no corpus to build an "
            f"index from. data/raw is committed to the repo, so this means the "
            f"deploy did not check the source out - check the build log."
        )

    target = _resolve_build_root(Path(root) if root is not None else default_index_dir(), log)

    log(f"loading {config.RAW_DIR} ...")
    chunks = load_and_chunk()
    if not chunks:
        raise IndexError_("the chunker produced zero chunks, so there is nothing to index")
    log(f"  -> {len(chunks)} chunks")

    # The same gate ingest applies. A deploy that half-cloned the corpus must
    # fail loudly rather than serve a bot that cannot answer "expense ratio".
    problems = check_coverage(chunks)
    if problems:
        raise IndexError_("coverage check failed:\n" + "\n".join(f"  - {p}" for p in problems))

    log(
        f"embedding {len(chunks)} chunks with {config.EMBED_MODEL} "
        f"(batch {config.EMBED_BATCH}, fp16={config.EMBED_FP16}) ..."
    )
    vectors = embed_texts_np([c.embed_text for c in chunks])
    # A hand-rolled embedder - and the test fixtures - may hand back a plain list
    # of lists, so the shape is read off the normalised matrix below rather than
    # off the return value.
    matrix = np.asarray(vectors, dtype="float32")
    log(f"  -> {matrix.shape[0]} vectors, dim {matrix.shape[1]}")

    index = LocalIndex.build(chunks, vectors, target)
    log(f"  -> wrote {index.manifest['count']} vectors to {target}")

    # Prove it is searchable rather than asserting it. An index that was written
    # but cannot be read back is the exact failure this whole path exists for.
    top = index.query(vectors[0], n=1)[0]
    if top["chunk_id"] != chunks[0].chunk_id:
        raise IndexError_(
            f"the freshly built index does not return its own first chunk: got "
            f"{top['chunk_id']!r}, expected {chunks[0].chunk_id!r}"
        )
    log(f"  -> self-query check PASS ({top['chunk_id']} at cosine {top['score']:.4f})")
    return index


def ensure_index(
    root: Path | None = None,
    log: Callable[[str], None] = print,
) -> LocalIndex:
    """Return a usable index, building it from `data/raw` if there isn't one.

    This is the call the UI, the CLI and `retrieve()` all make. Load is the
    fast path and costs one small JSON read; a missing, stale, corrupt or
    model-mismatched index is rebuilt, because every one of those states means
    the app cannot answer and there is nothing to lose by rebuilding.

    Raises `IndexError_` with the reason when the corpus is missing or the
    coverage gate fails, so the caller can show something actionable rather than
    a bare traceback.
    """
    target = Path(root) if root is not None else default_index_dir()

    reason = unusable_reason(target)
    if reason is None:
        try:
            return LocalIndex.load(target)
        except (IndexError_, OSError, ValueError) as exc:
            reason = f"index at {target} could not be read ({exc})"

    log(f"building the vector index: {reason}")
    return build_from_raw(target, log=log)
