"""Vector store - architecture.md §4.4.

ChromaDB `PersistentClient`, one collection per corpus version, cosine distance
over L2-normalised embeddings, upsert keyed on `chunk_id`.

The cosine assertion is the important part. `SIM_FLOOR` (0.30) is a *cosine*
threshold; if Chroma silently fell back to L2 the score would be a squared
Euclidean distance, the floor would stop meaning what it says, and retrieval
would quietly return nonsense. So `assert_cosine_space` verifies the collection
metadata actually reports cosine and raises otherwise. Ingest calls it on every
run and refuses to write to a misconfigured collection.
"""

from __future__ import annotations

from pathlib import Path

from app import config


class StoreError(RuntimeError):
    """Raised when the vector store is misconfigured or unreachable."""


def _chroma():
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover
        raise StoreError(
            "chromadb is not installed. Run: pip install chromadb sentence-transformers"
        ) from exc
    return chromadb


def get_client(path: Path | str | None = None):
    """A persistent client rooted at `chroma_db/` (or the given path)."""
    chromadb = _chroma()
    db_path = Path(path or config.CHROMA_PATH)
    db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(db_path))


def get_collection(client, name: str | None = None, create: bool = True):
    """Open (or create) the collection with cosine distance configured.

    `metadata` is only applied at creation time - passing it to
    `get_or_create` on an existing collection does not reconfigure it, which is
    exactly why the assertion below is necessary.
    """
    name = name or config.CHROMA_COLLECTION
    try:
        if create:
            return client.get_or_create_collection(
                name=name, metadata=dict(config.CHROMA_METADATA)
            )
        return client.get_collection(name=name)
    except Exception as exc:
        raise StoreError(f"could not open collection {name!r}: {exc}") from exc


def assert_cosine_space(collection) -> None:
    """Raise unless the collection really is cosine.

    Guards the assumption behind SIM_FLOOR. A silent L2 fallback would make the
    score incomparable across chunks of different length.
    """
    meta = getattr(collection, "metadata", None) or {}
    space = str(meta.get("hnsw:space", "")).lower()
    if space != "cosine":
        raise StoreError(
            f"collection {collection.name!r} reports hnsw:space="
            f"{meta.get('hnsw:space')!r}, expected 'cosine'. "
            "A silent fallback to L2 invalidates SIM_FLOOR "
            f"({config.SIM_FLOOR}) because the score would no longer be a "
            "cosine similarity. Delete chroma_db/ and re-run "
            "`python -m app.ingest --force` to rebuild the collection."
        )


def collection_count(collection) -> int:
    return int(collection.count())


def upsert(collection, chunks: list, embeddings: list[list[float]]) -> None:
    """Idempotent write keyed on `chunk_id`.

    Chroma metadata rejects `None`, so absent dates are stored as "".
    """
    if not chunks:
        return
    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            "scheme": c.scheme,
            "scheme_slug": c.scheme_slug,
            "amc": c.amc,
            "category": c.category,
            "source_type": c.source_type,
            "source_title": c.source_title,
            "source_url": c.source_url,
            "section": c.section,
            "chunk_index": int(c.chunk_index),
            "as_of_date": c.as_of_date or "",
            "pages": c.pages or "",
            "text_hash": c.text_hash,
            "text_quality": c.text_quality,
            "corpus_version": c.corpus_version,
        }
        for c in chunks
    ]
    collection.upsert(
        ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
    )


def get_ids(collection) -> set[str]:
    """All ids in the collection, for the idempotency check."""
    return set(collection.get(include=[])["ids"])
