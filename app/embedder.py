"""Shared SentenceTransformer instance.

One model, loaded once per process, used by BOTH ingest and query. Two separate
loaders is how a model id silently drifts between writing and reading, which
produces an index that cannot be searched. `get_model` is lru_cache'd so the
~90 MB MiniLM loads once no matter how many batches are embedded.

Never hardcode a model id here; the value comes from config.EMBED_MODEL.
"""

from __future__ import annotations

from functools import lru_cache

from app import config


class EmbedderError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_model():
    """Return the cached SentenceTransformer. Raises with setup guidance."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbedderError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        ) from exc
    return SentenceTransformer(config.EMBED_MODEL)


def embed_texts(texts: list[str], batch_size: int | None = None) -> list[list[float]]:
    """Embed with L2 normalisation, so cosine distance is a relevance score.

    `normalize_embeddings=True` is what makes `SIM_FLOOR` meaningful - without
    it, cosine and length get entangled.
    """
    if not texts:
        return []
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size or config.EMBED_BATCH,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return [list(map(float, v)) for v in vectors]


def embed_query(text: str) -> list[float]:
    """Embed a single question, normalised identically to the corpus."""
    return embed_texts([text])[0]
