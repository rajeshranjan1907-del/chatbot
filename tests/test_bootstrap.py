"""Index bootstrap tests - the fix for the Render "no vector index" failure.

The index is a gitignored build artifact, so a Render instance starts with the
corpus and no vectors. Before `ensure_index` existed, that state was terminal:
`LocalIndex.load()` raised, the UI showed the "build it with ..." error and the
deployment could not answer a single question. These tests pin the recovery
contract, using a fake chunker and a fake embedder so they stay fast and never
download the model:

  1. A missing index is built, and the result is searchable.
  2. A usable index is loaded, not rebuilt. Rebuilding on every call would turn
     a 1 MB load into a full re-embed of the corpus on each request.
  3. A stale index - wrong model, wrong corpus version, incomplete files - is
     rebuilt, because each of those states answers from vectors that mean
     nothing.
  4. An unwritable index dir does not fail the deploy; it falls back and says so.
  5. A corpus that is missing or fails coverage raises an actionable error
     instead of writing a half-built index the app would then trust.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app import config
from app import localindex
from app.chunking import Chunk
from app.localindex import (
    CHUNKS_FILE,
    MANIFEST_FILE,
    VECTORS_FILE,
    IndexError_,
    LocalIndex,
    build_from_raw,
    ensure_index,
    unusable_reason,
)

DIM = 8


# ------------------------------------------------------------------- fixtures
def _chunk(i: int) -> Chunk:
    return Chunk(
        chunk_id=f"{config.CORPUS_VERSION}:test-source:{i:04d}",
        text=f"EXPENSE RATIO chunk {i}: Regular 1.75% Direct 1.18%",
        embed_text=f"chunk {i}",
        scheme="HDFC Large Cap Fund",
        scheme_slug="hdfc-large-cap",
        amc=config.AMC_NAME,
        category="large-cap",
        source_type="factsheet",
        source_title="test",
        source_url="https://example.invalid/test",
        section="EXPENSE RATIO",
        chunk_index=i,
        as_of_date="2026-06-30",
        fetched_at="2026-09-28",
        corpus_version=config.CORPUS_VERSION,
        pages="1",
        text_quality="clean",
    )


@pytest.fixture(autouse=True)
def _no_leaked_active_root():
    """`_ACTIVE_ROOT` is process-global; a test must not change where the rest
    of the suite looks for an index."""
    saved = localindex._ACTIVE_ROOT
    localindex._ACTIVE_ROOT = None
    yield
    localindex._ACTIVE_ROOT = saved


@pytest.fixture
def fake_corpus(monkeypatch):
    """Replace the chunker, the embedder and the coverage gate. Counts builds so
    tests can assert that a usable index is not rebuilt.

    The gate is stubbed to pass because these five synthetic chunks cannot
    satisfy the real one - they are all from one scheme. The gate's own
    behaviour is covered in tests/test_ingest.py; what matters here is that
    `build_from_raw` calls it and refuses to write when it fails.
    """
    state = {"chunks": [_chunk(i) for i in range(5)], "embeds": 0}

    def fake_load_and_chunk(*_args, **_kwargs):
        return state["chunks"]

    def fake_embed_texts(texts, *_args, **_kwargs):
        state["embeds"] += 1
        # Deterministic, distinct and unit-ish vectors: index i is one-hot-ish so
        # a self-query for chunk i must return chunk i.
        vectors = []
        for i in range(len(texts)):
            v = [0.0] * DIM
            v[i % DIM] = 1.0
            vectors.append(v)
        return vectors

    monkeypatch.setattr("app.chunking.load_and_chunk", fake_load_and_chunk)
    monkeypatch.setattr("app.embedder.embed_texts", fake_embed_texts)
    monkeypatch.setattr("app.ingest.check_coverage", lambda chunks: [])
    return state


# --------------------------------------------------------- 1. builds on demand
def test_missing_index_is_built_and_searchable(tmp_path, fake_corpus):
    root = tmp_path / "vector_index"
    assert not LocalIndex.exists(root)

    index = ensure_index(root)

    assert LocalIndex.exists(root), "ensure_index did not write the index"
    assert index.manifest["count"] == len(fake_corpus["chunks"])
    assert index.manifest["embed_model"] == config.EMBED_MODEL
    # Searchable, not merely present: the first chunk must find itself.
    top = index.query([1.0] + [0.0] * (DIM - 1), n=1)[0]
    assert top["chunk_id"] == fake_corpus["chunks"][0].chunk_id


def test_built_index_is_written_to_the_requested_dir(tmp_path, fake_corpus):
    root = tmp_path / "nested" / "vector_index"
    ensure_index(root)
    for name in (VECTORS_FILE, CHUNKS_FILE, MANIFEST_FILE):
        assert (root / name).exists(), f"{name} missing from {root}"


def test_a_usable_index_is_loaded_not_rebuilt(tmp_path, fake_corpus):
    root = tmp_path / "vector_index"
    ensure_index(root)
    assert fake_corpus["embeds"] == 1

    again = ensure_index(root)

    assert fake_corpus["embeds"] == 1, "a valid index was re-embedded on load"
    assert again.manifest["count"] == 5


# ------------------------------------------------------------ 2. stale is fixed
def test_unusable_reason_is_none_for_a_fresh_index(tmp_path, fake_corpus):
    root = tmp_path / "vector_index"
    ensure_index(root)
    assert unusable_reason(root) is None


def test_unusable_reason_names_a_missing_index(tmp_path):
    reason = unusable_reason(tmp_path / "nowhere")
    assert reason is not None and "no index at" in reason


def test_index_built_with_another_model_is_rebuilt(tmp_path, fake_corpus):
    """Searching MiniLM vectors with a different encoder produces similarities
    that are simply wrong, so a mismatch must be repaired, not tolerated."""
    root = tmp_path / "vector_index"
    ensure_index(root)
    manifest_path = root / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["embed_model"] = "some/other-model"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert "embed_model" in unusable_reason(root)

    ensure_index(root)

    assert fake_corpus["embeds"] == 2, "a model-mismatched index was reused"
    assert unusable_reason(root) is None


def test_index_from_another_corpus_version_is_rebuilt(tmp_path, fake_corpus):
    """Chunk ids embed the corpus version, so a stale index is answering
    questions from a corpus that is no longer deployed."""
    root = tmp_path / "vector_index"
    ensure_index(root)
    manifest_path = root / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["corpus_version"] = "1999-01-01.1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert "corpus_version" in unusable_reason(root)
    ensure_index(root)
    assert fake_corpus["embeds"] == 2, "a stale-corpus index was reused"


def test_index_missing_its_vectors_is_rebuilt(tmp_path, fake_corpus):
    """A half-deleted index must not be read as if it were complete."""
    root = tmp_path / "vector_index"
    ensure_index(root)
    (root / VECTORS_FILE).unlink()

    assert VECTORS_FILE in unusable_reason(root)
    ensure_index(root)
    assert (root / VECTORS_FILE).exists()
    assert fake_corpus["embeds"] == 2, "an incomplete index was reused"


def test_unreadable_manifest_is_reported_not_raised(tmp_path, fake_corpus):
    root = tmp_path / "vector_index"
    ensure_index(root)
    (root / MANIFEST_FILE).write_text("{not json", encoding="utf-8")

    assert "unreadable" in unusable_reason(root)
    # ensure_index must recover rather than propagate a JSONDecodeError.
    assert ensure_index(root).manifest["count"] == 5


# ------------------------------------------------------ 3. unwritable location
def test_unwritable_index_dir_falls_back_to_a_temp_dir(tmp_path, monkeypatch, fake_corpus):
    """A read-only checkout must not fail a deploy.

    Forced with a path whose parent is a regular file, because that fails
    identically on Windows and Linux - a chmod-based read-only directory does
    not actually block writes on Windows.
    """
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path / "fallback"))

    index = ensure_index(blocker / "vector_index")

    assert index.root == tmp_path / "fallback" / "hdfc_mf_facts_vector_index"
    assert LocalIndex.exists(index.root)
    # The fallback has to be sticky, or the next load() would look in the
    # unwritable configured path and fail again.
    assert localindex.default_index_dir() == index.root
    assert LocalIndex.load().manifest["count"] == 5


def test_the_fallback_is_announced_in_the_log(tmp_path, monkeypatch, fake_corpus):
    """A silent relocation is how an operator ends up hunting for an index in a
    directory the app never wrote to."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path / "fallback"))

    lines: list[str] = []
    ensure_index(blocker / "vector_index", log=lines.append)

    assert any("not writable" in line for line in lines), lines


# ------------------------------------------------------------ 4. honest errors
def test_missing_corpus_raises_an_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "no_such_corpus")
    with pytest.raises(IndexError_) as excinfo:
        ensure_index(tmp_path / "vector_index")
    message = str(excinfo.value)
    assert "no_such_corpus" in message
    assert "data/raw" in message, "the error does not say where the corpus should be"
    assert not LocalIndex.exists(tmp_path / "vector_index"), "an index was written anyway"


def test_coverage_failure_raises_instead_of_writing_a_partial_index(
    tmp_path, monkeypatch, fake_corpus
):
    """A deploy that half-cloned the corpus must fail loudly, not serve a bot
    that cannot answer anything."""
    monkeypatch.setattr("app.ingest.check_coverage", lambda chunks: ["no chunks for scheme hdfc-elss"])

    with pytest.raises(IndexError_) as excinfo:
        build_from_raw(tmp_path / "vector_index")

    assert "hdfc-elss" in str(excinfo.value)
    assert not LocalIndex.exists(tmp_path / "vector_index"), (
        "an index was written despite the coverage gate failing"
    )


def test_an_unsearchable_index_is_rejected_rather_than_returned(
    tmp_path, monkeypatch, fake_corpus
):
    """The point of a self-healing build is that what it writes is usable. An
    index that is written but cannot answer a query is worse than no index,
    because the app would treat it as loaded."""
    monkeypatch.setattr(
        LocalIndex, "query", lambda self, q_vec, n: [{"chunk_id": "wrong", "score": 0.0}]
    )
    with pytest.raises(IndexError_) as excinfo:
        build_from_raw(tmp_path / "vector_index")
    assert "first chunk" in str(excinfo.value)


# --------------------------------------------------------------------- default
def test_default_index_dir_comes_from_config(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "from-config")
    localindex._ACTIVE_ROOT = None
    assert localindex.default_index_dir() == tmp_path / "from-config"
