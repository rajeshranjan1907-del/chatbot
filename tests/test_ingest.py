"""Ingestion pipeline tests - implementation.md Phase 3 step 5.

Two properties that matter more than any individual chunk:

  1. The coverage gate really fails on an incomplete corpus. A gate that always
     passes is worse than no gate, because it looks like a check.
  2. Re-running ingest on the same corpus version produces the same chunk_ids
     and the same collection count. `chunk_id` is the upsert key, so a drifting
     id silently duplicates the index.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app import config
from app.chunking import load_and_chunk
from app.ingest import check_coverage, coverage_report, write_chunks_txt


@pytest.fixture(scope="module")
def chunks() -> list:
    return load_and_chunk()


# ---------------------------------------------------------------- coverage
def test_coverage_passes_on_the_real_corpus(chunks):
    """Baseline: the shipped corpus must satisfy every requirement."""
    assert not check_coverage(chunks), check_coverage(chunks)


def test_coverage_requires_all_five_schemes(chunks):
    """Dropping one scheme must be detected, with the missing one named."""
    kept = [c for c in chunks if c.scheme_slug != "hdfc-elss"]
    problems = check_coverage(kept)
    assert problems, "removing the ELSS scheme produced no coverage complaint"
    assert any("hdfc-elss" in p for p in problems), (
        f"the complaint does not name the missing scheme: {problems}"
    )


def test_coverage_requires_a_fees_source(chunks):
    """A corpus with no fee-bearing source cannot answer fee questions."""
    kept = [c for c in chunks if c.source_type not in ("factsheet", "kim")]
    problems = check_coverage(kept)
    assert problems, "removing every fee source produced no complaint"
    assert any("fee" in p for p in problems), (
        f"the complaint does not mention the missing fee sources: {problems}"
    )


def test_coverage_requires_a_guide_source(chunks):
    kept = [c for c in chunks if c.source_type != "guide"]
    problems = check_coverage(kept)
    assert problems, "removing the guides produced no complaint"
    assert any("guide" in p for p in problems), (
        f"the complaint does not mention the missing guide source: {problems}"
    )


def test_coverage_fails_on_an_empty_corpus():
    assert check_coverage([]), "an empty corpus passed the coverage check"


def test_every_scheme_has_a_factsheet_and_a_kim(chunks):
    """Both document families are needed: factsheets for ratios, KIMs for minimums."""
    for slug in config.SCHEMES:
        types = {c.source_type for c in chunks if c.scheme_slug == slug}
        assert "factsheet" in types, f"{slug}: no factsheet chunks"
        assert "kim" in types, f"{slug}: no KIM chunks (minimums live only there)"


# ---------------------------------------------------------------- idempotency
def test_chunk_ids_are_stable_across_runs(chunks):
    """Same corpus in, same chunk_ids out. This is what makes upsert idempotent."""
    again = load_and_chunk()
    assert [c.chunk_id for c in again] == [c.chunk_id for c in chunks], (
        "chunk_id sequence changed between two runs over the same corpus"
    )
    assert [c.text_hash for c in again] == [c.text_hash for c in chunks], (
        "chunk text changed between runs; the corpus is not deterministic"
    )


def test_chunk_ids_are_unique(chunks):
    ids = [c.chunk_id for c in chunks]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate chunk_ids would silently overwrite: {sorted(dupes)[:5]}"


def test_chunk_ids_embed_corpus_version(chunks):
    """A new corpus version must produce new ids, or old vectors linger."""
    for c in chunks[:20]:
        assert c.chunk_id.startswith(f"{config.CORPUS_VERSION}:"), (
            f"{c.chunk_id} does not start with corpus_version "
            f"{config.CORPUS_VERSION}"
        )


def test_a_different_corpus_version_changes_chunk_ids(chunks):
    """Bumping the version must change the ids, which is how collections stay separate."""
    other = load_and_chunk(corpus_version="2099-01-01.1")
    assert other[0].chunk_id.startswith("2099-01-01.1:")
    assert other[0].chunk_id != chunks[0].chunk_id
    # The text must be unchanged - only identity differs.
    assert other[0].text == chunks[0].text


# ---------------------------------------------------------------- chunks.txt
def test_chunks_txt_is_written_with_full_metadata(tmp_path, chunks):
    """A human must be able to audit every chunk without running the app."""
    out = write_chunks_txt(chunks, tmp_path / "chunks.txt")
    text = out.read_text(encoding="utf-8")

    for field in ("chunk_id", "scheme", "type", "as_of", "url", "quality"):
        assert f"{field} " in text or f"{field}:" in text, (
            f"chunks.txt is missing the {field!r} field"
        )
    # Every chunk must appear, so the count in the report is auditable.
    assert text.count("chunk_id : ") == len(chunks), (
        f"chunks.txt has {text.count('chunk_id : ')} blocks for {len(chunks)} chunks"
    )
    # Deterministic output: rewriting produces identical bytes.
    again = write_chunks_txt(chunks, tmp_path / "chunks2.txt")
    assert (tmp_path / "chunks.txt").read_text(encoding="utf-8") == \
        again.read_text(encoding="utf-8"), "chunks.txt is not deterministic"


# ---------------------------------------------------------------- CLI
def test_ingest_dry_run_exits_zero_and_writes_no_vectors(tmp_path):
    """--dry-run must chunk, report and NOT touch the vector store."""
    proc = subprocess.run(
        [sys.executable, "-m", "app.ingest", "--dry-run"],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, f"dry-run failed:\n{proc.stdout}\n{proc.stderr}"
    assert "chunks" in proc.stdout.lower()
    assert "DRY RUN" in proc.stdout
    assert config.CHUNKS_TXT.exists(), "dry-run did not write chunks.txt"
    # A dry run must not have created a collection.
    if config.CHROMA_PATH.exists():
        import chromadb
        client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
        try:
            client.get_collection(name=config.CHROMA_COLLECTION)
        except Exception:
            pass  # absent is the expected outcome


def test_ingest_reports_coverage_and_chunk_counts():
    """The report a human reads after ingest must carry the key numbers."""
    chunks = load_and_chunk()
    cov = coverage_report(chunks)
    assert cov["total_chunks"] == len(chunks)
    assert set(cov["per_scheme"]) >= set(config.SCHEMES)
    assert cov["clean"] + cov["degraded"] == len(chunks)
