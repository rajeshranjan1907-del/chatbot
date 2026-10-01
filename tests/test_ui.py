"""UI tests - the Render failure mode, at the layer it actually broke.

The deployment error was not a retrieval bug: a Render instance starts with the
corpus and no `data/processed/vector_index`, so the first thing `_warm()` did was
raise, and the user got a traceback instead of a bot. These tests drive the real
Streamlit script in-process with `AppTest` and assert the two states that matter:

  1. No index on disk -> the app builds it on first load and then answers, with
     a citation, exactly as it would on a cold Render instance.
  2. No index AND no corpus -> the app explains that instead of raising.

The chunker and the embedder are faked, so nothing here downloads the model or
depends on the developer's local index. `GROQ_API_KEY` is cleared so the answer
takes the deterministic no-LLM fallback: the point here is that the UI renders
a grounded answer, not that a particular model was called.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from streamlit.testing.v1 import AppTest

from app import config
from app.chunking import Chunk

UI = str(REPO / "app" / "ui.py")
DIM = 8


def _chunk(i: int) -> Chunk:
    return Chunk(
        chunk_id=f"{config.CORPUS_VERSION}:test-factsheet:{i:04d}",
        text=(
            f"EXPENSE RATIO of HDFC Large Cap Fund (Direct): 0.65% p.a. "
            f"plus statutory levies. Benchmark NIFTY 500 Index (TRI). Part {i}."
        ),
        embed_text=f"expense ratio hdfc large cap {i}",
        scheme="HDFC Large Cap Fund",
        scheme_slug="hdfc-large-cap",
        amc=config.AMC_NAME,
        category="large-cap",
        source_type="factsheet",
        source_title="HDFC MF Factsheet - June 2026",
        source_url="https://files.hdfcfund.com/example.pdf",
        section="EXPENSE RATIO",
        chunk_index=i,
        as_of_date="2026-06-30",
        fetched_at="2026-09-28",
        corpus_version=config.CORPUS_VERSION,
        pages="1",
        text_quality="clean",
    )


@pytest.fixture
def deploy(monkeypatch, tmp_path):
    """A clean instance: no index, no model call, no network.

    `st.cache_resource` is a process-global cache keyed on the function's code,
    not on the config it happened to read, so it is cleared around each test.
    Without that, the index built by the first test is handed to the next one
    and the missing-index path is never actually exercised.
    """
    chunks = [_chunk(i) for i in range(4)]

    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "vector_index")
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    monkeypatch.setattr("app.chunking.load_and_chunk", lambda *a, **k: chunks)
    monkeypatch.setattr("app.ingest.check_coverage", lambda c: [])

    def fake_embed_texts(texts, *args, **kwargs):
        vectors = []
        for i in range(len(texts)):
            v = [0.0] * DIM
            v[i % DIM] = 1.0
            vectors.append(v)
        return vectors

    # The index build calls embed_texts_np; see the same note in test_bootstrap.
    monkeypatch.setattr("app.embedder.embed_texts_np", fake_embed_texts)

    st.cache_resource.clear()
    yield tmp_path / "vector_index"
    st.cache_resource.clear()


def test_app_loads_when_the_index_is_missing(deploy):
    """The Render case. No exception, and the index is built on the way."""
    at = AppTest.from_file(UI).run(timeout=300)

    assert not at.exception, [str(e.value) for e in at.exception]
    assert deploy.exists(), "the app did not build the missing index"
    assert at.title[0].value == "HDFC Mutual Fund Facts"


def test_the_build_is_visible_to_the_user(deploy):
    """A silent ~1 minute embed on a cold start reads as a hung app."""
    at = AppTest.from_file(UI).run(timeout=300)

    labels = [e.label for e in at.expander]
    assert any("Index log" in label for label in labels), labels
    log = next(e for e in at.expander if "Index log" in e.label)
    body = log.code[0].value if hasattr(log, "code") else ""
    assert "building the vector index" in body, body


def test_a_question_is_answered_with_a_citation_after_the_build(deploy):
    """The point of the fix: the deployment can answer, not just start."""
    at = AppTest.from_file(UI).run(timeout=300)
    at.chat_input[0].set_value("what is the expense ratio of HDFC Large Cap Fund?").run(timeout=300)

    assert not at.exception, [str(e.value) for e in at.exception]
    answers = [m.markdown[0].value for m in at.chat_message if m.name == "assistant"]
    assert answers, "no assistant message was rendered"
    assert "0.65%" in answers[0], answers[0]

    links = [m.markdown[1].value for m in at.chat_message if m.name == "assistant"]
    rendered = " ".join(links)
    assert "https://files.hdfcfund.com/example.pdf" in rendered, rendered


def test_a_missing_corpus_is_explained_not_raised(monkeypatch, deploy):
    """No index and no corpus is unrecoverable, so the user is told why in
    words instead of being handed a traceback."""
    monkeypatch.setattr(config, "RAW_DIR", deploy.parent / "absent_corpus")
    at = AppTest.from_file(UI).run(timeout=300)

    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.error, "the app failed without telling the user"
    detail = "\n".join(e.value for e in at.code)
    assert "absent_corpus" in detail, (
        f"the user is not told which corpus is missing: {detail}"
    )
    assert not deploy.exists(), "an index was written with no corpus behind it"


def test_a_model_download_failure_is_explained_not_raised(monkeypatch, deploy):
    """A blocked Hugging Face request is a deploy problem, not a user error, and
    it must not surface as an unhandled exception either."""
    def boom(*_args, **_kwargs):
        raise RuntimeError("could not reach huggingface.co")

    monkeypatch.setattr("app.embedder.embed_texts_np", boom)
    at = AppTest.from_file(UI).run(timeout=300)

    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.error, "the app failed without telling the user"
    detail = "\n".join(e.value for e in at.code)
    assert "huggingface.co" in detail, f"the reason is not shown to the user: {detail}"
    assert "RuntimeError" in detail, f"the exception type is not named: {detail}"
