"""Model-load failure tests - the cold-deploy 429.

The model is MOCKED everywhere, like test_generate.py mocks the LLM: a test that
really downloaded 87 MB would be testing huggingface.co's mood, not this code.

What is under test is the decision table around a load that cannot complete
immediately, which is the state a fresh Render instance is in every time it
spins up. The index is a gitignored artifact, so there are no vectors, so the
first embed is a cold ~87 MB download from a host that answers anonymous
callers past its quota with an HTTP 429 "Just a moment..." page. The service
stays LIVE throughout, and the user sees a page of HTML where a sentence should
be. So: retry what is worth retrying, refuse to retry what is not, and never let
the markup through.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app import config, embedder
from app.embedder import EmbedderError, _clean, _is_transient, _load

#: A cut-down copy of the interstitial Cloudflare returns for a throttled
#: anonymous download. Real structure, trimmed to the lines that matter.
CHALLENGE_HTML = (
    "<!DOCTYPE html><html><head><title>Just a moment...</title></head>"
    "<body><div class='main-wrapper'><h1>Checking your browser before "
    "accessing huggingface.co</h1><p>Please enable JS and disable any ad "
    "blocker.</p><p>Ray ID: 8f2c1a0b3d4e5f60</p></div></body></html>"
)


class FakeResponse:
    """The bit of a requests Response that the status lookup reads."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class HubError(Exception):
    """Stands in for HfHubHTTPError: an OSError carrying a Response."""

    def __init__(self, status_code: int, body: str = "") -> None:
        super().__init__(f"{status_code} for url: https://huggingface.co/x: {body}")
        self.response = FakeResponse(status_code)


def _scripted(*outcomes):
    """A loader replaying `outcomes`, counting the calls it received."""
    calls: list[int] = []

    def load():
        calls.append(1)
        outcome = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    load.calls = calls  # type: ignore[attr-defined]
    return load


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    """Keep the backoff instant so a retry test costs milliseconds, not minutes.

    Also restores the real tunables per test, so a test that patches them cannot
    leak a 30 s sleep or an attempts count into the next one.
    """
    monkeypatch.setattr(config, "MODEL_LOAD_BACKOFF", 0.0)
    monkeypatch.setattr(config, "MODEL_LOAD_ATTEMPTS", 4)
    monkeypatch.setattr(config, "HF_TOKEN", "")
    yield
    for name in ("MODEL_LOAD_BACKOFF", "MODEL_LOAD_ATTEMPTS", "HF_TOKEN"):
        monkeypatch.undo()


# --- what counts as worth retrying -------------------------------------------


def test_429_is_transient():
    assert _is_transient(HubError(429))


def test_5xx_is_transient():
    assert _is_transient(HubError(503))


def test_cloudflare_page_with_no_status_is_transient():
    """The case that motivated this: the 429 arrives as a body, not a status."""
    assert _is_transient(ConnectionError(CHALLENGE_HTML))


def test_timeout_is_transient():
    assert _is_transient(TimeoutError("timed out"))


def test_401_is_not_transient():
    """A bad token fails the same way every time; retrying only wastes startup."""
    assert not _is_transient(HubError(401))


def test_404_is_not_transient():
    assert not _is_transient(HubError(404))


def test_403_is_not_transient():
    assert not _is_transient(HubError(403))


def test_wrapped_cause_is_still_classified():
    """The hub and requests both wrap, so the status is often one link down."""
    wrapped = RuntimeError("while resolving model")
    wrapped.__cause__ = HubError(429)
    assert _is_transient(wrapped)


def test_cause_chain_terminates_on_a_cycle():
    """A self-referential __cause__ must not hang the classifier."""
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert not _is_transient(a)


# --- the retry loop ----------------------------------------------------------


def test_first_attempt_succeeds_without_retrying():
    load = _scripted("model")
    assert _load(load, log=lambda _m: None) == "model"
    assert len(load.calls) == 1


def test_429_then_success_returns_the_model():
    load = _scripted(HubError(429, CHALLENGE_HTML), "model")
    assert _load(load, log=lambda _m: None) == "model"
    assert len(load.calls) == 2


def test_retries_are_bounded_by_the_configured_attempts(monkeypatch):
    monkeypatch.setattr(config, "MODEL_LOAD_ATTEMPTS", 3)
    load = _scripted(HubError(429))
    with pytest.raises(EmbedderError):
        _load(load, log=lambda _m: None)
    assert len(load.calls) == 3


def test_a_persistent_429_ends_as_an_embedder_error():
    load = _scripted(HubError(429, CHALLENGE_HTML))
    with pytest.raises(EmbedderError) as caught:
        _load(load, log=lambda _m: None)
    assert "huggingface.co" in str(caught.value)
    assert "HF_TOKEN" in str(caught.value)


def test_a_non_transient_failure_is_not_retried():
    load = _scripted(HubError(404))
    with pytest.raises(EmbedderError):
        _load(load, log=lambda _m: None)
    assert len(load.calls) == 1


def test_backoff_grows_between_attempts(monkeypatch):
    """Each wait is double the last, so a slow host gets progressively room."""
    monkeypatch.setattr(config, "MODEL_LOAD_BACKOFF", 0.5)
    monkeypatch.setattr(config, "MODEL_LOAD_ATTEMPTS", 4)
    slept: list[float] = []
    real_sleep = time.sleep
    try:
        time.sleep = slept.append  # type: ignore[assignment]
        with pytest.raises(EmbedderError):
            _load(_scripted(HubError(429)), log=lambda _m: None)
    finally:
        time.sleep = real_sleep  # type: ignore[assignment]
    assert slept == [0.5, 1.0, 2.0]


def test_retry_notices_go_to_the_log_not_the_error():
    notices: list[str] = []
    load = _scripted(HubError(429), HubError(429), "model")
    _load(load, log=notices.append)
    assert len(notices) == 2
    assert all("attempt 1/4" in n or "attempt 2/4" in n for n in notices)
    assert all(n.endswith("retrying in 0s") for n in notices)


# --- the message that reaches the user ---------------------------------------


def test_the_cloudflare_page_never_reaches_the_message():
    """The reported bug was a page of HTML rendered as a connection error.

    Naming the challenge page in the guidance is the point - it tells the user
    what they are looking at - so the invariant is that none of the page's
    structure comes with it: no tags, no class names, no Ray ID, and short.
    """
    with pytest.raises(EmbedderError) as caught:
        _load(_scripted(HubError(429, CHALLENGE_HTML)), log=lambda _m: None)
    message = str(caught.value)
    assert "<" not in message
    assert ">" not in message
    assert "DOCTYPE" not in message
    assert "cf-chl" not in message
    assert "main-wrapper" not in message
    assert "8f2c1a0b3d4e5f60" not in message  # the fixture's unique Ray ID
    assert len(message) < 700  # fits a chat window without scrolling


def test_clean_keeps_the_status_and_url_from_a_rendered_page():
    """A page is not an error description: keep the request line, drop the body."""
    cleaned = _clean(HubError(429, CHALLENGE_HTML))
    assert "429" in cleaned
    assert "huggingface.co" in cleaned
    assert "HTML error page" in cleaned
    assert "8f2c1a0b3d4e5f60" not in cleaned  # the page's Ray ID is noise
    assert "main-wrapper" not in cleaned
    assert "<" not in cleaned


def test_clean_normalises_a_plain_text_error():
    cleaned = _clean(Exception("line one\nline two&#160;end"))
    assert cleaned == "Exception: line one line two end"


def test_clean_decodes_entities_in_a_plain_text_error():
    assert "&#" not in _clean(Exception("a&nbsp;b&#160;c"))


def test_clean_truncates_a_very_long_body():
    cleaned = _clean(Exception("x" * 5000), limit=50)
    assert len(cleaned) == 53
    assert cleaned.endswith("...")


def test_the_token_value_is_never_in_the_message(monkeypatch):
    """Whichever way the load fails, a secret must not ride along in the text."""
    monkeypatch.setattr(config, "HF_TOKEN", "hf_supersecrettoken")
    with pytest.raises(EmbedderError) as caught:
        _load(_scripted(HubError(429, CHALLENGE_HTML)), log=lambda _m: None)
    assert "hf_supersecrettoken" not in str(caught.value)


def test_the_message_says_whether_a_token_is_set(monkeypatch):
    monkeypatch.setattr(config, "HF_TOKEN", "hf_supersecrettoken")
    with pytest.raises(EmbedderError) as with_token:
        _load(_scripted(HubError(429)), log=lambda _m: None)
    monkeypatch.setattr(config, "HF_TOKEN", "")
    with pytest.raises(EmbedderError) as without_token:
        _load(_scripted(HubError(429)), log=lambda _m: None)
    assert "a token is set" in str(with_token.value)
    assert "no token is set" in str(without_token.value)


def test_config_summary_reports_token_presence_but_not_its_value(monkeypatch):
    monkeypatch.setattr(config, "HF_TOKEN", "hf_supersecrettoken")
    summary = config.summary()
    assert "hf_supersecrettoken" not in summary
    assert "hf_token         : <set>" in summary
    monkeypatch.setattr(config, "HF_TOKEN", "")
    assert "hf_token         : <unset>" in config.summary()


def test_get_model_rejects_a_missing_dependency(monkeypatch):
    """A missing package is not a network problem and must not be retried."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("No module named 'sentence_transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    embedder.get_model.cache_clear()
    try:
        with pytest.raises(EmbedderError) as caught:
            embedder.get_model()
        assert "sentence-transformers is not installed" in str(caught.value)
    finally:
        embedder.get_model.cache_clear()


def test_get_model_retries_a_429_from_the_hub(monkeypatch):
    """End to end through get_model: the constructor is the thing that 429s."""
    import sentence_transformers

    attempts: list[int] = []

    class Flaky:
        def __init__(self, model_id):
            attempts.append(model_id)
            if len(attempts) == 1:
                raise HubError(429, CHALLENGE_HTML)

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", Flaky)
    embedder.get_model.cache_clear()
    try:
        model = embedder.get_model()
        assert isinstance(model, Flaky)
        assert len(attempts) == 2
        assert attempts[0] == config.EMBED_MODEL
    finally:
        embedder.get_model.cache_clear()
