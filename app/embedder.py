"""Shared SentenceTransformer instance.

One model, loaded once per process, used by BOTH ingest and query. Two separate
loaders is how a model id silently drifts between writing and reading, which
produces an index that cannot be searched. `get_model` is lru_cache'd so the
~90 MB MiniLM loads once no matter how many batches are embedded.

Never hardcode a model id here; the value comes from config.EMBED_MODEL.

Loading used to be the app's one unavoidable network call, and on a fresh deploy it
is where a healthy service died. Serving a question needs this model in the process
(app/retrieve calls embed_query on every question), a Render instance's filesystem
starts empty, and huggingface.co is Cloudflare-fronted: past its quota it replies
with HTTP 429 and a "Just a moment..." interstitial rather than the weights. The
service stayed LIVE throughout, because the failure was in a dependency and not in
the app, so the symptom was a chat UI that loads and then reports a connection
error while warming up, with a page of HTML as the message.

So the weights are now committed under `models/` (see .gitignore) and EMBED_MODEL
points at that directory, which makes the default load path purely local and the
429 unreachable. Everything below it is kept for the case where EMBED_MODEL is
deliberately set to a hub id: the retry with backoff, and the one actionable
sentence. The markup is that bug's real face, so it is never passed on.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from functools import lru_cache

from app import config


class EmbedderError(RuntimeError):
    pass


#: Statuses worth another attempt. 429 is the rate limit huggingface_hub already
#: retries internally (file_download._DEFAULT_RETRY_ON_STATUS_CODES is 408 and
#: 429), so what reaches this module are the failures that outlived that. 5xx is
#: the same class of transient problem. 401/403/404 are deliberately absent: a
#: rejected token or a wrong model id will not fix itself, and retrying only
#: delays the error that actually explains the failure.
_RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

#: Text that identifies a rate limit even when it arrives without a usable
#: status. A Cloudflare interstitial reaches a caller as an exception whose body
#: is HTML, so the body is the only evidence in some paths - and it is the part
#: the user was staring at.
_RATE_LIMIT_MARKERS = (
    "just a moment",
    "cf-chl",
    "attention required",
    "rate limit",
    "too many requests",
    "429",
)

#: Exception class names that mean "the network moved under us". These are the
#: requests/urllib3 names the hub is built on, plus its own transfer backend.
#: OfflineModeIsEnabled is intentionally not here: that configuration will fail
#: identically forever, and retrying it would only bury the reason.
_TRANSIENT_NAMES = frozenset({
    "ReadTimeout",
    "ConnectTimeout",
    "Timeout",
    "ConnectionError",
    "ProxyError",
    "SSLError",
    "ChunkedEncodingError",
    "XetDownloadError",
})


def _status_of(exc: BaseException) -> int | None:
    """The HTTP status behind `exc`, if it carries one.

    huggingface_hub raises HfHubHTTPError, an OSError carrying a requests
    Response, so the status sits on `exc.response.status_code` rather than on
    `exc` itself. Both spellings are accepted so this behaves the same on an
    error that was already normalised by another library.
    """
    for holder in (exc, getattr(exc, "response", None)):
        status = getattr(holder, "status_code", None)
        if isinstance(status, int):
            return status
    return None


def _is_transient(exc: BaseException) -> bool:
    """True when the same call is worth repeating a moment later.

    The same question `generate.is_transient` answers for the Groq SDK, asked
    again for huggingface_hub, and the duplication is deliberate: generate
    imports retrieve which imports this module, so sharing the helper would be
    a cycle. The two agree on the cause chain, the class names and the 4xx rule,
    and differ only where the hub differs - the status lives on `response`, and
    a rate limit can arrive as a body with no status attached at all.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))

        status = _status_of(current)
        if status is not None:
            if status in _RETRY_STATUS:
                return True
            if 400 <= status < 500:
                return False

        if isinstance(current, (TimeoutError, ConnectionError)):
            return True
        if type(current).__name__ in _TRANSIENT_NAMES:
            return True

        current = current.__cause__ or current.__context__

    # Last resort: a Cloudflare challenge page is an HTML body wrapped in an
    # exception of some class, and the words in it are the only signal left.
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def _clean(exc: BaseException, limit: int = 240) -> str:
    """One short line describing `exc`, with markup removed.

    The thing being described is an HTML page, and `str(exc)` is that page. It
    cannot simply be quoted: stripping the tags still leaves the page's prose,
    and a Cloudflare interstitial's prose is a support form, not an explanation.

    So a body containing markup is treated as what it is - a rendered response -
    and reduced to the line in front of the first tag, which is where requests
    puts the status and the URL, plus a note that the rest was markup. A plain
    text body is kept, with entities and line breaks normalised, and truncated
    so a failure message fits on a screen.
    """
    raw = f"{type(exc).__name__}: {exc}"
    if re.search(r"<[a-zA-Z!/]", raw):
        head = " ".join(re.split(r"<[a-zA-Z!/]", raw, maxsplit=1)[0].split())
        if len(head) > limit:
            head = head[:limit] + "..."
        return f"{head} (HTML error page, {len(raw)} chars of markup elided)"
    text = re.sub(r"&(?:#\d+|#x[0-9a-f]+|[a-z]+);", " ", raw, flags=re.I)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


def _load_failure(exc: BaseException | None) -> str:
    """The message a caller sees when the model will not load.

    Names the cause a cold deploy actually hits and the two settings that fix
    it, and says whether a token is present without ever including one.
    """
    token = "a token is set" if config.HF_TOKEN else "no token is set"
    return (
        f"Could not load the embedding model {config.EMBED_MODEL!r} after "
        f"{config.MODEL_LOAD_ATTEMPTS} attempts. "
        f"Last error: {_clean(exc) if exc else 'unknown'}\n"
        "With the default vendored weights this is an incomplete checkout - run "
        "python -m scripts.vendor_model.\n"
        "If EMBED_MODEL is set to a hub id instead, the model is downloaded from "
        "huggingface.co, which is Cloudflare-fronted and rate-limits callers with "
        "an HTTP 429 \"Just a moment...\" page, so that is the likely cause.\n"
        "Fixes: point EMBED_MODEL at a local directory (recommended); or set "
        f"HF_TOKEN to a Hugging Face read token ({token}); or raise "
        f"MODEL_LOAD_BACKOFF to wait longer between attempts."
    )


def _load(load: Callable[[], object], log: Callable[[str], None] = print) -> object:
    """Run `load`, retrying transient failures with exponential backoff.

    The first attempt failing is normal on a cold instance and says nothing on
    its own, so the notice goes to the log rather than to the user. A failure
    that is not transient is reported immediately, because waiting cannot help.
    """
    attempts = max(1, config.MODEL_LOAD_ATTEMPTS)
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return load()
        except EmbedderError:
            raise
        except Exception as exc:
            if not _is_transient(exc):
                raise EmbedderError(_load_failure(exc)) from exc
            last = exc
            if attempt == attempts:
                break
            delay = config.MODEL_LOAD_BACKOFF * (2 ** (attempt - 1))
            log(
                f"model load attempt {attempt}/{attempts} failed "
                f"({_clean(exc, 120)}); retrying in {delay:.0f}s"
            )
            time.sleep(delay)
    raise EmbedderError(_load_failure(last)) from last


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

    # Resolved here rather than passed as config.EMBED_MODEL: that value is the
    # model's identity and is recorded verbatim in the index manifest, so it is a
    # relative path on a POSIX-slash spelling that happens to be loadable. The
    # working directory of a Streamlit process is not the repo root, so the
    # absolute path has to be derived.
    try:
        source = config.embed_model_source()
    except FileNotFoundError as exc:
        # Not retryable: no amount of waiting puts a missing directory back.
        raise EmbedderError(str(exc)) from exc

    model = _load(lambda: SentenceTransformer(source))
    if config.EMBED_FP16:
        _to_fp16(model)
    return model


def _to_fp16(model):
    """Halve the model weights in place, once per process.

    Half precision is the only lever measured to bring the serving floor inside
    a Render free instance: resident 489 MB -> 453 MB. int8 dynamic quantisation
    was measured at 600 MB, worse than doing nothing, because fbgemm packs its
    integers into wider buffers than the floats they replace.

    The cast is skipped when the model is already half, so calling it per batch
    costs nothing after the first. It is deliberately not wrapped in a
    try/except: a model that silently stayed fp32 would put the instance back
    over the limit while looking healthy in the logs, which is worse than a
    loud failure.
    """
    import torch

    try:
        current = next(model.parameters()).dtype
    except StopIteration:  # pragma: no cover - a model with no parameters
        return
    if current is torch.float16:
        return
    model.half()
    return model


def embed_texts(texts: list[str], batch_size: int | None = None) -> list[list[float]]:
    """Embed with L2 normalisation, so cosine distance is a relevance score.

    `normalize_embeddings=True` is what makes `SIM_FLOOR` meaningful - without
    it, cosine and length get entangled.
    """
    if not texts:
        return []
    # asarray rather than .tolist() on the assumption it is always an array: a
    # test double - or any caller that swaps in its own embedder - may hand back
    # a plain list of lists, and that should not read as a numpy bug.
    import numpy as np

    return np.asarray(embed_texts_np(texts, batch_size=batch_size), dtype="float32").tolist()


def embed_texts_np(texts: list[str], batch_size: int | None = None):
    """Embed and return a float32 numpy array rather than Python lists.

    720 chunks x 384 dims as `list[list[float]]` is 276,480 Python float objects,
    each with its own allocation and pointer - several times the 1.1 MB the same
    values occupy as one contiguous array. The index build held both that and the
    numpy matrix at once.

    `inference_mode` is the guard that matters here: without it torch builds an
    autograd graph per batch, holding every intermediate activation alive until
    the result is consumed. Measured neutral on a single short question, but the
    build embeds 720 chunks through this path and that is where the transient
    activations are large.
    """
    import numpy as np
    import torch

    if not texts:
        return np.zeros((0, 0), dtype="float32")

    model = get_model()
    with torch.inference_mode():
        vectors = model.encode(
            texts,
            batch_size=batch_size or config.EMBED_BATCH,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    # float32 regardless of the model's dtype: half precision halves the resident
    # model but would halve the stored vectors too, and float16 has ~3 decimal
    # digits - not enough to keep SIM_FLOOR stable across a rebuild.
    return np.asarray(vectors, dtype="float32")


def embed_query(text: str) -> list[float]:
    """Embed a single question, normalised identically to the corpus."""
    return embed_texts([text])[0]
