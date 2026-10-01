"""Application configuration.

Every tunable in one place, per architecture.md §7. No module may hardcode a
model id, a threshold or a path; it imports from here so a single edit changes
behaviour everywhere.

Environment overrides go in `.env` (see `.env.example`). `summary()` masks the
API key so config can be printed in a demo or a log without leaking it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


# --- corpus identity ---------------------------------------------------------

CORPUS_VERSION: str = _env("CORPUS_VERSION", "2026-09-28.1")

#: The five schemes in scope, plus cross-scheme "general" material.
SCHEMES: dict[str, str] = {
    "hdfc-large-cap": "HDFC Large Cap Fund",
    "hdfc-flexi-cap": "HDFC Flexi Cap Fund",
    "hdfc-elss": "HDFC ELSS Tax Saver",
    "hdfc-small-cap": "HDFC Small Cap Fund",
    "hdfc-balanced-advantage": "HDFC Balanced Advantage Fund",
}

AMC_NAME: str = "HDFC AMC"

#: Query-term -> scheme slug. Phase 4's scheme filter matches on these, so a
#: user typing "elss" or "tax saver" is understood without naming the fund.
SCHEME_ALIASES: dict[str, str] = {
    "hdfc large cap": "hdfc-large-cap",
    "hdfc large cap fund": "hdfc-large-cap",
    "large cap": "hdfc-large-cap",
    "largecap": "hdfc-large-cap",
    "hdfc flexi cap": "hdfc-flexi-cap",
    "hdfc flexi cap fund": "hdfc-flexi-cap",
    "flexi cap": "hdfc-flexi-cap",
    "flexicap": "hdfc-flexi-cap",
    "hdfc elss": "hdfc-elss",
    "hdfc elss tax saver": "hdfc-elss",
    "elss": "hdfc-elss",
    "elss tax saver": "hdfc-elss",
    "tax saver": "hdfc-elss",
    "80c": "hdfc-elss",
    "hdfc small cap": "hdfc-small-cap",
    "hdfc small cap fund": "hdfc-small-cap",
    "small cap": "hdfc-small-cap",
    "smallcap": "hdfc-small-cap",
    "hdfc balanced advantage": "hdfc-balanced-advantage",
    "hdfc balanced advantage fund": "hdfc-balanced-advantage",
    "balanced advantage": "hdfc-balanced-advantage",
}

#: Aliases belonging to *other* AMCs. Phase 4 uses these to refuse a question
#: about a fund that is not in this corpus rather than returning a lookalike.
FOREIGN_SCHEME_HINTS: tuple[str, ...] = (
    "parag parivartan",
    "parivartan",
    "axis",
    "sbi",
    "icici",
    "kotak",
    "sundaram",
    "tata",
    "mirae",
    "canara",
    "baroda",
    "pnb",
    "idfc",
    "nippon",
    "quant",
    "ppf",
    "parag",
)

# --- chunking (values measured in Phase 2, see docs/chunking_strategy.md) ----

CHUNK_SIZE: int = _env_int("CHUNK_SIZE", 900)
TAIL_OVERLAP: int = _env_int("TAIL_OVERLAP", 150)
MIN_CHUNK: int = _env_int("MIN_CHUNK", 80)

#: A single unsplittable unit may exceed CHUNK_SIZE by this factor before the
#: chunker force-splits it. 1.6 x 900 = 1440, the tolerance in
#: tests/test_chunking.py test 6.
FALLBACK_TOLERANCE_FACTOR: float = _env_float("FALLBACK_TOLERANCE_FACTOR", 0.6)

# --- retrieval / generation --------------------------------------------------

TOP_K: int = _env_int("TOP_K", 5)
FETCH_K: int = _env_int("FETCH_K", 10)
SIM_FLOOR: float = _env_float("SIM_FLOOR", 0.30)
MAX_CONTEXT_CHARS: int = _env_int("MAX_CONTEXT_CHARS", 8000)

EMBED_MODEL: str = _env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_BATCH: int = _env_int("EMBED_BATCH", 64)

#: Attempts and base backoff for the one network call the app cannot avoid.
#:
#: A cold instance has no model cache, so the first embed is an ~87 MB download
#: from huggingface.co. That host sits behind Cloudflare and answers anonymous
#: callers past its quota with HTTP 429 and a "Just a moment..." HTML page
#: instead of the weights - see app/embedder.py, which spends these two values
#: turning that transient answer into a successful load. 4 attempts with a 5 s
#: base is 5+10+20 s of waiting, which outlasts the limit window in practice.
MODEL_LOAD_ATTEMPTS: int = _env_int("MODEL_LOAD_ATTEMPTS", 4)
MODEL_LOAD_BACKOFF: float = _env_float("MODEL_LOAD_BACKOFF", 5.0)

#: Read straight from the environment by huggingface_hub, not by this module.
#: Declared here so summary() can report whether one is configured. The value is
#: never logged, printed or interpolated into an error message - the token's only
#: job is to raise the hub rate limit that turns a cold start into a 429.
HF_TOKEN: str = _env("HF_TOKEN", "")

GROQ_API_KEY: str = _env("GROQ_API_KEY", "")
GROQ_MODEL: str = _env("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_TEMPERATURE: float = _env_float("GROQ_TEMPERATURE", 0.0)

#: Shown on every refusal (architecture.md §5.3 FR-3.3: "a deterministic message
#: naming what is not covered plus the general HDFC MF link"). Not a citation -
#: it is never a source for a fact, only a route to the AMC.
HDFC_MF_LINK: str = "https://www.hdfcmf.com"

# --- storage -----------------------------------------------------------------

CHROMA_PATH: Path = Path(_env("CHROMA_PATH", str(REPO_ROOT / "chroma_db")))
CHROMA_COLLECTION: str = _env("CHROMA_COLLECTION", f"mf_facts_{CORPUS_VERSION}")

#: Cosine on L2-normalised embeddings makes distance a relevance score. A silent
#: L2 fallback invalidates SIM_FLOOR, so app/store.py asserts this at startup.
CHROMA_METADATA: dict[str, str] = {"hnsw:space": "cosine"}

# --- paths -------------------------------------------------------------------

RAW_DIR: Path = Path(_env("RAW_DIR", str(REPO_ROOT / "data" / "raw")))
PROCESSED_DIR: Path = Path(_env("PROCESSED_DIR", str(REPO_ROOT / "data" / "processed")))

#: Where the NumPy vector index lives. A build artifact, so it is gitignored
#: and absent from any fresh checkout - including a Render deploy, whose
#: filesystem is ephemeral and starts empty. Overridable so a deployment with a
#: read-only checkout can point it at a writable volume instead; app/localindex
#: falls back to a temp dir on its own if this one is not writable.
INDEX_DIR: Path = Path(_env("INDEX_DIR", str(PROCESSED_DIR / "vector_index")))
LOGS_DIR: Path = Path(_env("LOGS_DIR", str(REPO_ROOT / "data" / "logs")))
CHUNKS_TXT: Path = PROCESSED_DIR / "chunks.txt"
INGEST_REPORT: Path = PROCESSED_DIR / "ingest_report.json"

# --- retrieval term boosts (Phase 4 uses these in the lexical re-rank) --------

METRIC_TERMS: tuple[str, ...] = (
    "expense ratio",
    "exit load",
    "sip",
    "lock-in",
    "lock in",
    "riskometer",
    "benchmark",
    "nav",
    "aum",
    "inception",
    "fund manager",
    "minimum",
    "direct plan",
    "regular plan",
    "statement",
    "consolidated account statement",
    "cas",
    "capital gains",
    "redemption",
)

#: Source types the coverage check requires, per implementation.md Phase 3.
#:
#: implementation.md phrases the fee requirement as ">=1 `fees` source". This
#: corpus has no document literally typed `fees`: fees live in the two real HDFC
#: document families - the factsheet (expense ratio, exit load) and the KIM
#: (charges, minimum investment). Requiring a type that no source declares
#: would make the gate unpassable, so the requirement is expressed as the
#: families that actually carry fees. The gate's purpose is preserved: a run
#: that lost the factsheets or the KIMs must fail loudly rather than ship a bot
#: that cannot answer "what is the expense ratio".
FEE_BEARING_SOURCE_TYPES: tuple[str, ...] = ("factsheet", "kim")

#: Sources the bot needs to answer process/how-it-works questions.
GUIDE_SOURCE_TYPES: tuple[str, ...] = ("guide",)

REQUIRED_SOURCE_TYPES: tuple[str, ...] = FEE_BEARING_SOURCE_TYPES + GUIDE_SOURCE_TYPES


def summary() -> str:
    """Printable config with the API key masked."""
    key = GROQ_API_KEY
    masked = f"{key[:4]}...{key[-4:]}" if len(key) > 10 else ("<unset>" if not key else "***")
    lines = [
        f"corpus_version   : {CORPUS_VERSION}",
        f"collection       : {CHROMA_COLLECTION}",
        f"chroma_path      : {CHROMA_PATH}",
        f"hnsw:space       : {CHROMA_METADATA['hnsw:space']}",
        f"embed_model      : {EMBED_MODEL}",
        f"embed_batch      : {EMBED_BATCH}",
        f"model_load       : {MODEL_LOAD_ATTEMPTS} attempts, {MODEL_LOAD_BACKOFF}s base backoff",
        f"hf_token         : {'<set>' if HF_TOKEN else '<unset>'}",
        f"chunk_size       : {CHUNK_SIZE}",
        f"tail_overlap     : {TAIL_OVERLAP}",
        f"top_k / fetch_k  : {TOP_K} / {FETCH_K}",
        f"sim_floor        : {SIM_FLOOR}",
        f"max_context      : {MAX_CONTEXT_CHARS}",
        f"groq_model       : {GROQ_MODEL} (temp {GROQ_TEMPERATURE})",
        f"groq_api_key     : {masked}",
        f"raw_dir          : {RAW_DIR}",
        f"index_dir        : {INDEX_DIR}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
