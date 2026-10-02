# Deploying to Render

The bot answers questions about HDFC mutual fund facts from a local NumPy vector
index. That index is **committed to the repository** at
`data/processed/vector_index`, so a Render instance loads it instead of building
it. Nothing about startup needs the corpus, a model, or write access to
`data/processed`.

This is a change from the previous revision of this document, which described
building the index on every cold start. That approach was measured and it does
not work on the free tier - see "Memory" below.

## What happens at startup

```
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
exec streamlit run app/ui.py --server.address 0.0.0.0 --server.port $PORT
```

That is the whole start command, from the `Procfile`. There is no
`scripts/build_index` step. `app.localindex.ensure_index()` finds the committed
index, validates it against `config.CORPUS_VERSION` and `config.EMBED_MODEL`,
and loads it. Measured from a clean checkout of the repository: **0.47 s, 720
chunks, and `torch` is never imported.**

Two things still self-heal, so a misconfigured deploy degrades instead of
serving stale facts:

1. **Stale index.** A `corpus_version` or `embed_model` mismatch rebuilds. So a
   commit that changes `data/raw` without regenerating the index fails loudly
   rather than answering from old facts. If you see `IndexError_: index
   corpus_version is ... but config.CORPUS_VERSION is ...` in the log, rebuild and
   commit the index.
2. **Missing index.** If `data/processed/vector_index` is somehow absent, the app
   builds it on first load and shows progress in the **Index log** expander under
   the title. This path is for local use. On a free instance it is expected to
   hit the memory ceiling, so a failure there means "rebuild and commit the
   index", not "retry harder".

`app/ui.py` is unchanged by any of this. A valid index renders exactly as before.

## Rebuilding and committing the index

The index is a build artifact, but a deployable one. When `data/raw` changes:

```bash
python -m scripts.build_index
git add data/processed/vector_index
git commit -m "Rebuild the vector index"
```

Regenerate it on a developer machine. It takes ~2.2 minutes and ~565 MB there,
which is affordable; neither is affordable on a free instance.

The manifest carries `corpus_version`, `embed_model`, `embed_dtype` and
`embed_batch`. Any mismatch with the running config invalidates the index - that
check is what makes committing it safe.

The committed files contain chunk text and its provenance (`scheme`, `section`,
`as_of_date`, `source_url`). The only URLs are the eight public `hdfcfund.com`
fact sheets, which are already committed under `data/raw`. There are no
credentials in any of them.

## Service settings

| Setting | Value |
| --- | --- |
| Environment | Python |
| Python version | `.python-version` in the repo root - Render reads it (3.12.10) |
| Build command | `pip install -r requirements.txt` |
| Start command | leave blank to use the `Procfile`, or set it explicitly:<br>`export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false; exec streamlit run app/ui.py --server.address 0.0.0.0 --server.port $PORT` |
| Health check path | `/_stcore/health` |
| Instance type | Free (512 MB) - see "Memory" |

The **build command must not include `scripts/build_index.py`**. The build step
runs under the same 512 MB ceiling as the running service, so an index build
there is killed exactly as it would be at startup. This was the previous
recommendation in this file and it was wrong.

## Environment variables

Set these in the Render dashboard. **Never commit them** - `.env` is gitignored
and no key belongs in any file in the repository.

| Variable | Required | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | **Yes** | Generation. Without it the app still starts and still retrieves, and falls back to the top retrieved sentence with a banner saying so. |
| `HF_TOKEN` | Strongly recommended | Embedding model download. Free Hugging Face *read* token from `https://huggingface.co/settings/tokens`. Never logged. See "Hugging Face can refuse" below. |
| `EMBED_BATCH` | No | Index build batch size. Defaults to 8. Only matters if you rebuild on the instance. |
| `EMBED_FP16` | No | Off. Leave it off: torch here is CPU-only, so half precision is ~40x slower (2.2 min to 90 min) to save 36 MB. Only useful on a CUDA target. |
| `INDEX_DIR` | No | Where the index lives. Defaults to `data/processed/vector_index`. Point it at a writable volume if the checkout is read-only. |
| `MODEL_LOAD_ATTEMPTS` / `MODEL_LOAD_BACKOFF` | No | Retry count and base backoff for the model download. Defaults already outlast the rate-limit window. |

`GROQ_API_KEY` is the only one that is mandatory. Nothing in the repository
contains it, so a fresh instance cannot generate answers until it is set.

## Free-tier behaviour

- **Memory, and the out-of-memory 502.** This was the reported failure. Measured
  on this project against a 512 MB limit:

  | Path | Peak | Fits? |
  | --- | --- | --- |
  | Index build (fp32, batch 8) | 565 MB | No - killed |
  | Index build, unpinned threads | 950 MB | No - killed |
  | Load index, answer a question | 503 MB | Yes, thin |
  | Load index only, no model | 28 MB | Yes |

  So the build cannot run on a free instance and the serving path can. Committing
  the index is what separates the two. Render's proxy keeps the route registered
  while the container is reaped, which is why an out-of-memory death surfaces as
  a **502** rather than a clean 503.

  The serving peak is still close to the ceiling. `OMP_NUM_THREADS=1` and
  `MKL_NUM_THREADS=1` in the `Procfile` are load-bearing: torch sizes its thread
  pool from the CPU count, and a free instance reports more cores than it can run
  in parallel. If you override the start command in the dashboard, include them.

- **Ephemeral filesystem.** Anything written at runtime is lost on redeploy and
  spin-down. With the index committed this costs nothing - there is nothing to
  rebuild. A persistent disk is not needed for correctness and is a paid feature.
- **Spin-down after 15 minutes of inactivity.** The first request after a
  spin-down pays the ~87 MB embedding model download again. The index itself is
  always there.
- **Hugging Face can refuse.** The embedding model is downloaded rather than
  vendored. `huggingface.co` is Cloudflare-fronted and rate-limits anonymous
  callers: past its quota the response is an HTTP **429** carrying a
  "Just a moment..." HTML challenge page instead of the weights. The service stays
  LIVE, because nothing in the app failed - a dependency did - so you see a chat
  UI that loads and then reports a connection error while warming up, with a page
  of HTML where the message should be. Every cold start races the same limit from
  a shared free-tier IP, so restarting does not fix it.

  In order of effect:

  1. **Set `HF_TOKEN`.** The model is public, but an authenticated download gets a
     much higher rate limit. This is the fix.
  2. **Retries already happen.** Four attempts with a 5/10/20 s backoff
     (`MODEL_LOAD_ATTEMPTS`, `MODEL_LOAD_BACKOFF`), logged in the Index log. Raise
     the backoff rather than the count if the host is slow.
  3. If all attempts fail, the UI names the cause instead of rendering the
     challenge page, so the log says `Could not load the embedding model ...
     huggingface.co ...` rather than HTML.

## Verifying a deploy

```bash
curl -s https://<your-service>.onrender.com/_stcore/health
# {"status":"ok"}
```

Use `/_stcore/health`, not `/healthz`. Streamlit's catch-all serves the app shell
for any unmatched path, so `/healthz` returns 200 even when the app is broken.

Then in the browser: the title renders, no red traceback, and the first question
returns an answer with a clickable citation. A question the corpus cannot answer
must show the refusal copy, not an error.

The **Index log** expander is only populated if the app built the index. A healthy
deploy leaves it empty. That is the expected state now, not a missing feature.

Locally, both states can be reproduced without touching the committed index:

```bash
INDEX_DIR=$(mktemp -d)/vector_index python -m app.retrieve "expense ratio of HDFC ELSS TaxSaver Fund" --debug
```

The first run prints `building the vector index: no index at ...` and then
answers; the second run loads it and does not rebuild.