# Deploying to Render

The bot answers questions about HDFC mutual fund facts from a local NumPy vector
index. That index is a **build artifact**: it is gitignored, and it is not in the
repository. A Render instance therefore starts with the corpus and no vectors,
which is what produced the reported failure.

```
IndexError_: no vector index at /opt/render/project/src/data/processed/vector_index.
Build it with:
  python scripts/build_index.py
```

## What now happens

Three layers, so no single misconfiguration can leave the deployment unable to
answer:

1. **`Procfile` start command** - `python -m scripts.build_index` runs once per
   instance, before Streamlit binds the port. This is the primary path.
2. **Render build command** (optional, see below) - builds the same index at
   build time. Cheaper at runtime when the artifact survives into the image.
3. **`app.localindex.ensure_index()`** - the app builds the index itself on the
   first request of a cold start if neither of the above ran. It also rebuilds a
   *stale* index: wrong `embed_model`, wrong `corpus_version`, or missing
   `vectors.npy` / `chunks.jsonl`. Progress is shown in the **Index log**
   expander under the title, and a failure is rendered as an error rather than a
   traceback.

Layer 3 is a deliberate deviation from PRD FR-1.5 ("ingestion is a separate
command, never on app startup"). It is there because layers 1 and 2 depend on
configuration that lives in the Render dashboard, outside the repository: a
service that was never given a start command cannot answer a single question,
and there is no code change that can fix that from here. With the `Procfile` in
place the build is a separate command, run by the process manager, and layer 3
only ever fires when that command was skipped.

## Service settings

| Setting | Value |
| --- | --- |
| Environment | Python |
| Python version | `.python-version` in the repo root - Render reads it (3.12.10) |
| Build command | `pip install -r requirements.txt && python scripts/build_index.py` |
| Start command | `python -m scripts.build_index; exec streamlit run app/ui.py --server.address 0.0.0.0 --server.port $PORT` (or leave blank to use the `Procfile`) |
| Health check path | `/_stcore/health` (optional). Not `/healthz` - Streamlit's catch-all serves the app shell for any unmatched path, so `/healthz` returns 200 even when the app is broken. `/_stcore/health` returns a real `ok` body. |
| Environment variables | `GROQ_API_KEY` - required for generated answers. `HF_TOKEN` - strongly recommended, see below |

`GROQ_API_KEY` must be set in the dashboard. `.env` is gitignored, so it is not
deployed; without the key the app still starts and still retrieves, and falls
back to the top retrieved sentence with a banner saying so.

Set `HF_TOKEN` in the dashboard too. It is a free Hugging Face *read* token
(`https://huggingface.co/settings/tokens`), it is never logged, and it exists to
fix the failure described under "Hugging Face is reached at build and at first
start". Nothing else about the service changes when you set it.

## The build command alone is not enough

Worth knowing before you spend time on it: Render runs the build command on
separate compute from the running instance, and the pre-deploy command
explicitly does not carry filesystem changes into the deployed service. Do not
rely on the build command as the only place the index is created. Setting both
as above is deliberate.

## Free-tier behaviour

- **Ephemeral filesystem.** Everything written at runtime is lost on redeploy,
  restart and spin-down, so a free instance rebuilds the index on each cold
  start. That is one embed of 720 chunks: roughly 20-40 s on a free CPU, paid
  for once per cold start, shown in the Index log expander. A persistent disk
  removes the rebuild but is a paid feature and cannot be mounted during a build.
- **Spin-down after 15 minutes of inactivity.** The first request after a
  spin-down pays the model download and the rebuild again.
- **Memory, and the out-of-memory 502.** This is the cause of the reported
  failure. Measured on the cold-start path this repo actually runs
  (`scripts/build_index` then `streamlit run`):

  | Configuration | Working set | Private commit | Threads |
  | --- | --- | --- | --- |
  | Default | 504 MB | 950 MB | 32 |
  | `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` | 503 MB | 499 MB | 7 |

  The working set sits at the 512 MB free limit, and the private commit nearly
  doubles it, because `torch` sizes its thread pool from the CPU count and a
  free instance reports more cores than it can afford to run in parallel.
  Render's proxy still has the route registered while the container is reaped,
  which is why the symptom is a **502** rather than a clean 503.

  The `Procfile` now sets `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`, which is
  the whole fix: ~450 MB less committed memory, and the single-threaded embed
  of 720 chunks costs a few extra seconds once per cold start. Set the same two
  variables in the dashboard if you ever override the start command.

  If you would rather not serialise the embed, the alternative is a paid
  instance with more than 512 MB. Changing `EMBED_MODEL` to a smaller
  sentence-transformer is the third option, and note that it invalidates the
  existing index, which `ensure_index` detects and rebuilds automatically.
- **Hugging Face is reached at build and at first start, and it can refuse.**
  The embedding model is downloaded rather than vendored, so a cold instance
  pulls ~87 MB before it can answer anything. `huggingface.co` is Cloudflare-
  fronted and rate-limits anonymous callers: past its quota the response is an
  HTTP **429** carrying a "Just a moment..." HTML challenge page instead of the
  weights. The service stays LIVE throughout, because nothing about the app
  failed - a dependency did - so what you see is a chat UI that loads and then
  reports a connection error while warming up, with a page of HTML where the
  error message should be. This is the most likely reason a working deploy
  appears broken, and it is not fixed by restarting: every cold start races the
  same limit from a shared free-tier IP.

  Three things address it, in order of effect.

  1. **Set `HF_TOKEN`.** The model is public, but an authenticated download
     gets a much higher rate limit. This is the fix.
  2. **Retries already happen.** The load retries transient failures four times
     with a 5/10/20 s backoff (`MODEL_LOAD_ATTEMPTS`, `MODEL_LOAD_BACKOFF`),
     which outlasts the limit window in practice, and logs each attempt in the
     Index log. Raise the backoff rather than the count if the host is slow.
  3. **Keep the instance warm.** The model cache is on the same ephemeral disk as
     the index, so a paid instance or a persistent disk stops the download
     happening at all. Both are paid features.

  If all attempts fail, the UI now names the cause and the fix instead of
  rendering the challenge page, so the log will say
  `Could not load the embedding model ... huggingface.co ...` rather than HTML.

## Verifying a deploy

```bash
curl -s https://<your-service>.onrender.com/healthz          # 200
```

Then in the browser: the title renders, no red traceback, an **Index log**
expander is present, and the first question returns an answer with a clickable
citation and a sources expander. A question the corpus cannot answer must show
the refusal copy, not an error.

Locally, the same two states can be reproduced without touching the repo's own
index:

```bash
INDEX_DIR=$(mktemp -d)/vector_index python -m app.retrieve "expense ratio of HDFC ELSS Tax Saver Fund" --debug
```

The first run prints `building the vector index: no index at ...` and then
answers; the second run loads it and does not rebuild.
