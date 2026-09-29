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
| Health check path | `/healthz` (optional) |
| Environment variables | `GROQ_API_KEY` - required for generated answers |

`GROQ_API_KEY` must be set in the dashboard. `.env` is gitignored, so it is not
deployed; without the key the app still starts and still retrieves, and falls
back to the top retrieved sentence with a banner saying so.

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
- **Memory.** `torch` plus the ~90 MB MiniLM model needs well over the 512 MB a
  free instance has. If the service starts and then dies with an out-of-memory
  kill, that is the cause - move to a paid instance, or set `EMBED_MODEL` to a
  smaller sentence-transformer and rebuild the index. Note that changing
  `EMBED_MODEL` invalidates the existing index, which `ensure_index` detects and
  rebuilds automatically.
- **Hugging Face is reached at build and at first start.** The embedding model
  is downloaded rather than vendored. A blocked or rate-limited HF response
  shows up as a build failure in the log; layer 3 retries on the next cold start.

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
