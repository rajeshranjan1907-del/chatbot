# Render reads this for the web service's start command.
#
# There is no `python -m scripts.build_index` here any more, and that is the whole
# point of this revision. An earlier version built the index on boot because the
# index was gitignored, so a fresh instance had data/raw and no vectors. Measured
# against a 512 MB free instance, that build peaks at 565 MB (fp32, batch 8) and
# is killed before streamlit takes the port - which is what the 502s were.
#
# The index is now committed (see the note in .gitignore), so the build happens on
# a developer machine where it costs 2.2 minutes and no cold start is at risk, and
# this instance only *loads* it: 489 MB resident, inside the limit. The cost is
# that the vectors can drift from data/raw, which the manifest catches -
# localindex rebuilds on a corpus_version or embed_model mismatch rather than
# serving stale facts.
#
# If the index is ever missing from the image, the app still builds it on first
# load and shows the reason in the "Index log" expander. That path exists for
# local use; on a free instance it is expected to hit the same memory ceiling, so
# a build error there means "rebuild and commit the index", not "retry harder".
#
# The three environment variables are still here. torch sizes its thread pool from
# the CPU count; a free Render instance reports more cores than it can afford to
# run in parallel, and an unpinned private commit measured 950 MB - killed, and
# surfaced to the browser as a 502. One thread is what keeps the *serving* path
# (not just the build) under the limit.
#
# Locally:  streamlit run app/ui.py     (build first with: python -m scripts.build_index)
web: export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false; exec streamlit run app/ui.py --server.address 0.0.0.0 --server.port $PORT
