# Render reads this for the web service's start command.
#
# Why the index is built here rather than only in the build command: the vector
# index is a gitignored artifact under data/processed/, and Render's filesystem
# is ephemeral, so a fresh instance has data/raw and no vectors. Building here
# happens once per instance, before streamlit takes the port, and it is the only
# build that is guaranteed to land on the filesystem the app actually reads.
#
# The trailing `;` is deliberate: if this build fails, streamlit still starts and
# the app retries the build on its first load, showing the reason in the "Index
# log" expander. A transient Hugging Face download failure should not take the
# deployment down when there is a second path to the same index.
#
# Locally:  python -m scripts.build_index && streamlit run app/ui.py
web: python -m scripts.build_index; exec streamlit run app/ui.py --server.address 0.0.0.0 --server.port $PORT
