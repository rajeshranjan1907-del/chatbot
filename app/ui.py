"""Streamlit chat UI - architecture.md FR-6.

Renders the §5.6 response shape: the answer, a real clickable citation link, and
a sources expander showing the retrieved chunks. The expander is the point. It
is how the demo shows that retrieval is real rather than asserting it
(architecture.md:330) - a bot that only prints an answer is indistinguishable
from one that made it up.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Streamlit runs the script, not the package, so the repo root is not guaranteed
# to be importable - and it is not on sys.path when the app is started from
# another directory (a Render start command, say). Put it there explicitly
# before the first app import.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

st.set_page_config(page_title="HDFC Mutual Fund Facts", page_icon="📊", layout="centered")

from app import config
from app.generate import answer_question
from app.localindex import ensure_index
from app.retrieve import retrieve

EXAMPLES = [
    "What is the expense ratio of HDFC Large Cap Fund Direct Growth?",
    "Is there a lock-in for HDFC ELSS Tax Saver Fund?",
    "What is the minimum SIP for HDFC Equity Flexi Cap Fund?",
    "What is the exit load on HDFC Small Cap Fund?",
    "What is the riskometer and benchmark of HDFC Balanced Advantage Fund?",
    "How do I download my capital gains statement?",
]

#: Shown instead of a chat bubble when the corpus cannot answer. Deterministic,
#: never generated - the LLM must not shape a refusal.
REASON_COPY = {
    "fact_absent_from_corpus": (
        "That figure is not in these documents. HDFC's factsheets list the minimum "
        f"lump sum but not a minimum SIP, so check {config.HDFC_MF_LINK}."
    ),
    "model_declined_no_answer": (
        "The retrieved chunks don't cover that. The closest source is "
        f"{config.HDFC_MF_LINK}."
    ),
    "below_similarity_floor": (
        "Nothing in the corpus scored high enough similarity to answer that."
    ),
    "no_hits": "No matching content was found in the corpus.",
    "scheme_not_in_corpus": "The corpus holds HDFC AMC documents only.",
    "unknown_scheme": "I only have documents for HDFC AMC's mutual fund schemes.",
}


#: Lines captured while the index was built. Module-level on purpose: the build
#: happens inside a cached function, so the output has to outlive that call to be
#: renderable. `ensure_index` is chatty about what it did and why, and on Render
#: that output is the only evidence of what happened during a cold start.
_BUILD_LOG: list[str] = []


def _log(line: str) -> None:
    _BUILD_LOG.append(str(line))


@st.cache_resource
def _warm():
    """Load the index and embedding model once, not per rerun.

    Streamlit reruns the whole script on every interaction, so without this the
    ~90 MB embedding model would reload on each keystroke.

    `ensure_index` builds the index when it is absent. The index is a gitignored
    artifact and Render's filesystem is ephemeral, so a fresh instance has the
    corpus and no vectors and could not answer anything at all. A Render build
    command or pre-deploy step can produce it ahead of time - see
    docs/deploy_render.md - but neither is something the app can rely on: a
    failed build, a redeploy onto a clean disk, or a service that was never
    configured to build leaves the deployment unable to answer. Recovering here
    costs one build on the first request of a cold start and makes the app
    independent of deploy-time state.
    """
    index = ensure_index(log=_log)
    retrieve("warm up the index and embedding model", index)
    return index


def main() -> None:
    st.title("HDFC Mutual Fund Facts")
    st.caption(
        f"Answers only from {config.CORPUS_VERSION} official HDFC AMC documents. "
        "No advice, no performance figures."
    )

    if not config.GROQ_API_KEY:
        st.warning(
            "GROQ_API_KEY is not set, so answers fall back to the top retrieved "
            "sentence. Set it in the environment to get real answers."
        )

    with st.spinner("Loading the corpus index..."):
        try:
            _warm()
        except Exception as exc:
            # Anything that goes wrong warming up - no corpus, a coverage
            # failure, a blocked Hugging Face download, an out-of-memory kill -
            # leaves the bot unable to answer, and none of it is the user's
            # fault. A deployed app renders the reason; it does not show a
            # traceback to somebody asking about an expense ratio.
            st.error("The bot could not load its vector index.")
            st.code(f"{type(exc).__name__}: {exc}", language=None)
            st.caption(
                "The index is built from data/raw on first load. On a deployed "
                "service this usually means the build step did not run or the "
                "model could not be downloaded - see docs/deploy_render.md."
            )
            st.stop()

    if _BUILD_LOG:
        with st.expander("Index log"):
            st.code("\n".join(_BUILD_LOG[-40:]), language=None)

    for question in EXAMPLES:
        if st.sidebar.button(question, key=f"ex_{question[:24]}", use_container_width=True):
            st.session_state.pending = question
            st.rerun()

    question = st.chat_input("Ask about expense ratio, exit load, lock-in, benchmark...")
    if question:
        st.session_state.pending = question

    if question := st.session_state.pop("pending", None):
        with st.chat_message("user"):
            st.markdown(question)

        result = answer_question(question)

        with st.chat_message("assistant"):
            if result["ok"]:
                st.markdown(result["answer"])
                citation = result["citation"]
                if citation:
                    st.markdown(
                        f"[{citation['label']}]({citation['url']})",
                        help=f"chunk_id: {citation['chunk_id']}",
                    )
                st.caption(
                    f"As of {result['as_of']}"
                    + (f" · flags: {', '.join(result['flags'])}" if result["flags"] else "")
                )
            else:
                st.markdown(result["answer"])
                st.caption(REASON_COPY.get(result["reason"], result.get("detail", "")))
                if result.get("detail"):
                    st.caption(result["detail"])

        # The sources expander, shown for answerable questions.
        if result["ok"]:
            context = retrieve(question)
            if context.ok:
                with st.expander(f"Sources ({len(context.chunks)} retrieved chunks)"):
                    for rank, (chunk, score) in enumerate(
                        zip(context.chunks, context.scores), start=1
                    ):
                        st.markdown(
                            f"**[{rank}]** cosine `{score:.4f}` · {chunk.section or 'n/a'}"
                            f" · {chunk.source_type} · as of {chunk.as_of_date or 'n/a'}"
                        )
                        st.caption(f"`{chunk.chunk_id}`")
                        with st.popover("text"):
                            st.text(chunk.text)
                        st.markdown(
                            f"[source]({chunk.source_url})", unsafe_allow_html=False
                        )

    st.sidebar.divider()
    st.sidebar.caption(
        "Corpus is official HDFC AMC factsheets, KIMs, SAI extracts and addenda. "
        "Figures reflect the source date, not today."
    )


if __name__ == "__main__":
    main()
