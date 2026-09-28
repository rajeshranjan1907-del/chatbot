"""Streamlit chat UI - architecture.md FR-6.

Renders the §5.6 response shape: the answer, a real clickable citation link, and
a sources expander showing the retrieved chunks. The expander is the point. It
is how the demo shows that retrieval is real rather than asserting it
(architecture.md:330) - a bot that only prints an answer is indistinguishable
from one that made it up.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="HDFC Mutual Fund Facts", page_icon="📊", layout="centered")

from app import config
from app.generate import answer_question
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


@st.cache_resource
def _warm():
    """Load the model and index once, not per rerun.

    Streamlit reruns the whole script on every interaction, so without this the
    embedding model would reload on each keystroke.
    """
    retrieve("warm up the index and embedding model")
    return True


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
        _warm()

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
