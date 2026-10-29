"""Streamlit front end, deployed on Streamlit Community Cloud.

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st

from eulaw import config
from eulaw import generate
from eulaw import retrieve

TITLE = "EU AI Act & GDPR"
DESCRIPTION = (
    "Ask a question about the EU Artificial Intelligence Act (Regulation (EU) "
    "2024/1689) or the GDPR (Regulation (EU) 2016/679). Answers are grounded "
    "in the full text of both regulations and cite the provisions they come "
    "from. If the regulations do not contain the answer, the app says so "
    "instead of guessing.")
DISCLAIMER = (
    "Not legal advice. This answers what the text of these two regulations "
    "says; it does not track amendments, case law or national rules. Always "
    "check the linked provision.")
EXAMPLES = (
    "When is an AI system classified as high-risk?",
    "Which AI practices are prohibited outright?",
    "What are the lawful bases for processing personal data?",
    "How quickly must a personal data breach be notified, and to whom?",
    "What is the capital of Austria?",  # Shows the refusal.
)


@st.cache_resource(show_spinner="Loading the embedding model...")
def _warm_up() -> bool:
    """Loads the model and index once; Streamlit reruns the script often."""
    retrieve.warm_up()
    return True


def _use_example(text: str) -> None:
    st.session_state.question = text
    st.session_state.run = True


def _submit() -> None:
    st.session_state.run = True


def _show_answer(question: str, k: int) -> None:
    _warm_up()
    with st.spinner("Retrieving and generating..."):
        try:
            result = generate.answer(question, k=k)
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Shown to the user instead of a traceback; the cause is logged.
            print(f"LLM call failed: {e!r}")
            st.error(f"The language model is unavailable "
                     f"({type(e).__name__}). Please try again in a moment.")
            return
    st.markdown(result["answer"])
    st.markdown("**Sources**")
    for i, source in enumerate(result["sources"], start=1):
        st.markdown(f"{i}. [{retrieve.cite(source)}]({source['url']}) "
                    f"(score {source['score']:.2f})")


def main() -> None:
    st.set_page_config(page_title=TITLE, layout="centered")
    st.title(TITLE)
    st.write(DESCRIPTION)
    st.info(DISCLAIMER)

    st.caption("Try one of these:")
    columns = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        columns[i % 2].button(example, key=f"example_{i}",
                              on_click=_use_example, args=(example,),
                              width="stretch")

    # A form, so moving the slider does not fire another (paid, rate-limited)
    # call to the language model.
    with st.form("ask"):
        st.text_input("Your question", key="question")
        k = st.slider("Retrieved provisions (k)", 3, 10, config.TOP_K)
        st.form_submit_button("Ask", type="primary", on_click=_submit)

    if st.session_state.get("run"):
        st.session_state.run = False
        question = st.session_state.get("question", "").strip()
        if question:
            _show_answer(question, int(k))


if __name__ == "__main__":
    main()
