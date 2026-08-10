"""Streamlit web UI - deployment target for Streamlit Community Cloud.

Same pipeline as app.py (Gradio); a second front end because Community Cloud
hosts Streamlit apps for free while Gradio Spaces now require a paid plan.

Run locally:  streamlit run streamlit_app.py
"""

import pathlib
import sys

import streamlit as st

# Streamlit Community Cloud installs requirements.txt but does not run
# `pip install -e .`, so make the src/ layout importable explicitly.
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from askarxiv import config  # noqa: E402  (import needs the path line above)

EXAMPLES = [
    "What methods make LLM inference more efficient?",
    "What is Salience Bias in commonsense reasoning?",
    "How does ParliamentBench evaluate deception in LLM agents?",
    "What is the capital of Austria?",   # demonstrates the refusal contract
]

st.set_page_config(page_title="AskArxiv", page_icon=None, layout="centered")


@st.cache_resource(show_spinner="Loading the embedding model...")
def _warm() -> bool:
    """Load the embedding model and open the index once per server process.

    Streamlit reruns the whole script on every interaction, so without this
    cache the model would reload on each question.
    """
    from askarxiv.retrieve import _collection, _model

    _model()
    _collection()
    return True


def render_sources(sources: list[dict]) -> None:
    st.markdown("**Sources**")
    for i, s in enumerate(sources, start=1):
        url = f"https://arxiv.org/abs/{s['paper_id']}"
        st.markdown(f"{i}. [{s['title']}]({url}) — chunk {s['chunk_index']}, "
                    f"score {s['score']:.2f}")


st.title("AskArxiv")
st.write(
    "Ask questions about 50 recent LLM papers (arXiv, cs.CL). Answers are "
    "grounded in the papers and cited; if the papers don't contain the answer, "
    "the system says so instead of guessing."
)

if "question" not in st.session_state:
    st.session_state.question = ""

st.caption("Try one of these:")
cols = st.columns(2)
for i, example in enumerate(EXAMPLES):
    if cols[i % 2].button(example, use_container_width=True):
        st.session_state.question = example

question = st.text_input("Your question", key="question")
k = st.slider("Retrieved chunks (k)", 3, 10, config.TOP_K)

if st.button("Ask", type="primary") or question:
    if question.strip():
        _warm()
        from askarxiv.generate import answer

        with st.spinner("Retrieving and generating..."):
            try:
                result = answer(question, k=int(k))
            except Exception as e:
                st.error(f"The language model is unavailable ({type(e).__name__}). "
                         "Please try again in a moment.")
            else:
                st.markdown(result["answer"])
                render_sources(result["sources"])
