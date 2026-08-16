"""Streamlit UI - the deployed front end. Same pipeline as app.py.

Run locally:  streamlit run streamlit_app.py
"""

import pathlib
import sys

import streamlit as st

# Community Cloud installs requirements.txt but never runs `pip install -e .`.
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from eulaw import config  # noqa: E402  (import needs the path line above)

st.set_page_config(page_title=config.TITLE, page_icon=None, layout="centered")


@st.cache_resource(show_spinner="Loading the embedding model...")
def _warm() -> bool:
    """Load the model and index once; Streamlit reruns the script constantly."""
    from eulaw.retrieve import _collection, _model

    _model()
    _collection()
    return True


def render_sources(sources: list[dict]) -> None:
    from eulaw.retrieve import cite

    st.markdown("**Sources**")
    for i, s in enumerate(sources, start=1):
        st.markdown(f"{i}. [{cite(s)}]({s['url']}) — score {s['score']:.2f}")


def _run_example(text: str) -> None:
    st.session_state.question = text
    st.session_state.run = True


def _submit() -> None:
    st.session_state.run = True


st.title(config.TITLE)
st.write(config.DESCRIPTION)
st.write(
    "Answers are grounded in these regulations and cited; if they do not "
    "contain the answer, the system says so instead of guessing."
)
st.info(config.DISCLAIMER)

st.caption("Try one of these:")
cols = st.columns(2)
for i, example in enumerate(config.EXAMPLES):
    cols[i % 2].button(example, key=f"example_{i}", on_click=_run_example,
                       args=(example,), use_container_width=True)

# A form: without it, every rerun (slider, button) would fire another
# paid, rate-limited call to the language model.
with st.form("ask"):
    st.text_input("Your question", key="question")
    k = st.slider("Retrieved provisions (k)", 3, 10, config.TOP_K)
    st.form_submit_button("Ask", type="primary", on_click=_submit)

if st.session_state.get("run"):
    st.session_state.run = False
    question = st.session_state.get("question", "").strip()
    if question:
        _warm()
        from eulaw.generate import answer

        with st.spinner("Retrieving and generating..."):
            try:
                result = answer(question, k=int(k))
            except Exception as e:
                st.error(f"The language model is unavailable ({type(e).__name__}). "
                         "Please try again in a moment.")
            else:
                st.markdown(result["answer"])
                render_sources(result["sources"])
