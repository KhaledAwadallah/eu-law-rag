"""Checks that the Streamlit app renders (no model, index or LLM needed)."""

import pathlib

from streamlit.testing import v1 as streamlit_testing

_APP = pathlib.Path(__file__).parent.parent / "streamlit_app.py"


def test_app_renders_without_errors():
    app = streamlit_testing.AppTest.from_file(str(_APP)).run()
    assert not app.exception
    assert app.title[0].value == "EU AI Act & GDPR"
    assert len(app.button) >= 5  # The example questions.
