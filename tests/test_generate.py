"""Unit tests for the generation step (no LLM or network involved)."""

import io
import json
from unittest.mock import patch

from eulaw.generate import REFUSAL, build_prompt, call_llm, format_context


def _chunks():
    return [
        {"text": "alpha text", "title": "AI Act", "doc_id": "32024R1689",
         "label": "Article 6", "url": "https://eur-lex.europa.eu/x#art_6",
         "chunk_index": 0, "score": 0.9},
        {"text": "beta text", "title": "GDPR", "doc_id": "32016R0679",
         "label": "Recital 40", "url": "https://eur-lex.europa.eu/y#rct_40",
         "chunk_index": 3, "score": 0.8},
    ]


def test_format_context_numbers_excerpts_and_names_provisions():
    ctx = format_context(_chunks())
    assert '[1] From "AI Act, Article 6":' in ctx
    assert '[2] From "GDPR, Recital 40":' in ctx
    assert "alpha text" in ctx and "beta text" in ctx


def test_prompt_contains_the_grounding_rules_and_question():
    prompt = build_prompt("Q?", _chunks())
    assert REFUSAL in prompt          # refusal instruction present
    assert "ONLY" in prompt           # grounding instruction present
    assert "Q?" in prompt
    assert prompt.rstrip().endswith("Answer:")


def test_prompt_asks_for_provision_level_citation():
    assert "Article 6(2) of the AI Act" in build_prompt("Q?", _chunks())


def test_prompt_forbids_attributing_excerpts_to_another_law():
    # Regression on the Cyber Resilience Act failure: the model answered a
    # question about a regulation absent from the corpus by relabelling AI Act
    # excerpts, with every individual claim genuinely present in the sources.
    prompt = build_prompt("Q?", _chunks()).lower()
    assert "never attribute an excerpt to a law it does not come from" in prompt


def test_call_llm_parses_openai_response():
    fake = {"choices": [{"message": {"content": "  the answer [1]  "}}]}

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    with patch("urllib.request.urlopen",
               return_value=FakeResp(json.dumps(fake).encode())) as mock_open:
        out = call_llm("prompt text")

    assert out == "the answer [1]"    # stripped
    req = mock_open.call_args[0][0]
    assert req.full_url.endswith("/chat/completions")
    sent = json.loads(req.data)
    assert sent["messages"][0]["content"] == "prompt text"
    assert sent["stream"] is False
