"""Tests for prompt building and the LLM client (no LLM or network needed)."""

import io
import json
from unittest import mock

from eulaw import generate


def _chunks():
    return [
        {"text": "alpha text", "title": "AI Act", "label": "Article 6"},
        {"text": "beta text", "title": "GDPR", "label": "Recital 40"},
    ]


def test_format_context_numbers_and_names_excerpts():
    context = generate.format_context(_chunks())
    assert '[1] From "AI Act, Article 6":\nalpha text' in context
    assert '[2] From "GDPR, Recital 40":\nbeta text' in context


def test_build_prompt_contains_rules_and_question():
    prompt = generate.build_prompt("Q?", _chunks())
    assert generate.REFUSAL in prompt
    assert "ONLY" in prompt
    assert "Article 6(2) of the AI Act" in prompt  # Provision-level citation.
    assert "Q?" in prompt
    assert prompt.endswith("Answer:")


def test_build_prompt_forbids_relabelling_excerpts():
    # Added after the model answered a question about the Cyber Resilience
    # Act by presenting AI Act excerpts as that law.
    prompt = generate.build_prompt("Q?", _chunks()).lower()
    assert "never attribute an excerpt to a law it does not come from" in prompt


class _FakeResponse(io.BytesIO):

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_call_llm_parses_openai_response():
    reply = {"choices": [{"message": {"content": "  the answer [1]  "}}]}
    response = _FakeResponse(json.dumps(reply).encode())
    with mock.patch("urllib.request.urlopen",
                    return_value=response) as urlopen:
        assert generate.call_llm("prompt text") == "the answer [1]"

    request = urlopen.call_args[0][0]
    assert request.full_url.endswith("/chat/completions")
    sent = json.loads(request.data)
    assert sent["messages"][0]["content"] == "prompt text"
    assert sent["stream"] is False
