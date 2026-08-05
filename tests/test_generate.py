"""Unit tests for the generation step (no LLM or network involved)."""

import io
import json
from unittest.mock import patch

from askarxiv.generate import PROMPT_TEMPLATE, REFUSAL, call_llm, format_context


def _chunks():
    return [
        {"text": "alpha text", "title": "Paper A", "paper_id": "a", "chunk_index": 0, "score": 0.9},
        {"text": "beta text", "title": "Paper B", "paper_id": "b", "chunk_index": 3, "score": 0.8},
    ]


def test_format_context_numbers_and_titles():
    ctx = format_context(_chunks())
    assert '[1] From "Paper A":' in ctx
    assert '[2] From "Paper B":' in ctx
    assert "alpha text" in ctx and "beta text" in ctx


def test_prompt_contains_rules_and_question():
    prompt = PROMPT_TEMPLATE.format(refusal=REFUSAL, context="CTX", question="Q?")
    assert REFUSAL in prompt          # refusal instruction present
    assert "ONLY" in prompt           # grounding instruction present
    assert prompt.rstrip().endswith("Answer:")


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
