"""Builds the grounded prompt, calls the LLM and returns answer plus sources.

Usage:
    python -m eulaw.generate "Which AI practices are prohibited?"
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

from eulaw import config
from eulaw import retrieve

REFUSAL = (
    "The indexed regulations do not contain enough information to answer "
    "this.")

_PROMPT_TEMPLATE = """\
You are a careful legal research assistant. Answer the question using ONLY \
the numbered excerpts below.

Rules:
- Every claim in your answer must be supported by an excerpt; cite it like \
[1] or [2][3]. When an excerpt is a numbered provision, name it in your \
answer (for example "Article 6(2) of the AI Act").
- Do not use any knowledge that is not in the excerpts.
- Never attribute an excerpt to a law it does not come from. Each excerpt \
names its own regulation; if the question asks about a different one, you \
do not have it.
- If the excerpts do not contain the answer, reply exactly: "{refusal}"
- Be concise: a few sentences, no preamble.

Excerpts:
{context}

Question: {question}

Answer:"""

# Transient errors worth retrying; anything else will not fix itself.
_RETRY_CODES = (429, 500, 502, 503, 504)
_MAX_RETRIES = 6


def format_context(chunks: list[dict]) -> str:
    """Numbers the excerpts and heads each with its provision."""
    return "\n\n".join(
        f'[{i}] From "{retrieve.cite(chunk)}":\n{chunk["text"]}'
        for i, chunk in enumerate(chunks, start=1))


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Fills the prompt template with the question and its excerpts."""
    return _PROMPT_TEMPLATE.format(refusal=REFUSAL,
                                   context=format_context(chunks),
                                   question=question)


def call_llm(prompt: str,
             base_url: str = config.LLM_BASE_URL,
             model: str = config.LLM_MODEL) -> str:
    """Sends one chat completion to an OpenAI-compatible endpoint.

    Rate limits and server errors are retried with exponential backoff.
    The API key, if the endpoint needs one, is read from LLM_API_KEY.

    Args:
        prompt: The full prompt, sent as a single user message.
        base_url: The endpoint, e.g. "http://localhost:11434/v1" for Ollama.
        model: The model name at that endpoint.

    Returns:
        The model's reply, stripped of surrounding whitespace.

    Raises:
        urllib.error.HTTPError: The request failed, after retries for
            transient errors.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.LLM_TEMPERATURE,
        "stream": False,
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        # Some providers' bot protection rejects Python's default user agent.
        "User-Agent": "eu-law-rag/0.1",
    }
    api_key = os.environ.get("LLM_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                data = json.loads(response.read())
            return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            if e.code not in _RETRY_CODES or attempt == _MAX_RETRIES:
                raise
            wait = float(e.headers.get("Retry-After") or 2**attempt)
            print(f"HTTP {e.code}, retrying in {wait:.0f}s "
                  f"({attempt}/{_MAX_RETRIES})", file=sys.stderr)
            time.sleep(wait)
    raise AssertionError("unreachable: the loop returns or raises")


def answer(question: str, k: int = config.TOP_K) -> dict:
    """Answers a question from the regulations.

    Args:
        question: The user's question.
        k: How many chunks to retrieve as context.

    Returns:
        A dict with the model's 'answer' and the retrieved 'sources'.
    """
    chunks = retrieve.retrieve(question, k=k)
    return {"answer": call_llm(build_prompt(question, chunks)),
            "sources": chunks}


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer one question.")
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    args = parser.parse_args()
    # Legal text has characters a Windows console cannot always encode.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    result = answer(args.question, k=args.k)
    print(result["answer"])
    print("\nSources:")
    for i, source in enumerate(result["sources"], start=1):
        print(f"  [{i}] {retrieve.cite(source)} ({source['score']:.3f}) "
              f"{source['url']}")


if __name__ == "__main__":
    main()
