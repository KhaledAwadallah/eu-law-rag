"""Grounded generation - answer a question from retrieved chunks.

Provides:
    format_context(chunks) -> str          # numbered excerpts block for the prompt
    call_llm(prompt) -> str                # OpenAI-compatible chat completion
    answer(question, k) -> dict            # {"answer": str, "sources": list[dict]}

Command line with an example prompt:
    python -m askarxiv.generate "what methods reduce LLM inference cost?"
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

from askarxiv import config

# Transient HTTP failures worth retrying: rate limiting and server-side errors.
# A 400/401/404 is our mistake and will never succeed on retry.
RETRY_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 6

REFUSAL = "The indexed papers do not contain enough information to answer this."

PROMPT_TEMPLATE = """You are a careful research assistant. Answer the question \
using ONLY the numbered excerpts below.

Rules:
- Every claim in your answer must be supported by an excerpt; cite it like [1] or [2][3].
- Do not use any knowledge that is not in the excerpts.
- If the excerpts do not contain the answer, reply exactly: "{refusal}"
- Be concise: a few sentences, no preamble.

Excerpts:
{context}

Question: {question}

Answer:"""


def format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks as numbered excerpts the prompt can cite."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] From \"{c['title']}\":\n{c['text']}")
    return "\n\n".join(blocks)


def call_llm(prompt: str,
             base_url: str = config.LLM_BASE_URL,
             model: str = config.LLM_MODEL) -> str:
    """Send one chat completion request to any OpenAI-compatible endpoint."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.LLM_TEMPERATURE,
        "stream": False,
    }
    # Some hosted providers sit behind bot protection that rejects Python's
    # default user agent with 403, so identify the client explicitly.
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "askarxiv/0.1",
        "Accept": "application/json",
    }
    api_key = os.environ.get(config.LLM_API_KEY_ENV)
    if api_key:                       # hosted providers need it; local models don't
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_CODES or attempt == MAX_RETRIES - 1:
                raise
            # Honour the server's own advice when it gives some, otherwise
            # back off exponentially: 2, 4, 8, 16, 32 seconds.
            wait = float(e.headers.get("Retry-After") or 2 ** (attempt + 1))
            print(f"  [{e.code}] retrying in {wait:.0f}s "
                  f"(attempt {attempt + 1}/{MAX_RETRIES})", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("unreachable")   # loop either returns or raises


def answer(question: str, k: int = config.TOP_K) -> dict:
    """Full RAG loop: retrieve chunks, build the prompt, generate the answer."""
    from askarxiv.retrieve import retrieve

    chunks = retrieve(question, k=k)
    prompt = PROMPT_TEMPLATE.format(
        refusal=REFUSAL,
        context=format_context(chunks),
        question=question,
    )
    text = call_llm(prompt)
    return {"answer": text, "sources": chunks}


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What methods reduce LLM inference cost?"
    print(f"question: {question}\n")
    result = answer(question)
    print(result["answer"])
    print("\nSources:")
    for i, s in enumerate(result["sources"], start=1):
        print(f"  [{i}] {s['title'][:70]}  (chunk {s['chunk_index']}, score {s['score']:.3f})")
