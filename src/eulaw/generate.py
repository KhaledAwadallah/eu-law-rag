"""Build the grounded prompt, call the LLM, return the answer and its sources.

    python -m eulaw.generate "which AI practices are prohibited?"
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

from eulaw import config

# Transient failures only: a 400/401/404 is our bug and will never succeed.
RETRY_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 6

REFUSAL = "The indexed regulations do not contain enough information to answer this."

PROMPT_TEMPLATE = """You are a careful legal research assistant. Answer the \
question using ONLY the numbered excerpts below.

Rules:
- Every claim in your answer must be supported by an excerpt; cite it like [1] or [2][3].{citation_hint}
- Do not use any knowledge that is not in the excerpts.
- Never attribute an excerpt to a law it does not come from. Each excerpt names \
its own regulation; if the question asks about a different one, you do not have it.
- If the excerpts do not contain the answer, reply exactly: "{refusal}"
- Be concise: a few sentences, no preamble.

Excerpts:
{context}

Question: {question}

Answer:"""


def format_context(chunks: list[dict]) -> str:
    """Numbered excerpts, each headed by its provision so citations can be exact."""
    from eulaw.retrieve import cite

    return "\n\n".join(f"[{i}] From \"{cite(c)}\":\n{c['text']}"
                       for i, c in enumerate(chunks, start=1))


def build_prompt(question: str, chunks: list[dict]) -> str:
    return PROMPT_TEMPLATE.format(
        refusal=REFUSAL,
        citation_hint=config.CITATION_HINT,
        context=format_context(chunks),
        question=question,
    )


def call_llm(prompt: str,
             base_url: str = config.LLM_BASE_URL,
             model: str = config.LLM_MODEL) -> str:
    """One chat completion against any OpenAI-compatible endpoint."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.LLM_TEMPERATURE,
        "stream": False,
    }
    # Some providers' bot protection rejects Python's default user agent with 403.
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "eu-law-rag/0.1",
        "Accept": "application/json",
    }
    api_key = os.environ.get(config.LLM_API_KEY_ENV)
    if api_key:                       # hosted providers need it; local ones don't
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
            # Honour Retry-After when given, else back off 2, 4, 8, 16, 32s.
            wait = float(e.headers.get("Retry-After") or 2 ** (attempt + 1))
            print(f"  [{e.code}] retrying in {wait:.0f}s "
                  f"(attempt {attempt + 1}/{MAX_RETRIES})", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("unreachable")   # the loop either returns or raises


def answer(question: str, k: int = config.TOP_K) -> dict:
    """Retrieve, prompt, generate."""
    from eulaw.retrieve import retrieve

    chunks = retrieve(question, k=k)
    return {"answer": call_llm(build_prompt(question, chunks)), "sources": chunks}


if __name__ == "__main__":
    # Legal text has characters the Windows console cannot encode by default.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from eulaw.retrieve import cite

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", nargs="*", default=[])
    parser.add_argument("--k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    question = " ".join(args.question) or config.EXAMPLES[0]
    print(f"question: {question}\n")
    result = answer(question, k=args.k)
    print(result["answer"])
    print("\nSources:")
    for i, s in enumerate(result["sources"], start=1):
        print(f"  [{i}] {cite(s)[:70]}  (score {s['score']:.3f})  {s['url']}")
