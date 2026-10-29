# EU AI Act & GDPR: question answering with citations

[![CI](https://github.com/KhaledAwadallah/eu-law-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/KhaledAwadallah/eu-law-rag/actions/workflows/ci.yml)
[![Live demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://eu-law-rag-o7k3yvaafngnv6ljlkxg6h.streamlit.app/)

Ask a question about the EU AI Act or the GDPR and get a short answer based
only on the legal text, with links to the exact articles on EUR-Lex. If the
regulations don't contain the answer, the app says so instead of guessing.
It is a RAG pipeline built from scratch, without a framework.

**[Try the live demo](https://eu-law-rag-o7k3yvaafngnv6ljlkxg6h.streamlit.app/)**,
e.g. *"When is an AI system high-risk?"*, or *"What is the capital of
Austria?"* to see it refuse.

> Not legal advice. It answers from the text of the two regulations only, not
> from amendments, case law or national rules.

## How it works

1. **Corpus**: both regulations are downloaded from EUR-Lex and split into
   their 578 articles, recitals and annexes.
2. **Chunks**: one chunk per provision (long ones are split), 1,357 in total.
   EUR-Lex gives every provision an anchor, so each chunk links straight to it.
3. **Index**: chunks are embedded with `bge-small-en-v1.5` and stored in
   ChromaDB.
4. **Retrieval**: the top 5 chunks, with two rules on top of similarity:
   at most 2 chunks from the same provision, and 3 of the 5 slots reserved for
   articles and annexes. Without the second rule, recitals (the explanatory
   preamble) push out the binding articles on plain-language questions.
5. **Answer**: an LLM answers only from the retrieved excerpts and cites them,
   or replies with a fixed refusal.

`eulaw/` has one file per step and all settings in `config.py`, where adding
a regulation means adding its CELEX id.

## Run it

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The vector index is in the repo. Answers need an LLM: [Ollama](https://ollama.com)
with `gpt-oss:20b` by default, or any OpenAI-compatible API set through
`LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY`. The live demo runs on Streamlit
Community Cloud with those three set as secrets.

To rebuild the index, or to query from the command line:

```bash
python -m eulaw.corpus          # download and parse the regulations
python -m eulaw.chunk           # split into chunks
python -m eulaw.index           # embed and index
python -m eulaw.retrieve "When is an AI system high-risk?"
python -m eulaw.generate "When is an AI system high-risk?"
```

## Evaluation

[`eval/questions.jsonl`](eval/questions.jsonl) has 100 test questions:

- 80 answerable ones, each labelled with the provision that answers it
  (e.g. GDPR Article 33 for the data-breach deadline).
- 20 traps with no answer in the corpus: other EU laws (Digital Services Act,
  Data Act, NIS2, ...), case law, and off-topic questions.

> The evaluation questions are AI-generated.

```bash
python -m eval.run_eval --no-llm   # retrieval only
python -m eval.run_eval            # with answers, refusals and LLM-as-judge
python -m eval.run_eval --plain    # plain top-k, without the two rules
```

**Retrieval** (`eval/results/`): how often the top-k contains the provision
that answers the question.

| k  | plain top-k | with the two rules |
|----|-------------|--------------------|
| 3  | 58/80       | **73/80**          |
| 5  | 66/80       | **73/80**          |
| 10 | 74/80       | **75/80**          |

**Answers**: the full run with a local `gpt-oss:20b` was done on the first 32
questions (`eval/results/k5.json`). It answered all 26 answerable questions,
refused 5 of the 6 traps, cited its sources in all 26 answers, and scored
0.96 on faithfulness (judged by the same model, so an upper bound).

### Findings

- **The ranking rules matter most when k is small.** At k=3 they lift
  retrieval from 58 to 73 of 80; at k=10 there is room for everything anyway.
- **Some misses find the explanation, not the rule.** For 3 of the 7 misses at
  k=5 (emotion recognition at work, credit scoring, the household exemption)
  a recital stating the same rule is retrieved, but not the article. Others are
  worse: the maximum fine for a prohibited AI practice retrieves Article 100
  (fines for EU institutions, up to EUR 1.5 million) instead of Article 99
  (EUR 35 million or 7% of turnover).
- **One trap breaks the refusal.** Asked about the Cyber Resilience Act, which
  the AI Act only mentions indirectly (Recital 77), the model presented AI Act
  excerpts as that law. A prompt rule did not stop it; a string check at answer
  time (like the one the eval uses) would.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests need no network, model or LLM.
