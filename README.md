# AskArxiv

[![CI](https://github.com/KhaledAwadallah/askarxiv/actions/workflows/ci.yml/badge.svg)](https://github.com/KhaledAwadallah/askarxiv/actions/workflows/ci.yml)

Ask questions about a collection of ML research papers and get answers grounded in those papers, with citations — a Retrieval-Augmented Generation (RAG) pipeline built from scratch, no framework.

**Pipeline:** arXiv ingestion → PDF parsing → sliding-window chunking → embeddings (`bge-small-en-v1.5`) → ChromaDB vector search with per-paper diversity cap → grounded generation with citations and refusal (local `gpt-oss:20b` via Ollama, or any OpenAI-compatible endpoint) → quantitative evaluation harness.

## Status

- [x] Step 1 — project scaffold, environment, tests, CI-ready structure
- [x] Step 2 — paper ingestion (arXiv API + PyMuPDF): 50 cs.CL papers
- [x] Step 3 — chunking: 2,414 overlapping passages (size 1000, overlap 150)
- [x] Step 4 — embedding + vector index (ChromaDB, cosine, HNSW)
- [x] Step 5 — grounded generation: citations `[n]`, exact-refusal contract
- [x] Step 6 — evaluation harness: hit-rate, refusals, citations, LLM-as-judge faithfulness
- [x] Step 7 — Gradio web app, Dockerfile, GitHub Actions CI (lint + tests + image build)
- [ ] Step 8 — deployment (Hugging Face Spaces)

## Setup

```powershell
.\setup.ps1          # venv, dependencies, editable install, tests, git init
```

Generation additionally needs [Ollama](https://ollama.com) with the model pulled: `ollama pull gpt-oss:20b`. Any OpenAI-compatible provider works instead — set `LLM_BASE_URL`/`LLM_MODEL` in `src/askarxiv/config.py` and the `LLM_API_KEY` environment variable.

## Usage

```powershell
python -m askarxiv.ingest                  # download + parse the corpus
python -m askarxiv.chunk                   # split into overlapping passages
python -m askarxiv.index                   # embed + build the vector index
python -m askarxiv.retrieve "your question"    # inspect raw retrieval
python -m askarxiv.generate "your question"    # full RAG answer with sources
```

## Web app

```powershell
python app.py        # -> http://localhost:7860 (Ollama must be running)
```

Question box, adjustable retrieval depth, answers with `[n]` citations linked to the arXiv pages of the source papers. Try "What is the capital of Austria?" to see the refusal contract in action.

## Docker

```powershell
docker build -t askarxiv .
docker run -p 7860:7860 -v "${PWD}/data:/app/data" -e LLM_BASE_URL=http://host.docker.internal:11434/v1 askarxiv
```

The image contains code and dependencies only; the vector index is mounted at runtime (`-v`), and the LLM endpoint is injected via environment (`-e`) — inside a container, `host.docker.internal` reaches the Ollama server on your host machine.

## Evaluation

A 32-question benchmark (`eval/questions.jsonl`) hand-written against the actual corpus: 27 answerable questions tagged with their source paper, plus 5 trap questions with no answer in the corpus (including ML questions the model knows but the papers don't cover — leak detectors for the grounding contract).

```powershell
python eval\run_eval.py --no-llm           # retrieval metrics only (fast)
python eval\run_eval.py                    # full run incl. LLM-as-judge faithfulness
python eval\run_eval.py --k 10 --name k10  # experiments at different retrieval depth
```

Metrics: retrieval hit-rate@k, false-refusal rate, refusal accuracy on traps, citation rate, and faithfulness (the local LLM judges whether each answer is supported by its excerpts — checked UNSUPPORTED/PARTIAL/SUPPORTED). Results are saved to `eval/results/*.json` with a full config snapshot for reproducibility.

### Results

| Run | k | Chunk size | Hit-rate | False refusals | Refusal acc. | Citation rate | Faithfulness |
|---|---|---|---|---|---|---|---|
| k3 | 3 | 1000 | 1.00 | 0.00 | 1.00 | 0.93 | 0.94 |
| baseline | 5 | 1000 | 1.00 | 0.04 | 1.00 | 0.88 | 0.94 |
| k10 | 10 | 1000 | 1.00 | 0.00 | 1.00 | 0.93 | 0.93 |

LLM: `gpt-oss:20b` (local, Ollama). **Findings:** (1) Refusal accuracy is 1.00 in every run — 15/15 trap questions refused across three runs, including in-domain ML questions the model knows but the corpus doesn't contain; the grounding contract holds against the model's own knowledge. (2) Retrieval depth has no measurable effect between k=3 and k=10; with 27 answerable questions, one verdict flip moves a metric by ~4 points, so all observed differences are within benchmark noise (generation at temperature 0.2 adds run-to-run variance). (3) Two questions are consistently judged PARTIAL across all runs — systematic cases identified for error analysis.

## Configuration

Every pipeline knob (chunk size/overlap, top-k, diversity cap, embedding model, corpus query, LLM endpoint) lives in `src/askarxiv/config.py`.

## Tests

```powershell
pytest -q
```
