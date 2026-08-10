# AskArxiv

[![CI](https://github.com/KhaledAwadallah/askarxiv/actions/workflows/ci.yml/badge.svg)](https://github.com/KhaledAwadallah/askarxiv/actions/workflows/ci.yml)
[![Live demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://askarxiv-khwgghzpnn4h25fy4c3esd.streamlit.app/)

**[Try the live demo](https://askarxiv-khwgghzpnn4h25fy4c3esd.streamlit.app/)** — ask about the papers, or ask "What is the capital of Austria?" to see the grounding contract refuse a question it cannot answer from the corpus.

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
- [x] Step 8 — deployed: Streamlit front end on Community Cloud, hosted LLM via Groq, prebuilt index

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
python app.py                        # Gradio -> http://localhost:7860
streamlit run streamlit_app.py       # Streamlit -> http://localhost:8501
```

Question box, adjustable retrieval depth, answers with `[n]` citations linked to the arXiv pages of the source papers. Try "What is the capital of Austria?" to see the refusal contract in action.

## Docker

```powershell
docker build -t askarxiv .
docker run -p 7860:7860 -v "${PWD}/data:/app/data" -e LLM_BASE_URL=http://host.docker.internal:11434/v1 askarxiv
```

The image contains code and dependencies only; the vector index is mounted at runtime (`-v`), and the LLM endpoint is injected via environment (`-e`) — inside a container, `host.docker.internal` reaches the Ollama server on your host machine.

## Deployment

Two front ends share one pipeline: `app.py` (Gradio, local and Docker) and `streamlit_app.py` (Streamlit, deployed). The deployed app runs the embedding model and vector search on CPU and calls a hosted OpenAI-compatible API for generation — configured entirely through environment variables, so the same code runs against local Ollama or a cloud provider without modification:

To run your own instance, set these in the hosting platform (Streamlit Cloud: *Advanced settings → Secrets*):

| Setting | Deployed value | Type |
|---|---|---|
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | variable |
| `LLM_MODEL` | `openai/gpt-oss-120b` | variable |
| `LLM_API_KEY` | your provider key | **secret** |

(`EMBEDDING_DEVICE` is available to force `cpu` where CUDA is detected but unusable; unnecessary on CPU-only hosts.)

The prebuilt Chroma index (~30 MB) is committed so the deployed app never has to run ingestion; the API key lives in the platform's secret store and never enters version control.

A packaging script for Hugging Face Spaces is also included (`python deploy\build_space.py` → a self-contained `deploy/space/` with the package promoted to top level, the index bundled, and runtime-only dependencies). Hugging Face restricted Gradio and Docker Spaces to paid plans in 2026, so Streamlit Community Cloud is the free deployment path; the Space package remains ready if that changes.

## Evaluation

A 32-question benchmark (`eval/questions.jsonl`) hand-written against the actual corpus: 27 answerable questions tagged with their source paper, plus 5 trap questions with no answer in the corpus (including ML questions the model knows but the papers don't cover — leak detectors for the grounding contract).

```powershell
python eval\run_eval.py --no-llm           # retrieval metrics only (fast)
python eval\run_eval.py                    # full run incl. LLM-as-judge faithfulness
python eval\run_eval.py --k 10 --name k10  # experiments at different retrieval depth
```

Metrics: retrieval hit-rate@k, false-refusal rate, refusal accuracy on traps, citation rate, and faithfulness (the local LLM judges whether each answer is supported by its excerpts — checked UNSUPPORTED/PARTIAL/SUPPORTED). Results are saved to `eval/results/*.json` with a full config snapshot for reproducibility.

### Results

**Retrieval depth sweep** (local `gpt-oss:20b` via Ollama):

| Run | k | Chunk size | Hit-rate | False refusals | Refusal acc. | Citation rate | Faithfulness |
|---|---|---|---|---|---|---|---|
| k3 | 3 | 1000 | 1.00 | 0.00 | 1.00 | 1.00 | 0.94 |
| baseline | 5 | 1000 | 1.00 | 0.04 | 1.00 | 1.00 | 0.94 |
| k10 | 10 | 1000 | 1.00 | 0.00 | 1.00 | 1.00 | 0.93 |

**Model comparison** (k=5, identical corpus, index and prompt):

| Model | Host | False refusals | Refusal acc. | Citation rate | Faithfulness |
|---|---|---|---|---|---|
| `gpt-oss:20b` | local (Ollama, RTX 5070 Ti) | 0.04 | 1.00 | 1.00 | 0.94 |
| `openai/gpt-oss-120b` | Groq (free tier) | 0.00 | 1.00 | 1.00 | 0.94 |

**Findings**

1. **Refusal accuracy is 1.00 in every run** — 20/20 trap questions refused across four runs and two models, including in-domain ML questions the models demonstrably know but the corpus doesn't contain. The grounding contract holds against the model's own knowledge.
2. **Retrieval depth has no measurable effect between k=3 and k=10.** With 27 answerable questions, one verdict flip moves a metric by ~4 points, so all observed differences sit within benchmark noise (generation at temperature 0.2 adds run-to-run variance).
3. **A 6× larger model changed nothing measurable.** Hosted `gpt-oss-120b` matched the local 20B on every metric, suggesting the ceiling here is set by retrieval and prompt design rather than model capacity — and justifying a free-tier model for the public demo.
4. **The citation metric was under-reporting, caught by error analysis.** Models cite in different formats: `[1]`, `【3】`, and `【1†L1-L3】`. The original pattern required a digit immediately followed by a closing bracket, so it scored correctly-cited answers as uncited (0.67 for the 120B, 0.88 for the 20B). Because the harness stores every raw answer next to its metrics, the corrected pattern was applied retroactively to all past runs without re-running a single LLM call — the true citation rate is 1.00 throughout. A metric never checked against raw outputs is a metric not yet worth trusting.

## Configuration

Every pipeline knob (chunk size/overlap, top-k, diversity cap, embedding model, corpus query, LLM endpoint) lives in `src/askarxiv/config.py`.

## Tests

```powershell
pytest -q
```
