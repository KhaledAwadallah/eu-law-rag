# AskArxiv

Ask questions about a collection of ML research papers and get answers grounded in those papers, with citations — a Retrieval-Augmented Generation (RAG) pipeline built from scratch.

**Pipeline:** arXiv ingestion → PDF parsing → chunking → embeddings (sentence-transformers) → ChromaDB vector search → LLM answer with citations. Answer quality is measured on a hand-built Q&A benchmark (faithfulness, relevancy, retrieval hit-rate).

## Status

- [x] Step 1 — project scaffold, environment, CI-ready structure
- [x] Step 2 — paper ingestion (arXiv API + PyMuPDF): 50 papers in `data/pdfs`, parsed text in `data/text`
- [x] Step 3 — chunking: 3,479 overlapping passages in `data/chunks.jsonl`
- [ ] Step 4 — embedding + vector index (ChromaDB)
- [ ] Step 5 — retrieval + grounded generation
- [ ] Step 6 — evaluation benchmark
- [ ] Step 7 — Gradio app, tests, Docker, CI
- [ ] Step 8 — deployment (Hugging Face Spaces)

## Setup

```powershell
.\setup.ps1        # creates .venv, installs dependencies, verifies, first commit
```

Or manually:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest
```

## Configuration

All pipeline knobs (chunk size, top-k, embedding model, corpus query) live in `src/askarxiv/config.py`.
