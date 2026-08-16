# EU AI Act & GDPR — grounded question answering

[![CI](https://github.com/KhaledAwadallah/eu-law-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/KhaledAwadallah/eu-law-rag/actions/workflows/ci.yml)

Ask a question about EU tech regulation, get an answer grounded in the legal text, cited to the exact article and deep-linked to it on EUR-Lex. A RAG pipeline built from scratch, no framework.

**Corpus:** the AI Act (Reg. 2024/1689) and the GDPR (Reg. 2016/679) in full — 578 provisions, 908k characters, 1,357 chunks.

**Pipeline:** EUR-Lex XHTML → one chunk per provision → `bge-small-en-v1.5` embeddings → ChromaDB search under two ranking rules → grounded generation with citations and refusal (`gpt-oss:20b` via Ollama, or any OpenAI-compatible endpoint) → evaluation harness.

> **Not legal advice.** This reports what the consolidated text says; it does not track amendments, case law or national implementations.

## Why provisions

EUR-Lex serves each regulation as XHTML where every article, recital and annex sits in a `<div>` whose id is also its page anchor (`art_6`, `rct_47`, `anx_III`). So a chunk is a provision, and a citation is a link the reader can check in one click instead of searching a 100-page document.

Adding another regulation is one CELEX id in [config.py](src/eulaw/config.py) — e.g. the Data Act (`32023R2854`) or the DSA (`32022R2065`).

## Two ranking rules

Plain top-k similarity is wrong here in two measurable ways.

**Diversity is per provision, not per document.** With two regulations, capping per *document* starves the top-k instead of spreading it — k=5 returned 4 hits. AI Act Article 5 alone is 11k characters, so the unit that must not dominate is the article.

**Articles outrank recitals; embeddings disagree.** Recitals are explanatory preamble written in flowing prose, so they out-embed the binding articles on plain-language questions. *"What are the lawful bases for processing personal data?"* returned five recitals and no GDPR Article 6 — the provision that answers it — which sat at candidate rank 5. Fix: reserve 60% of the top-k for articles and annexes, then fill by score. No reranker.

## The code

Seven files, 361 lines of executable code, one per pipeline stage:

| file | what it does |
|---|---|
| [config.py](src/eulaw/config.py) | every setting: CELEX ids, chunk size, top-k, ranking rules, LLM endpoint, UI copy |
| [corpus.py](src/eulaw/corpus.py) | download from EUR-Lex, split into articles, recitals and annexes |
| [chunk.py](src/eulaw/chunk.py) | one chunk per provision, splitting only those too long to fit |
| [index.py](src/eulaw/index.py) | embed into ChromaDB |
| [retrieve.py](src/eulaw/retrieve.py) | vector search plus the two ranking rules |
| [generate.py](src/eulaw/generate.py) | prompt → LLM → answer with sources |

Plus `app.py` and `streamlit_app.py` (two front ends over the same `answer()` call) and `eval/run_eval.py`.

## Usage

```powershell
.\setup.ps1                                 # venv, dependencies, tests
python -m eulaw.corpus                      # download + parse from EUR-Lex
python -m eulaw.chunk                       # one chunk per provision
python -m eulaw.index                       # embed + build the index
python -m eulaw.retrieve "your question"    # inspect raw retrieval
python -m eulaw.generate "your question"    # full answer with sources
```

The prebuilt index ships in the repo, so the app and the retrieval evaluation run straight after install. Generation needs [Ollama](https://ollama.com) (`ollama pull gpt-oss:20b`) or any OpenAI-compatible provider via `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY`.

```powershell
python app.py                        # Gradio  -> http://localhost:7860
streamlit run streamlit_app.py       # Streamlit -> http://localhost:8501

docker build -t eu-law-rag .
docker run -p 7860:7860 -v "${PWD}/data:/app/data" -e LLM_BASE_URL=http://host.docker.internal:11434/v1 eu-law-rag
```

Deployed on Streamlit Community Cloud: embeddings and search on CPU, generation via a hosted API. Set `LLM_BASE_URL`, `LLM_MODEL` and the `LLM_API_KEY` **secret** in the platform; the index is committed so the app never runs ingestion.

## Evaluation

32 hand-written questions (`eval/questions.jsonl`): 26 answerable, each tagged with the exact provision that answers it, plus 6 traps with no answer in the corpus.

The traps are leak detectors, not softballs. Two ask about the **Digital Services Act** and the **Cyber Resilience Act** — EU regulation the model knows well, in the same register as the corpus, but absent from it.

```powershell
python eval\run_eval.py --no-llm           # retrieval metrics only (fast)
python eval\run_eval.py                    # full run incl. LLM-as-judge
python eval\run_eval.py --k 10 --name k10  # other retrieval depths
```

Each run stores every answer *and the excerpts behind it*, so re-analysis costs no further LLM calls.

### Results

k=5, local `gpt-oss:20b`, 32 questions (`eval/results/k5.json`):

| Metric | Value | |
|---|---|---|
| Document hit-rate | 1.00 | saturated by construction — only two documents |
| **Provision hit-rate** | **0.96** | 25/26 |
| False refusals | 0.00 | all 26 answerable questions answered |
| **Refusal accuracy** | **0.83** | **5/6 — one trap broke the grounding contract** |
| Citation rate | 1.00 | |
| **Misattribution rate** | **0.03** | 1/32 |
| Faithfulness | 0.96 | self-judged, see limitations |

### Findings

**1. One trap breaks the contract, and prompting does not fix it.** Asked what the **Cyber Resilience Act** requires, the system does not refuse. Retrieval hands it AI Act Recitals 77-78 and Articles 41/42/47 — all genuinely about conformity assessment and cybersecurity — and it answers *"The EU Cyber Resilience Act requires that products with digital elements undergo a conformity assessment…"*. Every excerpt is real and every claim appears in the sources; only the law it is attributed to is invented.

An explicit prompt rule against this was added and **measured as ineffective** — the trap failed again with the same opening sentence. It is kept but documented as insufficient. The failure is specific: the neighbouring DSA trap is refused correctly, because nothing in the corpus is close enough to bridge to. The model is not disobeying an instruction; it fails to notice that "the Act" in its excerpts and "the Act" in the question are different laws. A deterministic guard — running the `misattribution` check at answer time and forcing a refusal — is the honest fix.

**2. Provision hit-rate is the only retrieval metric with headroom.** Document hit-rate cannot be other than 1.00 with two documents. One question misses at provision level: *"What security measures must controllers and processors implement?"* returns GDPR Recital 95 and Articles 24/28/70 but not Article 32, titled *Security of processing*. The gold label was verified — a genuine retrieval failure.

**3. Cross-regulation bleed causes both PARTIAL verdicts.** *"What conditions must consent meet?"* retrieves GDPR Article 7 in the top three and AI Act Article 61 (consent for real-world testing) in slots 4-5, and the answer merges the two. Finding 1 in a milder form.

**4. The citation metric was under-reporting, caught by error analysis.** Models cite as `[1]`, `【3】` or `【1†L1-L3】`; the original pattern scored the last form as uncited. Because raw answers are stored, the fix was applied retroactively without re-running a single LLM call.

### Limitations

- **The judge is the model judging itself**, so faithfulness is an upper bound. Using a different judge is a config change, not a code change.
- **Faithfulness never scores traps** — exactly where finding 1 lives. Hence the separate string-match misattribution metric.
- **No answer-correctness metric.** Gold answers sit unused in each question's `note` field; faithfulness measures grounding, not rightness.
- **No retrieval baseline** — no BM25 floor, no reranker to compare against the reserved-slot heuristic.
- **HNSW is approximate**, so rebuilding the index can move a borderline result. Metrics are stable; rankings are not bit-reproducible.
- **Consolidated text only** — no amendments, case law or national implementations.

## Tests

```powershell
pytest -q        # 41 tests, no network, model or index required
```

Covering the EUR-Lex parser, provision chunking, the selection algorithm (diversity cap, reserved slots, no duplicates), prompt construction, and config/benchmark invariants.
