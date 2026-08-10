"""Assemble a self-contained Hugging Face Space from this project.

The Space differs from local development in three ways:
  1. no editable install     -> the package is copied as a top-level `askarxiv/`
  2. no Ollama (CPU only)    -> the LLM comes from a hosted OpenAI-compatible API
  3. no ingestion at runtime -> the prebuilt Chroma index ships with the Space

Run:  python deploy/build_space.py
Then push deploy/space/ to your Space repo (see README).
"""

import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "deploy" / "space"

# Runtime dependencies only: ingestion (arxiv, pymupdf) and dev tools
# (pytest, ruff) are not needed to serve the app, and every extra package
# slows the Space's build.
REQUIREMENTS = """\
gradio
chromadb
sentence-transformers
"""

# Hugging Face rejects non-LFS files over 10 MB, and the Chroma index holds a
# ~26 MB SQLite file, so the binary artifacts must be declared as LFS.
GITATTRIBUTES = """\
*.sqlite3 filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.pickle filter=lfs diff=lfs merge=lfs -text
"""

# Hugging Face reads this YAML front matter to configure the Space.
SPACE_README = """\
---
title: AskArxiv
emoji:
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
short_description: RAG over 50 LLM papers, with citations and grounded refusals
---

# AskArxiv

Ask questions about 50 recent LLM papers (arXiv, cs.CL). Answers are grounded in
the papers and cited; when the papers do not contain the answer, the system says
so instead of guessing.

Retrieval-Augmented Generation built from scratch: PDF ingestion, sliding-window
chunking, `bge-small-en-v1.5` embeddings, ChromaDB vector search with a
per-paper diversity cap, and grounded generation with a strict citation and
refusal contract.

Measured on a 32-question benchmark: retrieval hit-rate 1.00, refusal accuracy
1.00 on trap questions, faithfulness 0.94 (LLM-as-judge).

Source and evaluation details: https://github.com/KhaledAwadallah/askarxiv
"""


def build() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)          # always a clean build, never stale leftovers
    OUT.mkdir(parents=True)

    # 1. the package, promoted from src/ to the top level so plain
    #    `import askarxiv` works without any install step
    shutil.copytree(ROOT / "src" / "askarxiv", OUT / "askarxiv",
                    ignore=shutil.ignore_patterns("__pycache__"))

    # 2. the app itself
    shutil.copy(ROOT / "app.py", OUT / "app.py")

    # 3. the prebuilt vector index (~30 MB) - the Space cannot rebuild it
    shutil.copytree(ROOT / "data" / "chroma", OUT / "data" / "chroma")

    # 4. Space-specific metadata and dependencies
    (OUT / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (OUT / "README.md").write_text(SPACE_README, encoding="utf-8")
    (OUT / ".gitattributes").write_text(GITATTRIBUTES, encoding="utf-8")

    files = sum(1 for _ in OUT.rglob("*") if _.is_file())
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"built {OUT}")
    print(f"{files} files, {size / 1e6:.1f} MB")
    print("\nnext: push this folder to your Space, then set these in")
    print("Space settings -> Variables and secrets:")
    print("  LLM_BASE_URL = https://api.groq.com/openai/v1   (variable)")
    print("  LLM_MODEL    = openai/gpt-oss-120b              (variable)")
    print("  LLM_API_KEY  = <your Groq key>                  (secret)")


if __name__ == "__main__":
    build()
