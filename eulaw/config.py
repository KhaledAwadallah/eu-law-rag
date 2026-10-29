"""Settings for the whole pipeline."""

import os
import pathlib

# Regulations to index, by EUR-Lex CELEX id. Add one and re-run the pipeline.
DOCUMENTS = (
    {"celex": "32024R1689", "title": "AI Act"},  # Regulation (EU) 2024/1689
    {"celex": "32016R0679", "title": "GDPR"},  # Regulation (EU) 2016/679
)

# Storage.
DATA_DIR = pathlib.Path("data")
RAW_DIR = DATA_DIR / "raw"
DOCUMENTS_FILE = DATA_DIR / "documents.jsonl"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
DB_PATH = DATA_DIR / "chroma"
COLLECTION = "eu-law"

# Chunking. Only provisions longer than CHUNK_SIZE characters are split.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Retrieval.
TOP_K = 5
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# Set to "cpu" where CUDA is detected but unusable; None lets torch choose.
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE") or None
# At most this many chunks from one article, recital or annex.
MAX_CHUNKS_PER_PROVISION = 2
# Share of the top-k reserved for binding provisions (articles and annexes).
PRIMARY_KINDS = ("Article", "Annex")
MIN_PRIMARY = 0.6

# Generation. Any OpenAI-compatible endpoint: Ollama locally, a hosted API in
# the deployed demo.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-oss:20b")
LLM_TEMPERATURE = 0.2
