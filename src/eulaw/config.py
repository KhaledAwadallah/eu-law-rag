"""Every setting for the pipeline."""

import os
import pathlib

# --- Corpus ---
# EUR-Lex CELEX ids. Add a regulation here and re-run the pipeline.
DOCUMENTS = [
    {"celex": "32024R1689", "title": "AI Act",
     "full_title": "Regulation (EU) 2024/1689 (Artificial Intelligence Act)"},
    {"celex": "32016R0679", "title": "GDPR",
     "full_title": "Regulation (EU) 2016/679 (General Data Protection Regulation)"},
]

TITLE = "EU AI Act & GDPR"
DESCRIPTION = (
    "The EU Artificial Intelligence Act (Regulation (EU) 2024/1689) and the "
    "General Data Protection Regulation (Regulation (EU) 2016/679), in full: "
    "articles, recitals and annexes, straight from EUR-Lex."
)
DISCLAIMER = (
    "Not legal advice. This answers what the consolidated text of these two "
    "regulations says; it does not track amendments, case law or national "
    "implementations. Always check the linked provision."
)
EXAMPLES = (
    "When is an AI system classified as high-risk?",
    "Which AI practices are prohibited outright?",
    "What are the lawful bases for processing personal data?",
    "How quickly must a personal data breach be notified, and to whom?",
    "What is the capital of Austria?",   # demonstrates the refusal contract
)

# --- Storage ---
DATA_DIR = pathlib.Path("data")
RAW_DIR = DATA_DIR / "raw"                     # XHTML as downloaded
DOCUMENTS_FILE = DATA_DIR / "documents.jsonl"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
DB_PATH = str(DATA_DIR / "chroma")             # ChromaDB wants a str
COLLECTION = "eu-law"

# --- Chunking ---
CHUNK_SIZE = 1000     # characters; only provisions longer than this are split
CHUNK_OVERLAP = 150   # characters shared between consecutive windows

# --- Retrieval ---
TOP_K = 5
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"   # small, strong, runs on CPU
# Set to "cpu" where CUDA is detected but unusable; None lets torch choose.
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE") or None

# Cap per provision, not per document: with only two regulations, a per-document
# cap leaves the top-k short instead of diverse.
MAX_CHUNKS_PER_PROVISION = 2

# Recitals are explanatory preamble and out-embed the binding articles on
# plain-language questions, so reserve part of the top-k for articles/annexes.
PRIMARY_KINDS = ("Article", "Annex")
MIN_PRIMARY = 0.6     # fraction of k reserved, when any are in reach

# --- Generation ---
# Any OpenAI-compatible endpoint: Ollama locally, or a hosted provider via env.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-oss:20b")
LLM_API_KEY_ENV = "LLM_API_KEY"
LLM_TEMPERATURE = 0.2
CITATION_HINT = (
    " When an excerpt is a numbered provision, name it in your answer "
    "(for example \"Article 6(2) of the AI Act\")."
)


def has_index() -> bool:
    return pathlib.Path(DB_PATH).is_dir()
