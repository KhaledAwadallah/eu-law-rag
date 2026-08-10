import os

# --- Chunking ---
CHUNK_SIZE = 1000     # characters per chunk; ~200 words
CHUNK_OVERLAP = 150   # characters shared between consecutive chunks

# --- Retrieval ---
TOP_K = 5             # how many chunks are retrieved per question
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"   # small, strong, runs on CPU
# None = let sentence-transformers pick (GPU locally). Set EMBEDDING_DEVICE=cpu
# where CUDA is detected but not usable, e.g. inside a ZeroGPU container.
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE") or None

# --- Storage ---
DB_PATH = "data/chroma"        # ChromaDB persistence directory
PDF_DIR = "data/pdfs"          # downloaded papers
METADATA_FILE = "data/metadata.json"

# --- Corpus ---
# Field-prefixed arXiv query: cat: restricts to a category, abs: searches
# abstracts, quotes make phrases. This keeps the corpus on-topic, unlike a
# bare keyword query which matches those words anywhere in any paper.
ARXIV_QUERY = 'cat:cs.CL AND abs:"large language model"'
N_PAPERS = 50

# --- Retrieval diversity ---
MAX_CHUNKS_PER_PAPER = 2   # cap per paper in top-k so answers cite multiple sources

# --- Generation ---
# Any OpenAI-compatible endpoint works: Ollama locally (default), or a hosted
# provider (e.g. Groq) via environment variables - no code change needed.
# Env overrides matter for Docker/deployment, where config arrives from outside.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-oss:20b")
LLM_API_KEY_ENV = "LLM_API_KEY"              # name of the env var holding the key
LLM_TEMPERATURE = 0.2                        # low = factual, less creative drift
