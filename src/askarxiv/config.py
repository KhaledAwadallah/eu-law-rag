"""Central configuration — every tunable knob of the pipeline lives here.

Later (Step 6, evaluation) we will vary these values systematically and
measure their effect on answer quality. Keeping them in one file means an
experiment is a one-line change, not a hunt through the codebase.
"""

# --- Chunking (Step 3) ---
CHUNK_SIZE = 1000     # characters per chunk; ~200 words
CHUNK_OVERLAP = 150   # characters shared between consecutive chunks

# --- Retrieval (Steps 4-5) ---
TOP_K = 5             # how many chunks are retrieved per question
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"   # small, strong, runs on CPU

# --- Storage ---
DB_PATH = "data/chroma"        # ChromaDB persistence directory
PDF_DIR = "data/pdfs"          # downloaded papers
METADATA_FILE = "data/metadata.json"

# --- Corpus (Step 2) ---
ARXIV_QUERY = "LLM efficiency"   # theme of the paper collection
N_PAPERS = 50
