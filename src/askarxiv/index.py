"""Step 4: embed all chunks and store them in a ChromaDB vector index.

Provides:
    load_chunks(path) -> list[dict]      # read data/chunks.jsonl back in
    build_index(chunks, db_path) -> None # embed + store everything

Run the whole step:  python -m askarxiv.index
"""

import json
import pathlib

import chromadb
from sentence_transformers import SentenceTransformer

from askarxiv import config

COLLECTION = "papers"


def load_chunks(path: str = "data/chunks.jsonl") -> list[dict]:
    """Read the chunk file written by Step 3, one JSON object per line."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_index(chunks: list[dict] | None = None,
                db_path: str = config.DB_PATH) -> None:
    """Embed every chunk and store vectors + text + metadata in ChromaDB.

    Rebuilds the collection from scratch each run: after re-chunking with a
    different size, stale vectors must never linger next to new ones.
    """
    if chunks is None:
        chunks = load_chunks()

    model = SentenceTransformer(config.EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=db_path)

    try:
        client.delete_collection(COLLECTION)   # start fresh (see docstring)
    except Exception:
        pass                                   
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    batch = 256
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        texts = [c["text"] for c in part]
        vecs = model.encode(texts, batch_size=64, normalize_embeddings=True)
        col.add(
            ids=[c["id"] for c in part],
            embeddings=vecs.tolist(),
            documents=texts,
            metadatas=[{"paper_id": c["paper_id"],
                        "title": c["title"],
                        "chunk_index": c["chunk_index"]} for c in part],
        )
        print(f"indexed {min(i + batch, len(chunks))}/{len(chunks)} chunks")

    print(f"index complete: {col.count()} vectors in {pathlib.Path(db_path).resolve()}")


if __name__ == "__main__":
    build_index()
