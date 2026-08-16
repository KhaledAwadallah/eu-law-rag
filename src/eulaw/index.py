"""Embed the chunks into a ChromaDB vector index.

    python -m eulaw.index
"""

import json
import pathlib

from eulaw import config

# Copied onto every vector; retrieval and the UI read it back.
METADATA_FIELDS = ("doc_id", "title", "url", "label", "anchor", "chunk_index")


def load_chunks() -> list[dict]:
    if not config.CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"{config.CHUNKS_FILE} not found - run `python -m eulaw.chunk` first"
        )
    with open(config.CHUNKS_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_index(chunks: list[dict] | None = None) -> None:
    """Embed every chunk and store vectors, text and metadata."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    if chunks is None:
        chunks = load_chunks()

    model = SentenceTransformer(config.EMBEDDING_MODEL, device=config.EMBEDDING_DEVICE)
    client = chromadb.PersistentClient(path=config.DB_PATH)

    try:
        # Rebuild from scratch: after re-chunking, stale vectors must not linger.
        client.delete_collection(config.COLLECTION)
    except Exception:
        pass
    col = client.create_collection(config.COLLECTION,
                                   metadata={"hnsw:space": "cosine"})

    batch = 256
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        texts = [c["text"] for c in part]
        vecs = model.encode(texts, batch_size=64, normalize_embeddings=True)
        col.add(
            ids=[c["id"] for c in part],
            embeddings=vecs.tolist(),
            documents=texts,
            metadatas=[{f: c[f] for f in METADATA_FIELDS} for c in part],
        )
        print(f"indexed {min(i + batch, len(chunks))}/{len(chunks)} chunks")

    print(f"index complete: {col.count()} vectors in "
          f"{pathlib.Path(config.DB_PATH).resolve()}")


if __name__ == "__main__":
    build_index()
