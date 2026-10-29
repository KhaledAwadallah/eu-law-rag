"""Embeds the chunks into a ChromaDB vector index.

Usage:
    python -m eulaw.index
"""

import json

import chromadb
import sentence_transformers

from eulaw import config

# Stored with every vector; retrieval and the app read them back.
_METADATA_FIELDS = ("doc_id", "title", "url", "label", "anchor", "chunk_index")
_BATCH_SIZE = 256


def load_chunks() -> list[dict]:
    """Reads chunks.jsonl as written by `python -m eulaw.chunk`.

    Raises:
        FileNotFoundError: The chunking step has not been run yet.
    """
    if not config.CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"{config.CHUNKS_FILE} not found; run `python -m eulaw.chunk` "
            "first")
    with open(config.CHUNKS_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_index(chunks: list[dict]) -> None:
    """Embeds the chunks and stores vectors, text and metadata in ChromaDB.

    Any existing collection is replaced, so no stale vectors survive a
    re-chunking.

    Args:
        chunks: Chunks as produced by chunk.chunk_document().
    """
    model = sentence_transformers.SentenceTransformer(
        config.EMBEDDING_MODEL, device=config.EMBEDDING_DEVICE)
    client = chromadb.PersistentClient(path=str(config.DB_PATH))
    if config.COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(config.COLLECTION)
    collection = client.create_collection(
        config.COLLECTION, metadata={"hnsw:space": "cosine"})

    for start in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[start:start + _BATCH_SIZE]
        texts = [c["text"] for c in batch]
        vectors = model.encode(texts, batch_size=64, normalize_embeddings=True)
        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=vectors.tolist(),
            documents=texts,
            metadatas=[{f: c[f] for f in _METADATA_FIELDS} for c in batch],
        )
        print(f"indexed {start + len(batch)}/{len(chunks)} chunks")


if __name__ == "__main__":
    build_index(load_chunks())
