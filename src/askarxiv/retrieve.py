""" Similarity search - find the chunks most relevant to a question.

Provides:
    retrieve(question, k) -> list[dict]   # top-k chunks with titles and scores

Command line:
    python -m askarxiv.retrieve "how can LLM inference be made faster?"
"""

import sys
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from askarxiv import config

# BGE models are trained to expect this instruction in front of QUERIES;
# it noticeably improves retrieval quality.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    """Load the embedding model once and reuse it across calls."""
    return SentenceTransformer(config.EMBEDDING_MODEL, device=config.EMBEDDING_DEVICE)


@lru_cache(maxsize=1)
def _collection():
    """Open the ChromaDB collection once and reuse the handle."""
    client = chromadb.PersistentClient(path=config.DB_PATH)
    return client.get_collection("papers")


def retrieve(question: str, k: int = config.TOP_K,
             max_per_paper: int = config.MAX_CHUNKS_PER_PAPER) -> list[dict]:
    """Return the k best chunks, with at most max_per_paper from any one paper.

    Without the cap, one strongly-matching paper can fill the whole top-k,
    leaving the LLM a single source to answer from. We over-fetch (4*k
    candidates), then walk them best-first, skipping chunks from papers that
    already hit the cap, until k survive.
    """
    vec = _model().encode([QUERY_PREFIX + question], normalize_embeddings=True)
    res = _collection().query(query_embeddings=vec.tolist(), n_results=4 * k)

    hits = []
    per_paper: dict[str, int] = {}
    for doc, meta, dist in zip(res["documents"][0],
                               res["metadatas"][0],
                               res["distances"][0]):
        pid = meta["paper_id"]
        if per_paper.get(pid, 0) >= max_per_paper:
            continue
        per_paper[pid] = per_paper.get(pid, 0) + 1
        hits.append({
            "text": doc,
            "paper_id": pid,
            "title": meta["title"],
            "chunk_index": meta["chunk_index"],
            "score": round(1.0 - dist, 4),   # cosine distance -> similarity
        })
        if len(hits) == k:
            break
    return hits


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "How can LLM inference be made more efficient?"
    print(f"query: {query}\n")
    for hit in retrieve(query):
        print(f"[{hit['score']:.3f}] {hit['title'][:65]}  (chunk {hit['chunk_index']})")
        print(f"        {hit['text'][:160]}...\n")
