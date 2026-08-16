"""Vector search, plus two ranking rules that plain top-k gets wrong here.

    diversity   at most MAX_CHUNKS_PER_PROVISION chunks from one article, so a
                long provision cannot fill the whole context.
    authority   part of the top-k is reserved for binding provisions. Recitals
                are prose and out-embed articles: "What are the lawful bases
                for processing personal data?" returns five recitals and no
                Article 6, which sits at candidate rank 5.

    python -m eulaw.retrieve "when is an AI system high-risk?"
"""

import argparse
import math
import sys
from functools import lru_cache

from eulaw import config
from eulaw.corpus import provision_kind

# BGE models expect this in front of queries; it measurably improves retrieval.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL, device=config.EMBEDDING_DEVICE)


@lru_cache(maxsize=1)
def _collection():
    import chromadb

    if not config.has_index():
        raise FileNotFoundError(
            f"no index at {config.DB_PATH} - run `python -m eulaw.index` first"
        )
    return chromadb.PersistentClient(path=config.DB_PATH).get_collection(
        config.COLLECTION)


def retrieve(question: str, k: int = config.TOP_K,
             max_per_provision: int = config.MAX_CHUNKS_PER_PROVISION) -> list[dict]:
    """Top-k chunks. Over-fetches 4*k so both rules have room to work."""
    vec = _model().encode([QUERY_PREFIX + question], normalize_embeddings=True)
    res = _collection().query(query_embeddings=vec.tolist(), n_results=4 * k)

    candidates = [
        {
            "text": doc,
            "doc_id": meta["doc_id"],
            "title": meta["title"],
            "label": meta["label"],          # "Article 6"
            "anchor": meta["anchor"],        # "art_6" - also the diversity key
            "url": meta["url"],
            "chunk_index": meta["chunk_index"],
            "score": round(1.0 - dist, 4),   # cosine distance -> similarity
            "_id": chunk_id,
        }
        for chunk_id, doc, meta, dist in zip(res["ids"][0],
                                             res["documents"][0],
                                             res["metadatas"][0],
                                             res["distances"][0])
    ]

    hits = select(candidates, k, max_per_provision)
    if len(hits) < k:
        # A silently short context would look like a retrieval failure later.
        print(f"note: only {len(hits)} of {k} chunks survived the "
              f"max {max_per_provision}-per-provision cap")
    return [{key: value for key, value in h.items() if key != "_id"} for h in hits]


def select(candidates: list[dict], k: int = config.TOP_K,
           max_per_provision: int = config.MAX_CHUNKS_PER_PROVISION) -> list[dict]:
    """Apply both rules to best-first candidates; returns them in score order.

    Separate from the search so it can be tested without an index or a model.
    """
    hits: list[dict] = []
    per_provision: dict[str, int] = {}
    taken: set[str] = set()

    def fill(limit: int, primary_only: bool) -> None:
        for c in candidates:
            if len(hits) >= limit:
                return
            if c["_id"] in taken:
                continue
            if primary_only and not is_primary(c):
                continue
            if per_provision.get(c["anchor"], 0) >= max_per_provision:
                continue
            per_provision[c["anchor"]] = per_provision.get(c["anchor"], 0) + 1
            taken.add(c["_id"])
            hits.append(c)

    if config.MIN_PRIMARY:
        fill(min(k, math.ceil(k * config.MIN_PRIMARY)), primary_only=True)
    fill(k, primary_only=False)

    # Reserved slots are filled out of order; restore it so [1] is the best match.
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits


def is_primary(chunk: dict) -> bool:
    return provision_kind(chunk["label"]) in config.PRIMARY_KINDS


def cite(hit: dict) -> str:
    """'GDPR, Article 6'."""
    return f"{hit['title']}, {hit['label']}" if hit["label"] else hit["title"]


if __name__ == "__main__":
    # Legal text has characters the Windows console cannot encode by default.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", nargs="*", default=[])
    parser.add_argument("--k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    query = " ".join(args.question) or config.EXAMPLES[0]
    print(f"query: {query}\n")
    for hit in retrieve(query, k=args.k):
        print(f"[{hit['score']:.3f}] {cite(hit)[:70]}  (chunk {hit['chunk_index']})")
        print(f"        {hit['text'][:160]}...\n")
