"""Vector search plus two ranking rules that plain top-k gets wrong here.

Diversity: at most config.MAX_CHUNKS_PER_PROVISION chunks come from one
provision, so a long article (AI Act Article 5 is 11k characters) cannot fill
the whole context.

Authority: part of the top-k is reserved for articles and annexes. Recitals are
explanatory prose and out-embed the binding text on plain-language questions:
"What are the lawful bases for processing personal data?" returns five recitals
and no GDPR Article 6 without this rule.

Usage:
    python -m eulaw.retrieve "When is an AI system high-risk?"
"""

import argparse
import functools
import math
import sys

from eulaw import config
from eulaw import corpus

# BGE models expect this prefix on queries (but not on passages).
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# chromadb and sentence_transformers are imported inside these two functions:
# loading torch takes several seconds, and select() is used without either.
@functools.lru_cache(maxsize=1)
def _model():
    import sentence_transformers  # pylint: disable=import-outside-toplevel

    return sentence_transformers.SentenceTransformer(
        config.EMBEDDING_MODEL, device=config.EMBEDDING_DEVICE)


@functools.lru_cache(maxsize=1)
def _collection():
    import chromadb  # pylint: disable=import-outside-toplevel

    if not config.DB_PATH.is_dir():
        raise FileNotFoundError(
            f"No index at {config.DB_PATH}; run `python -m eulaw.index` first")
    client = chromadb.PersistentClient(path=str(config.DB_PATH))
    return client.get_collection(config.COLLECTION)


def warm_up() -> None:
    """Loads the embedding model and opens the index before the first query."""
    _model()
    _collection()


def retrieve(question: str,
             k: int = config.TOP_K,
             max_per_provision: int = config.MAX_CHUNKS_PER_PROVISION,
             min_primary: float = config.MIN_PRIMARY) -> list[dict]:
    """Finds the chunks most relevant to a question.

    Args:
        question: The user's question.
        k: How many chunks to return.
        max_per_provision: Cap on chunks from one provision.
        min_primary: Share of k reserved for articles and annexes.

    Returns:
        Up to k chunks, best match first, each with its text, regulation,
        provision label, EUR-Lex URL and similarity score.
    """
    vector = _model().encode([_QUERY_PREFIX + question],
                             normalize_embeddings=True)
    # Over-fetch so both ranking rules have candidates to choose from.
    result = _collection().query(query_embeddings=vector.tolist(),
                                 n_results=4 * k)
    candidates = []
    for chunk_id, text, meta, distance in zip(result["ids"][0],
                                              result["documents"][0],
                                              result["metadatas"][0],
                                              result["distances"][0]):
        candidates.append({
            "id": chunk_id,
            "text": text,
            "doc_id": meta["doc_id"],
            "title": meta["title"],
            "label": meta["label"],
            "anchor": meta["anchor"],
            "url": meta["url"],
            "chunk_index": meta["chunk_index"],
            "score": round(1.0 - distance, 4),  # Cosine distance to similarity.
        })
    return select(candidates, k, max_per_provision, min_primary)


def select(candidates: list[dict],
           k: int = config.TOP_K,
           max_per_provision: int = config.MAX_CHUNKS_PER_PROVISION,
           min_primary: float = config.MIN_PRIMARY) -> list[dict]:
    """Applies both ranking rules to candidates sorted best first.

    Kept separate from the search so it can be tested without an index.

    Args:
        candidates: Chunks with at least 'id', 'label', 'anchor' and 'score',
            sorted by descending score.
        k: How many chunks to return.
        max_per_provision: Cap on chunks from one provision (one anchor).
        min_primary: Share of k reserved for articles and annexes, used only
            as far as such candidates exist.

    Returns:
        Up to k candidates in descending score order. Fewer than k when the
        cap rules out the rest.
    """
    hits = []
    taken = set()
    per_provision = {}

    def fill(limit: int, primary_only: bool) -> None:
        for candidate in candidates:
            if len(hits) >= limit:
                return
            if candidate["id"] in taken:
                continue
            if primary_only and not is_primary(candidate):
                continue
            anchor = candidate["anchor"]
            if per_provision.get(anchor, 0) >= max_per_provision:
                continue
            per_provision[anchor] = per_provision.get(anchor, 0) + 1
            taken.add(candidate["id"])
            hits.append(candidate)

    fill(math.ceil(k * min_primary), primary_only=True)
    fill(k, primary_only=False)
    # The reserved slots were filled first; restore score order.
    hits.sort(key=lambda hit: hit["score"], reverse=True)
    return hits


def is_primary(chunk: dict) -> bool:
    """Whether a chunk comes from an article or an annex."""
    return corpus.provision_kind(chunk["label"]) in config.PRIMARY_KINDS


def cite(chunk: dict) -> str:
    """Names a chunk's source, e.g. 'GDPR, Article 6'."""
    if not chunk["label"]:
        return chunk["title"]
    return f"{chunk['title']}, {chunk['label']}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Show raw retrieval results.")
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    args = parser.parse_args()
    # Legal text has characters a Windows console cannot always encode.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    for hit in retrieve(args.question, k=args.k):
        print(f"[{hit['score']:.3f}] {cite(hit)}")
        print(f"        {hit['text'][:160]}...\n")


if __name__ == "__main__":
    main()
