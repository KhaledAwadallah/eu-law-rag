"""One chunk per provision, so a retrieved chunk maps to one citable unit.

    python -m eulaw.chunk
"""

import json
import re

from eulaw import config


def clean_text(text: str) -> str:
    """Collapse whitespace runs to single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def sliding_windows(text: str,
                    size: int = config.CHUNK_SIZE,
                    overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping windows, so a sentence cut at a boundary
    survives intact in the next one."""
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")
    if not text:
        return []

    step = size - overlap
    return [text[start:start + size] for start in range(0, len(text), step)]


def chunk_document(doc: dict,
                   size: int = config.CHUNK_SIZE,
                   overlap: int = config.CHUNK_OVERLAP) -> list[dict]:
    """Split one regulation, splitting only provisions too long to fit."""
    chunks: list[dict] = []
    for section in doc["sections"]:
        for window in sliding_windows(clean_text(section["text"]), size, overlap):
            # The header goes in the text, not just the metadata: the embedding
            # sees only the text, and both regulations have an "Article 6".
            header = f"{doc['title']}, {section['label']}"
            if section["subtitle"]:
                header += f" ({section['subtitle']})"
            index = len(chunks)
            chunks.append({
                "id": f"{doc['doc_id']}_{index}",
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "url": f"{doc['url']}#{section['anchor']}",   # deep-link
                "label": section["label"],
                "anchor": section["anchor"],
                "chunk_index": index,
                "text": f"{header}: {window}",
            })
    return chunks


def chunk_corpus() -> list[dict]:
    """Chunk every regulation and write chunks.jsonl."""
    from eulaw.corpus import load_documents

    all_chunks: list[dict] = []
    for doc in load_documents():
        doc_chunks = chunk_document(doc)
        all_chunks.extend(doc_chunks)
        print(f"{doc['doc_id']} ({doc['title']}): {len(doc['sections'])} "
              f"provisions -> {len(doc_chunks)} chunks")

    out = config.CHUNKS_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    docs = len({c["doc_id"] for c in all_chunks})
    print(f"done: {len(all_chunks)} chunks from {docs} regulations -> {out}")
    return all_chunks


if __name__ == "__main__":
    chunk_corpus()
