"""Splits the regulations into chunks, one per provision.

A retrieved chunk then maps to exactly one citable article, recital or annex.
Only provisions longer than config.CHUNK_SIZE are split further.

Usage:
    python -m eulaw.chunk
"""

import json
import re

from eulaw import config
from eulaw import corpus


def clean_text(text: str) -> str:
    """Collapses runs of whitespace into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def sliding_windows(text: str,
                    size: int = config.CHUNK_SIZE,
                    overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Splits text into overlapping windows.

    The overlap means a sentence cut at a window boundary still appears whole
    in the next window.

    Args:
        text: The text to split.
        size: Window length in characters.
        overlap: Characters shared by consecutive windows.

    Returns:
        The windows in order; empty if the text is empty.

    Raises:
        ValueError: overlap is not smaller than size.
    """
    if overlap >= size:
        raise ValueError(f"overlap must be smaller than size: {overlap=}, "
                         f"{size=}")
    step = size - overlap
    return [text[start:start + size] for start in range(0, len(text), step)]


def chunk_document(doc: dict,
                   size: int = config.CHUNK_SIZE,
                   overlap: int = config.CHUNK_OVERLAP) -> list[dict]:
    """Splits one regulation into chunks.

    Args:
        doc: A regulation as produced by corpus.build().
        size: Maximum chunk length in characters, excluding the header.
        overlap: Characters shared by consecutive chunks of one provision.

    Returns:
        One dict per chunk with its id, text, regulation and provision.
    """
    chunks = []
    for section in doc["sections"]:
        # The header goes into the text, not just the metadata: the embedding
        # only sees the text, and both regulations have an "Article 6".
        header = f"{doc['title']}, {section['label']}"
        if section["subtitle"]:
            header += f" ({section['subtitle']})"
        text = clean_text(section["text"])
        for window in sliding_windows(text, size, overlap):
            index = len(chunks)
            chunks.append({
                "id": f"{doc['doc_id']}_{index}",
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "url": f"{doc['url']}#{section['anchor']}",
                "label": section["label"],
                "anchor": section["anchor"],
                "chunk_index": index,
                "text": f"{header}: {window}",
            })
    return chunks


def main() -> None:
    all_chunks = []
    for doc in corpus.load_documents():
        doc_chunks = chunk_document(doc)
        all_chunks.extend(doc_chunks)
        print(f"{doc['title']}: {len(doc['sections'])} provisions -> "
              f"{len(doc_chunks)} chunks")

    config.CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"wrote {len(all_chunks)} chunks to {config.CHUNKS_FILE}")


if __name__ == "__main__":
    main()
