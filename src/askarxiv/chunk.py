"""Split paper text into overlapping passages ("chunks").

Provides:
    clean_text(text) -> str                      # normalize whitespace
    chunk_text(text, paper_id, title, ...) -> list[dict]
    chunk_corpus(txt_dir, out_file) -> list[dict]  # chunk every paper, save JSONL

Command line:
    python -m askarxiv.chunk
"""

import json
import pathlib
import re

from askarxiv import config


def clean_text(text: str) -> str:
    """Collapse all whitespace runs (newlines, tabs, spaces) to single spaces.

    PDF extraction inserts a line break after every visual line, cutting
    sentences into fragments. Embedding models work best on flowing prose,
    so we flatten the text before chunking.
    """
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, paper_id: str, title: str,
               size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> list[dict]:
    """Split one paper's text into overlapping windows of `size` characters.

    Consecutive chunks share `overlap` characters, so a sentence cut at a
    boundary survives intact in the neighboring chunk.
    """
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")

    chunks = []
    step = size - overlap
    start = 0
    i = 0
    while start < len(text):
        piece = text[start:start + size]
        chunks.append({
            "id": f"{paper_id}_{i}",
            "paper_id": paper_id,
            "title": title,
            "chunk_index": i,
            "text": piece,
        })
        start += step
        i += 1
    return chunks


def chunk_corpus(txt_dir: str = "data/text",
                 out_file: str = "data/chunks.jsonl") -> list[dict]:
    """Chunk every parsed paper and write one JSON object per line (JSONL).

    Titles are looked up from the metadata file so every chunk carries its
    paper's title.
    """
    meta_list = json.loads(pathlib.Path(config.METADATA_FILE).read_text(encoding="utf-8"))

    all_chunks = []
    for m in sorted(meta_list, key=lambda m: m["id"]):
        stem = m["file"].removesuffix(".pdf")
        txt = pathlib.Path(txt_dir) / f"{stem}.txt"
        if not txt.exists():
            print(f"no text file for {stem} (parse failed?), skipping")
            continue
        title = m["title"]
        text = clean_text(txt.read_text(encoding="utf-8"))
        paper_chunks = chunk_text(text, paper_id=stem, title=title)
        all_chunks.extend(paper_chunks)
        print(f"{stem}: {len(text):,} chars -> {len(paper_chunks)} chunks")

    out = pathlib.Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    papers = len({c["paper_id"] for c in all_chunks})
    print(f"done: {len(all_chunks)} chunks from {papers} papers -> {out}")
    return all_chunks


if __name__ == "__main__":
    chunk_corpus()
