"""Unit tests for the chunking step."""

import pytest

from askarxiv.chunk import chunk_text, clean_text


def test_clean_text_flattens_whitespace():
    assert clean_text("a\nb\t c  \n\n d ") == "a b c d"


def test_short_text_gives_single_chunk():
    chunks = chunk_text("hello world", "p1", "T", size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "hello world"


def test_chunks_cover_all_text_with_overlap():
    text = "x" * 2500
    chunks = chunk_text(text, "p1", "T", size=1000, overlap=150)
    # windows start every size-overlap=850 chars: 0, 850, 1700, 2550(no) -> 3 chunks
    assert len(chunks) == 3
    # reconstruction: dropping each chunk's first `overlap` chars (except the
    # first chunk) and concatenating must give back the original text
    rebuilt = chunks[0]["text"] + "".join(c["text"][150:] for c in chunks[1:])
    assert rebuilt == text


def test_consecutive_chunks_share_overlap():
    text = "".join(chr(65 + i % 26) for i in range(3000))  # ABC...XYZABC...
    chunks = chunk_text(text, "p1", "T", size=1000, overlap=150)
    for a, b in zip(chunks, chunks[1:]):
        assert a["text"][-150:] == b["text"][:150]


def test_ids_are_unique_and_ordered():
    chunks = chunk_text("y" * 5000, "paper42", "T", size=1000, overlap=100)
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids[0] == "paper42_0"
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("abc", "p1", "T", size=100, overlap=100)
