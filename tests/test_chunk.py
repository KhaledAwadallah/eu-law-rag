"""Unit tests for the chunking step."""

import pytest

from eulaw.chunk import chunk_document, clean_text, sliding_windows


def _document(*sections):
    return {
        "doc_id": "32016R0679",
        "title": "GDPR",
        "url": "https://eur-lex.europa.eu/x",
        "sections": [
            {"anchor": a, "label": lb, "subtitle": st, "text": tx}
            for a, lb, st, tx in sections
        ],
    }


def test_clean_text_flattens_whitespace():
    assert clean_text("a\nb\t c  \n\n d ") == "a b c d"


def test_short_text_gives_single_window():
    assert sliding_windows("hello world", size=100, overlap=10) == ["hello world"]


def test_windows_cover_all_text_with_overlap():
    text = "x" * 2500
    windows = sliding_windows(text, size=1000, overlap=150)
    # windows start every size-overlap=850 chars: 0, 850, 1700, 2550(no) -> 3
    assert len(windows) == 3
    # reconstruction: dropping each window's first `overlap` chars (except the
    # first) and concatenating must give back the original text
    assert windows[0] + "".join(w[150:] for w in windows[1:]) == text


def test_consecutive_windows_share_overlap():
    text = "".join(chr(65 + i % 26) for i in range(3000))  # ABC...XYZABC...
    windows = sliding_windows(text, size=1000, overlap=150)
    for a, b in zip(windows, windows[1:]):
        assert a[-150:] == b[:150]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        sliding_windows("abc", size=100, overlap=100)


def test_one_chunk_per_provision():
    doc = _document(("art_6", "Article 6", "Lawfulness", "short article text"),
                    ("rct_40", "Recital 40", "", "short recital text"))
    chunks = chunk_document(doc)
    assert len(chunks) == 2
    assert [c["label"] for c in chunks] == ["Article 6", "Recital 40"]


def test_each_chunk_deep_links_to_its_own_provision():
    doc = _document(("art_6", "Article 6", "Lawfulness", "text"),
                    ("anx_III", "Annex III", "", "more text here"))
    chunks = chunk_document(doc)
    assert chunks[0]["url"] == "https://eur-lex.europa.eu/x#art_6"
    assert chunks[1]["url"] == "https://eur-lex.europa.eu/x#anx_III"


def test_long_provisions_split_but_keep_their_label_and_anchor():
    doc = _document(("art_5", "Article 5", "Prohibited practices", "z" * 2500))
    chunks = chunk_document(doc)
    assert len(chunks) > 1
    assert all(c["label"] == "Article 5" for c in chunks)
    assert all(c["anchor"] == "art_5" for c in chunks)


def test_chunks_name_their_regulation_and_provision_inline():
    # The embedding is computed from the text alone, and both regulations have
    # an "Article 6" - so the text itself must say which one this is.
    doc = _document(("art_6", "Article 6", "Lawfulness", "body text"))
    text = chunk_document(doc)[0]["text"]
    assert text.startswith("GDPR, Article 6 (Lawfulness): ")
    assert "body text" in text


def test_ids_are_unique_and_indices_ordered():
    doc = _document(("art_5", "Article 5", "", "z" * 2500),
                    ("art_6", "Article 6", "", "short"))
    chunks = chunk_document(doc)
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids[0] == "32016R0679_0"
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
