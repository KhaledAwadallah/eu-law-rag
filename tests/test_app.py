"""Unit tests for the web app's pure helpers."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))  # project root
from app import format_sources  # noqa: E402  (import needs the path tweak above)


def test_format_sources_links_and_numbering():
    sources = [
        {"paper_id": "2607.27591v1", "title": "Prox", "chunk_index": 2, "score": 0.765},
        {"paper_id": "2607.26891v1", "title": "DIRECT", "chunk_index": 4, "score": 0.7},
    ]
    out = format_sources(sources)
    assert "1. [Prox](https://arxiv.org/abs/2607.27591v1)" in out
    assert "2. [DIRECT]" in out
    assert "score 0.77" in out          # rounded to 2 decimals


def test_format_sources_empty():
    assert format_sources([]) == ""
