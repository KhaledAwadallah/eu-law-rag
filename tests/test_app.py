"""Unit tests for the web app's pure helpers."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))  # project root
from app import format_sources  # noqa: E402  (import needs the path tweak above)


def test_format_sources_links_each_provision():
    sources = [
        {"title": "AI Act", "label": "Article 6", "score": 0.765,
         "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689#art_6"},
        {"title": "GDPR", "label": "Recital 40", "score": 0.7,
         "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679#rct_40"},
    ]
    out = format_sources(sources)
    assert "1. [AI Act, Article 6](" in out
    assert "#art_6)" in out            # deep-links to the exact article
    assert "2. [GDPR, Recital 40]" in out
    assert "score 0.77" in out         # rounded to 2 decimals


def test_format_sources_empty():
    assert format_sources([]) == ""
