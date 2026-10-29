"""Downloads the regulations from EUR-Lex and splits them into provisions.

EUR-Lex serves each regulation as XHTML in which every article, recital and
annex sits in a div whose id is also its page anchor (art_6, rct_47, anx_III).
That anchor is what lets a citation link to the exact provision.

Usage:
    python -m eulaw.corpus
"""

import json
import pathlib
import re
import urllib.request

import bs4

from eulaw import config

HTML_URL = (
    "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}"
)
_USER_AGENT = "eu-law-rag/0.1 (github.com/KhaledAwadallah/eu-law-rag)"

# Chapters and sections are skipped: they only wrap articles, so indexing them
# would return the same text again under a less precise anchor.
_SECTION_ID = re.compile(r"^(art_\d+|rct_\d+|anx_[IVXLC]+)$")

# Anything shorter is a heading or a cross-reference stub.
MIN_SECTION_CHARS = 120

_PROVISION_NAMES = {"art": "Article", "rct": "Recital", "anx": "Annex"}


def anchor_label(anchor: str) -> str:
    """Turns an anchor into a label, e.g. 'art_6' into 'Article 6'."""
    kind, _, number = anchor.partition("_")
    return f"{_PROVISION_NAMES[kind]} {number}"


def provision_kind(label: str) -> str:
    """Returns 'Article', 'Recital' or 'Annex' for a label like 'Article 6'."""
    return label.split()[0] if label else ""


def _normalize(text: str) -> str:
    """Collapses whitespace, including EUR-Lex's non-breaking spaces."""
    return re.sub(r"\s+", " ", text).strip()


def parse_sections(html: str) -> list[dict]:
    """Extracts every article, recital and annex from one regulation.

    Args:
        html: The XHTML page of a regulation as served by EUR-Lex.

    Returns:
        One dict per provision with the keys 'anchor', 'label', 'subtitle'
        (e.g. 'Human oversight', or '' for recitals) and 'text'.
    """
    soup = bs4.BeautifulSoup(html, "html.parser")
    sections = []
    for div in soup.find_all("div", id=_SECTION_ID):
        # get_text walks the nested title divs, paragraphs and the tables that
        # hold lettered points, so a provision comes back as one string.
        text = _normalize(div.get_text(" "))
        if len(text) < MIN_SECTION_CHARS:
            continue
        subtitle = div.find("p", class_="oj-sti-art")
        sections.append({
            "anchor": div["id"],
            "label": anchor_label(div["id"]),
            "subtitle": _normalize(subtitle.get_text(" ")) if subtitle else "",
            "text": text,
        })
    return sections


def download(celex: str, dest: pathlib.Path) -> str:
    """Fetches one regulation, reusing a cached copy if there is one.

    Args:
        celex: The CELEX id of the regulation, e.g. '32016R0679'.
        dest: Where to cache the downloaded page.

    Returns:
        The page's HTML.
    """
    if dest.exists():
        print(f"already have {celex}, using cached copy")
        return dest.read_text(encoding="utf-8")

    print(f"downloading {celex} from EUR-Lex...")
    request = urllib.request.Request(HTML_URL.format(celex=celex),
                                     headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        html = response.read().decode("utf-8")

    # Write to a temporary file first, so an interrupted download is never
    # mistaken for a complete one on the next run.
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".part")
    partial.write_text(html, encoding="utf-8")
    partial.rename(dest)
    return html


def build() -> list[dict]:
    """Downloads and parses every regulation and writes documents.jsonl.

    Returns:
        One dict per regulation with its id, title, URL and provisions.

    Raises:
        RuntimeError: A page contained no provisions, which means the EUR-Lex
            markup has changed.
    """
    documents = []
    for spec in config.DOCUMENTS:
        celex = spec["celex"]
        html = download(celex, config.RAW_DIR / f"{celex}.html")
        sections = parse_sections(html)
        if not sections:
            raise RuntimeError(
                f"No provisions found in {celex}; EUR-Lex markup may have "
                "changed.")
        print(f"{celex} ({spec['title']}): {len(sections)} provisions")
        documents.append({
            "doc_id": celex,
            "title": spec["title"],
            "url": HTML_URL.format(celex=celex),
            "sections": sections,
        })

    config.DOCUMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.DOCUMENTS_FILE, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"wrote {config.DOCUMENTS_FILE}")
    return documents


def load_documents() -> list[dict]:
    """Reads documents.jsonl as written by build().

    Raises:
        FileNotFoundError: build() has not been run yet.
    """
    if not config.DOCUMENTS_FILE.exists():
        raise FileNotFoundError(
            f"{config.DOCUMENTS_FILE} not found; run "
            "`python -m eulaw.corpus` first")
    with open(config.DOCUMENTS_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    build()
