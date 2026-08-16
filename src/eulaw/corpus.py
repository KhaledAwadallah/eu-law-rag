"""Download the regulations from EUR-Lex and split them into provisions.

EUR-Lex serves each regulation as XHTML where every article, recital and annex
sits in a div whose id is also its page anchor (art_6, rct_47, anx_III) - which
is what lets a citation deep-link to the exact provision.

    python -m eulaw.corpus
"""

import json
import pathlib
import re
import urllib.request

from bs4 import BeautifulSoup

from eulaw import config

HTML_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}"
USER_AGENT = "eu-law-rag/0.1 (research project; contact via GitHub KhaledAwadallah/eu-law-rag)"

# Chapters and sections are skipped: they only contain articles, so indexing
# them would return the same text again under a less precise anchor.
SECTION_ID = re.compile(r"^(art_\d+|rct_\d+|anx_[IVXLC]+)$")

# Shorter than this and it is a heading or a cross-reference stub.
MIN_SECTION_CHARS = 120

PROVISION_NAMES = {"art": "Article", "rct": "Recital", "anx": "Annex"}


def anchor_label(anchor: str) -> str:
    """art_6 -> 'Article 6', rct_47 -> 'Recital 47', anx_III -> 'Annex III'."""
    kind, _, number = anchor.partition("_")
    return f"{PROVISION_NAMES[kind]} {number}"


def provision_kind(label: str) -> str:
    """'Article', 'Recital' or 'Annex'."""
    return label.split()[0] if label else ""


def _normalize(text: str) -> str:
    r"""Collapse whitespace. `\s` also covers EUR-Lex's non-breaking spaces."""
    return re.sub(r"\s+", " ", text).strip()


def parse_sections(html: str) -> list[dict]:
    """Every article, recital and annex in one regulation."""
    soup = BeautifulSoup(html, "html.parser")

    sections = []
    for div in soup.find_all("div", id=SECTION_ID):
        # get_text walks the nested title divs, paragraphs and the tables
        # holding lettered points, so a provision comes back as one string.
        text = _normalize(div.get_text(" "))
        if len(text) < MIN_SECTION_CHARS:
            continue
        subtitle = div.find("p", class_="oj-sti-art")   # e.g. "Human oversight"
        sections.append({
            "anchor": div["id"],
            "label": anchor_label(div["id"]),
            "subtitle": _normalize(subtitle.get_text(" ")) if subtitle else "",
            "text": text,
        })
    return sections


def download(celex: str, dest: pathlib.Path) -> str:
    """Fetch one regulation, cached so re-runs do not re-download."""
    if dest.exists():
        print(f"already have {celex}, using cached copy")
        return dest.read_text(encoding="utf-8")

    print(f"downloading {celex} from EUR-Lex...")
    req = urllib.request.Request(HTML_URL.format(celex=celex),
                                 headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        html = resp.read().decode("utf-8")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")          # write-then-rename, so a crash
    tmp.write_text(html, encoding="utf-8")   # cannot leave a partial file that
    tmp.rename(dest)                         # a re-run mistakes for complete
    return html


def build() -> list[dict]:
    """Download and parse every regulation, writing documents.jsonl."""
    documents = []
    for spec in config.DOCUMENTS:
        celex = spec["celex"]
        sections = parse_sections(download(celex, config.RAW_DIR / f"{celex}.html"))
        if not sections:
            raise RuntimeError(
                f"{celex}: no provisions found - EUR-Lex markup may have changed"
            )
        counts: dict[str, int] = {}
        for s in sections:
            kind = provision_kind(s["label"])
            counts[kind] = counts.get(kind, 0) + 1
        print(f"{celex} ({spec['title']}): {len(sections)} provisions "
              f"({', '.join(f'{v} {k}' for k, v in sorted(counts.items()))})")

        documents.append({
            "doc_id": celex,
            "title": spec["title"],
            "full_title": spec.get("full_title", spec["title"]),
            "url": HTML_URL.format(celex=celex),
            "sections": sections,
        })

    out = config.DOCUMENTS_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    provisions = sum(len(d["sections"]) for d in documents)
    chars = sum(len(s["text"]) for d in documents for s in d["sections"])
    print(f"done: {len(documents)} regulations, {provisions} provisions, "
          f"{chars:,} characters -> {out}")
    return documents


def load_documents() -> list[dict]:
    if not config.DOCUMENTS_FILE.exists():
        raise FileNotFoundError(
            f"{config.DOCUMENTS_FILE} not found - run `python -m eulaw.corpus` first"
        )
    with open(config.DOCUMENTS_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    build()
