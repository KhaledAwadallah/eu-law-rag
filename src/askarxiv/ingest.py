""" Download arXiv papers and extract text from their PDFs.

Provides:
    download(query, n, outdir) -> list[dict]   # saves PDFs, returns + saves metadata
    pdf_to_text(path) -> str                   # PDF -> plain text, references removed
    extract_all(pdf_dir, txt_dir) -> None      # runs pdf_to_text over every PDF

Command line:
    python -m askarxiv.ingest
"""

import json
import pathlib
import re
import time
import urllib.request

import arxiv
import pymupdf

from askarxiv import config

USER_AGENT = "askarxiv/0.1"


def _fetch_pdf(url: str, dest: pathlib.Path) -> None:
    """Download url to dest safely: write to a temp file, rename when complete.

    The rename-at-the-end pattern means a crash mid-download can never leave a
    truncated file that a later re-run would mistake for a finished one.
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
        f.write(resp.read())
    tmp.rename(dest)


def download(query: str = config.ARXIV_QUERY,
             n: int = config.N_PAPERS,
             outdir: str = config.PDF_DIR) -> list[dict]:
    """Download the n most recent papers matching an arXiv query.

    Skips PDFs that already exist, so the function is safe to re-run.
    Writes metadata (title, authors, year, abstract, url) to METADATA_FILE.
    """
    out = pathlib.Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    client = arxiv.Client()  # respects arXiv rate limits (3 s between requests)
    search = arxiv.Search(
        query=query,
        max_results=n,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    meta = []
    for result in client.results(search):
        paper_id = result.get_short_id()             # e.g. "2607.01234v1"
        if result.pdf_url is None:
            print(f"no PDF available for {paper_id}, skipping paper")
            continue
        filename = paper_id.replace("/", "_") + ".pdf"
        pdf_path = out / filename
        if not pdf_path.exists():
            print(f"downloading {paper_id}: {result.title[:60]}...")
            _fetch_pdf(result.pdf_url, pdf_path)
            time.sleep(1)
        else:
            print(f"already have {paper_id}, skipping download")
        meta.append({
            "id": paper_id,
            "file": filename,
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "year": result.published.year,
            "abstract": result.summary,
            "url": result.entry_id,
        })

    meta_path = pathlib.Path(config.METADATA_FILE)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved metadata for {len(meta)} papers -> {meta_path}")
    return meta


def strip_references(text: str) -> str:
    """Cut everything from the LAST 'References'/'Bibliography' heading onward.

    Reference lists are noise for retrieval: they are dense with author names
    and paper titles, so they match many queries strongly while containing no
    explanations. Using the last match avoids cutting at a mention of the word
    'references' in the body text.
    """
    matches = list(re.finditer(r"^\s*(references|bibliography)\s*$",
                               text, flags=re.IGNORECASE | re.MULTILINE))
    if matches:
        return text[: matches[-1].start()]
    return text


def pdf_to_text(path: str | pathlib.Path) -> str:
    """Extract plain text from one PDF (all pages), references removed."""
    doc = pymupdf.open(path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return strip_references(text)


def extract_all(pdf_dir: str = config.PDF_DIR, txt_dir: str = "data/text") -> None:
    """Convert every downloaded PDF to a .txt file for the chunking step."""
    out = pathlib.Path(txt_dir)
    out.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(pathlib.Path(pdf_dir).glob("*.pdf"))
    for pdf in pdfs:
        txt_path = out / (pdf.stem + ".txt")
        try:
            text = pdf_to_text(pdf)
        except Exception as e:  # a single corrupt PDF must not kill the run
            print(f"FAILED to parse {pdf.name}: {e}")
            continue
        txt_path.write_text(text, encoding="utf-8")
        print(f"parsed {pdf.name}: {len(text):,} characters")
    print(f"done: {len(list(out.glob('*.txt')))} of {len(pdfs)} PDFs parsed -> {out}")


if __name__ == "__main__":
    download()
    extract_all()
