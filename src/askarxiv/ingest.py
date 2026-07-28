"""Step 2: download arXiv papers and extract text from their PDFs.

Will provide:
    download(query, n, outdir) -> list[dict]   # saves PDFs, returns metadata
    pdf_to_text(path) -> str                   # PDF -> plain text (PyMuPDF)
"""
