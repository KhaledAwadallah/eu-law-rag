"""Unit tests for the EUR-Lex parser (no network involved).

The fixture mirrors the real markup: provisions live in nested divs whose id
is the page anchor, article headings sit in an `oj-sti-art` paragraph, and
text is broken across table cells for lettered points.
"""

from eulaw.corpus import (
    MIN_SECTION_CHARS,
    anchor_label,
    parse_sections,
    provision_kind,
)

HTML = """<html><body>
<div class="eli-container" id="cpt_II">
  <div class="eli-subdivision" id="art_6">
    <p class="oj-ti-art">Article&nbsp;6</p>
    <div class="eli-title" id="art_6.tit_1">
      <p class="oj-sti-art">Lawfulness of processing</p>
    </div>
    <div id="006.001">
      <p class="oj-normal">1.&nbsp;&nbsp;&nbsp;Processing shall be lawful only if
      at least one of the following applies:</p>
      <table><tbody><tr>
        <td><p class="oj-normal">(a)</p></td>
        <td><p class="oj-normal">the data subject has given consent;</p></td>
      </tr></tbody></table>
    </div>
  </div>
  <div class="eli-subdivision" id="rct_47">
    <p class="oj-normal">(47) The legitimate interests of a controller may
    provide a legal basis for processing, provided the interests of the data
    subject do not override them.</p>
  </div>
  <div class="eli-subdivision" id="art_99">
    <p class="oj-ti-art">Article&nbsp;99</p>
    <p class="oj-normal">Too short.</p>
  </div>
</div>
<div class="eli-container" id="anx_III">
  <p class="oj-doc-ti">ANNEX&nbsp;III</p>
  <p class="oj-normal">High-risk AI systems pursuant to Article 6(2) are those
  listed in the following areas: biometrics, critical infrastructure.</p>
</div>
</body></html>"""


def _by_anchor(sections):
    return {s["anchor"]: s for s in sections}


def test_anchor_label_names_the_provision():
    assert anchor_label("art_6") == "Article 6"
    assert anchor_label("rct_47") == "Recital 47"
    assert anchor_label("anx_III") == "Annex III"


def test_provision_kind_reads_the_first_word():
    assert provision_kind("Article 6") == "Article"
    assert provision_kind("Annex III") == "Annex"
    assert provision_kind("") == ""


def test_parses_articles_recitals_and_annexes():
    sections = _by_anchor(parse_sections(HTML))
    assert set(sections) == {"art_6", "rct_47", "anx_III"}   # art_99 too short
    assert sections["art_6"]["label"] == "Article 6"
    assert sections["anx_III"]["label"] == "Annex III"


def test_article_subtitle_is_captured_separately():
    assert _by_anchor(parse_sections(HTML))["art_6"]["subtitle"] == \
        "Lawfulness of processing"
    assert _by_anchor(parse_sections(HTML))["rct_47"]["subtitle"] == ""


def test_nested_divs_and_table_cells_all_land_in_one_section():
    # The article's text spans a title div, a numbered paragraph and a table;
    # stopping at the first </div> would truncate it after the heading.
    text = _by_anchor(parse_sections(HTML))["art_6"]["text"]
    assert "Processing shall be lawful" in text
    assert "(a)" in text and "given consent" in text
    assert "Article 6" in text


def test_whitespace_and_non_breaking_spaces_are_normalized():
    text = _by_anchor(parse_sections(HTML))["art_6"]["text"]
    assert "\n" not in text
    assert " " not in text
    assert "  " not in text
    assert text.startswith("Article 6 Lawfulness of processing")


def test_stub_provisions_are_dropped():
    # Cross-reference stubs carry no retrievable substance.
    assert "art_99" not in _by_anchor(parse_sections(HTML))
    assert MIN_SECTION_CHARS > 0


def test_structural_containers_are_not_indexed_as_provisions():
    # Chapters only contain articles; indexing them too would return the same
    # text twice under a less precise anchor.
    assert "cpt_II" not in _by_anchor(parse_sections(HTML))


def test_no_provisions_gives_empty_result_not_an_exception():
    assert parse_sections("<html><body><p>nothing here</p></body></html>") == []
