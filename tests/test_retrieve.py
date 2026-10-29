"""Tests for chunk selection: the per-provision cap and the reserved slots.

select() is separate from the vector search, so these need no index, no
embedding model and no network.
"""

from eulaw import retrieve


def _candidate(chunk_id, label, score, anchor=None):
    return {"id": chunk_id, "label": label, "score": score,
            "anchor": anchor or label.lower().replace(" ", "_")}


def test_select_returns_score_order():
    candidates = [_candidate("r1", "Recital 40", 0.90),
                  _candidate("a1", "Article 6", 0.80),
                  _candidate("r2", "Recital 50", 0.85)]
    hits = retrieve.select(candidates, k=3)
    assert [h["score"] for h in hits] == [0.90, 0.85, 0.80]


def test_select_caps_chunks_per_provision():
    # Without the cap, one long article would fill the whole top-k.
    candidates = [_candidate(f"a{i}", "Article 5", 0.9 - i / 100)
                  for i in range(5)]
    candidates.append(_candidate("b0", "Article 6", 0.5))
    hits = retrieve.select(candidates, k=3)
    assert [h["label"] for h in hits] == ["Article 5", "Article 5",
                                          "Article 6"]


def test_select_reserves_slots_for_articles():
    # Five recitals outrank every article, as on plain-language questions;
    # min_primary=0.6 still reserves 3 of 5 slots for articles.
    candidates = [_candidate(f"r{i}", f"Recital {i}", 0.90 - i / 100)
                  for i in range(5)]
    candidates += [_candidate(f"a{i}", f"Article {i}", 0.70 - i / 100)
                   for i in range(4)]
    kinds = [h["label"].split()[0]
             for h in retrieve.select(candidates, k=5, min_primary=0.6)]
    assert kinds.count("Article") == 3
    assert kinds.count("Recital") == 2


def test_select_counts_annexes_as_binding():
    candidates = [_candidate(f"r{i}", f"Recital {i}", 0.9 - i / 100)
                  for i in range(5)]
    candidates.append(_candidate("x1", "Annex III", 0.60))
    hits = retrieve.select(candidates, k=5)
    assert any(h["label"] == "Annex III" for h in hits)


def test_select_fills_with_recitals_when_no_articles_exist():
    candidates = [_candidate(f"r{i}", f"Recital {i}", 0.9 - i / 100)
                  for i in range(6)]
    hits = retrieve.select(candidates, k=5)
    assert len(hits) == 5
    assert all(h["label"].startswith("Recital") for h in hits)


def test_select_returns_fewer_than_k_rather_than_duplicates():
    # Two provisions with a cap of 2 each cannot fill k=5.
    candidates = [_candidate(f"a{i}", "Article 6", 0.9 - i / 100)
                  for i in range(3)]
    candidates += [_candidate(f"b{i}", "Article 7", 0.5 - i / 100)
                   for i in range(3)]
    hits = retrieve.select(candidates, k=5)
    assert len(hits) == 4
    assert len({h["id"] for h in hits}) == 4


def test_select_without_rules_is_plain_top_k():
    candidates = [_candidate(f"r{i}", f"Recital {i}", 0.9 - i / 100)
                  for i in range(3)]
    candidates += [_candidate(f"a{i}", "Article 5", 0.8 - i / 100)
                   for i in range(3)]
    hits = retrieve.select(candidates, k=4, max_per_provision=4,
                           min_primary=0.0)
    assert [h["id"] for h in hits] == ["r0", "r1", "r2", "a0"]


def test_is_primary_reads_the_provision_kind():
    assert retrieve.is_primary({"label": "Article 6"})
    assert retrieve.is_primary({"label": "Annex III"})
    assert not retrieve.is_primary({"label": "Recital 40"})
    assert not retrieve.is_primary({"label": ""})


def test_cite_names_regulation_and_provision():
    assert retrieve.cite({"title": "GDPR", "label": "Article 6"}) == (
        "GDPR, Article 6")
    assert retrieve.cite({"title": "AI Act", "label": ""}) == "AI Act"
