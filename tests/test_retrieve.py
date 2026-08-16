"""Unit tests for chunk selection - the diversity cap and the reserved slots.

`select` is deliberately separate from the vector search, so these run with no
index, no embedding model and no network.
"""

from eulaw.retrieve import cite, is_primary, select


def _candidate(cid, label, score, anchor=None):
    """One scored candidate, in the shape `select` expects."""
    return {"_id": cid, "label": label, "score": score,
            "anchor": anchor or label.lower().replace(" ", "_")}


def test_selection_returns_score_order():
    candidates = [_candidate("r1", "Recital 40", 0.90),
                  _candidate("a1", "Article 6", 0.80),
                  _candidate("r2", "Recital 50", 0.85)]
    assert [h["score"] for h in select(candidates, k=3)] == [0.90, 0.85, 0.80]


def test_cap_stops_one_long_article_dominating():
    # Article 5 of the AI Act is 11k characters: without a cap its chunks
    # would fill the whole top-k and the model would see one provision.
    candidates = [_candidate(f"a{i}", "Article 5", 0.9 - i / 100) for i in range(5)]
    candidates += [_candidate("b0", "Article 6", 0.5)]
    hits = select(candidates, k=3)
    assert [h["label"] for h in hits] == ["Article 5", "Article 5", "Article 6"]


def test_reserved_slots_pull_articles_past_higher_scoring_recitals():
    # Five recitals outrank every article, as happens on plain-language
    # questions; MIN_PRIMARY=0.6 must still reserve 3 of 5 slots for articles.
    candidates = [_candidate(f"r{i}", f"Recital {i}", 0.90 - i / 100) for i in range(5)]
    candidates += [_candidate(f"a{i}", f"Article {i}", 0.70 - i / 100) for i in range(4)]
    hits = select(candidates, k=5)
    kinds = [h["label"].split()[0] for h in hits]
    assert kinds.count("Article") == 3
    assert kinds.count("Recital") == 2
    assert len(hits) == 5


def test_annexes_count_as_binding_alongside_articles():
    candidates = [_candidate(f"r{i}", f"Recital {i}", 0.9 - i / 100) for i in range(5)]
    candidates += [_candidate("x1", "Annex III", 0.60)]
    assert any(h["label"] == "Annex III" for h in select(candidates, k=5))


def test_reserved_slots_do_not_invent_articles_that_are_not_there():
    """With no binding provision in reach, the top-k is still filled."""
    candidates = [_candidate(f"r{i}", f"Recital {i}", 0.9 - i / 100) for i in range(6)]
    hits = select(candidates, k=5)
    assert len(hits) == 5
    assert all(h["label"].startswith("Recital") for h in hits)


def test_cap_can_leave_fewer_than_k_and_never_duplicates():
    # Two provisions, cap of 2 each: k=5 cannot be filled. The caller warns;
    # what matters here is that nothing is returned twice to pad the list.
    candidates = [_candidate(f"a{i}", "Article 6", 0.9 - i / 100) for i in range(3)]
    candidates += [_candidate(f"b{i}", "Article 7", 0.5 - i / 100) for i in range(3)]
    hits = select(candidates, k=5)
    assert len(hits) == 4
    assert len({h["_id"] for h in hits}) == 4


def test_higher_cap_admits_more_from_the_same_provision():
    candidates = [_candidate(f"a{i}", "Article 5", 0.9 - i / 100) for i in range(5)]
    assert len(select(candidates, k=4, max_per_provision=4)) == 4


def test_diversity_is_per_provision_not_per_regulation():
    # Same regulation, different articles: all four are allowed through,
    # because the unit that must not dominate is the provision.
    candidates = [_candidate(f"a{i}", f"Article {i}", 0.9 - i / 100) for i in range(4)]
    assert len(select(candidates, k=4)) == 4


def test_is_primary_reads_the_provision_kind():
    assert is_primary({"label": "Article 6"})
    assert is_primary({"label": "Annex III"})
    assert not is_primary({"label": "Recital 40"})
    assert not is_primary({"label": ""})


def test_cite_names_the_regulation_and_provision():
    assert cite({"title": "GDPR", "label": "Article 6"}) == "GDPR, Article 6"
    assert cite({"title": "AI Act", "label": ""}) == "AI Act"
