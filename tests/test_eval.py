"""Tests for the evaluation's scoring helpers and the question set."""

import collections

from eulaw import config
from eval import run_eval

_AI_ACT = "32024R1689"
_GDPR = "32016R0679"


def test_provision_hit_needs_the_right_regulation():
    # Both regulations have an Article 99; only the AI Act's one counts here.
    question = {"source_ids": [_AI_ACT], "source_labels": ["Article 99"]}
    assert run_eval.provision_hit(
        question, [{"doc_id": _AI_ACT, "label": "Article 99"}])
    assert not run_eval.provision_hit(
        question, [{"doc_id": _GDPR, "label": "Article 99"}])


def test_parse_verdict():
    assert run_eval.parse_verdict("SUPPORTED") == ("SUPPORTED", 1.0)
    assert run_eval.parse_verdict(" unsupported.") == ("UNSUPPORTED", 0.0)
    assert run_eval.parse_verdict("PARTIAL") == ("PARTIAL", 0.5)
    assert run_eval.parse_verdict("maybe") == ("UNPARSEABLE", 0.0)


def test_misattributed_matches_whole_names_only():
    assert run_eval.misattributed(
        "The Cyber Resilience Act requires ...") == ["Cyber Resilience Act"]
    assert run_eval.misattributed("Under the EU Data Act ...") == ["Data Act"]
    assert not run_eval.misattributed("records of data activities [1]")


def test_question_set_is_well_formed():
    questions = run_eval.load_questions()
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))

    known = {d["celex"] for d in config.DOCUMENTS}
    for q in questions:
        if q["answerable"]:
            # A gold provision outside the corpus could never be retrieved.
            assert set(q["source_ids"]) <= known, q["id"]
            assert q["source_labels"], q["id"]
        else:
            assert not q["source_labels"], q["id"]

    counts = collections.Counter(q["answerable"] for q in questions)
    assert counts[True] and counts[False]  # Traps test the refusals.
