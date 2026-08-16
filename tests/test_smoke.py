"""Smoke tests - prove the package imports and the configuration is coherent."""

import json
import pathlib

from eulaw import config

REPO_ROOT = pathlib.Path(__file__).parent.parent


def test_config_is_sane():
    assert config.CHUNK_SIZE > config.CHUNK_OVERLAP >= 0
    assert config.TOP_K > 0
    assert config.EMBEDDING_MODEL
    assert config.MAX_CHUNKS_PER_PROVISION >= 1
    assert 0.0 <= config.MIN_PRIMARY <= 1.0
    assert config.PRIMARY_KINDS


def test_package_imports():
    import eulaw

    assert eulaw.__version__


def test_corpus_documents_are_well_formed():
    assert config.DOCUMENTS, "the corpus cannot be empty"
    for doc in config.DOCUMENTS:
        assert doc["celex"].startswith("3"), "CELEX ids for regulations start with 3"
        assert doc["title"] and doc["full_title"]
    celex = [d["celex"] for d in config.DOCUMENTS]
    assert len(celex) == len(set(celex)), "a regulation must not be indexed twice"


def test_ui_copy_is_present():
    # The app renders these directly; empty values would ship a blank page.
    assert config.TITLE and config.DESCRIPTION and config.DISCLAIMER
    assert config.EXAMPLES
    assert any("capital of Austria" in q for q in config.EXAMPLES), \
        "one example should demonstrate the refusal contract"


def test_question_set_has_both_answerable_questions_and_traps():
    questions = REPO_ROOT / "eval" / "questions.jsonl"
    rows = [json.loads(line) for line in
            questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(q["answerable"] for q in rows)
    assert any(not q["answerable"] for q in rows), "traps test the refusal contract"

    known = {d["celex"] for d in config.DOCUMENTS}
    for q in rows:
        if q["answerable"]:
            # A gold answer pointing at a regulation outside the corpus would
            # make the hit-rate unachievable rather than merely hard.
            assert set(q["source_ids"]) <= known, f"{q['id']} cites an unindexed doc"
            assert q["source_labels"], f"{q['id']} needs a gold provision"

