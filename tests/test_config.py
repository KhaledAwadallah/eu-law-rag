"""Sanity checks on the settings."""

from eulaw import config


def test_settings_are_consistent():
    assert config.CHUNK_SIZE > config.CHUNK_OVERLAP >= 0
    assert config.TOP_K > 0
    assert config.MAX_CHUNKS_PER_PROVISION >= 1
    assert 0.0 <= config.MIN_PRIMARY <= 1.0


def test_each_regulation_is_indexed_once():
    celex = [d["celex"] for d in config.DOCUMENTS]
    assert celex
    assert len(celex) == len(set(celex))
    # CELEX ids of regulations start with 3 (sector) and contain an R.
    assert all(c.startswith("3") and "R" in c for c in celex)
