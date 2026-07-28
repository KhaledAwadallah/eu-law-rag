"""Smoke tests — prove the package imports and the config is sane.

Real tests for chunking/retrieval arrive with their modules in Steps 3-4.
"""

from askarxiv import config


def test_config_is_sane():
    assert config.CHUNK_SIZE > config.CHUNK_OVERLAP >= 0
    assert config.TOP_K > 0
    assert config.EMBEDDING_MODEL


def test_package_imports():
    import askarxiv

    assert askarxiv.__version__
