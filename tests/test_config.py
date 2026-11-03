import pytest

from thesis_rag.config import Settings


def test_env_overrides_yaml(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "3")
    assert Settings().top_k == 3


def test_kwargs_override_env(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "3")
    assert Settings(top_k=4).top_k == 4


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        Settings(chunk_size=100, chunk_overlap=100)


def test_index_dir_depends_on_chunking():
    a = Settings(chunk_size=600, chunk_overlap=72).index_dir
    b = Settings(chunk_size=400, chunk_overlap=48).index_dir
    c = Settings(structured_chunking=False).index_dir
    assert len({a, b, c}) == 3
