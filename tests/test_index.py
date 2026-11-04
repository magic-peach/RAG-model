import pytest

from thesis_rag import index as index_mod
from thesis_rag.config import Settings
from thesis_rag.index import Index, bm25_tokenize, build_index, fingerprint

from .fakes import FakeEmbedder


@pytest.fixture
def built(tmp_path, monkeypatch, parsed):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(index_mod, "parse_pdf", lambda path: parsed)
    monkeypatch.setattr(index_mod, "Embedder", lambda settings: FakeEmbedder())
    settings = Settings(
        data_dir=tmp_path / "data", summarize_sections=False, chunk_size=120, chunk_overlap=20
    )
    return pdf, settings


def test_fingerprint_tracks_settings():
    s = Settings()
    assert fingerprint("abc", s) == fingerprint("abc", Settings())
    assert fingerprint("abc", s) != fingerprint("abd", s)
    assert fingerprint("abc", s) != fingerprint("abc", Settings(chunk_size=400, chunk_overlap=40))


def test_build_is_idempotent(built):
    pdf, settings = built
    first = build_index(pdf, settings)
    second = build_index(pdf, settings)
    assert first["cached"] is False and second["cached"] is True
    assert first["fingerprint"] == second["fingerprint"]
    assert build_index(pdf, settings, force=True)["cached"] is False


def test_index_search(built):
    pdf, settings = built
    build_index(pdf, settings)
    idx = Index(settings, embedder=FakeEmbedder())
    bm = idx.bm25_search("variant calling GATK", k=3)
    assert bm and idx.chunks[bm[0].chunk_id].section == "2.2"
    dense = idx.dense_search("variant calling GATK", k=3)
    assert len(dense) == 3
    tables = idx.bm25_search("genotype drought", k=5, types=["table"])
    assert tables and all(idx.chunks[h.chunk_id].chunk_type == "table" for h in tables)


def test_bm25_tokenize_keeps_identifiers():
    assert bm25_tokenize("The gene Mi00022 and §3.10 in GATK") == [
        "gene",
        "mi00022",
        "3.10",
        "gatk",
    ]


def test_with_retry_recovers_then_gives_up(monkeypatch):
    monkeypatch.setattr(index_mod.time, "sleep", lambda s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("llama-server did not start")
        return "ok"

    assert index_mod.with_retry(flaky) == "ok" and len(calls) == 3

    def broken():
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        index_mod.with_retry(broken, attempts=2)


def test_collection_is_opened_once_under_concurrency(built, monkeypatch):
    import threading
    import time

    pdf, settings = built
    build_index(pdf, settings)
    idx = Index(settings, embedder=FakeEmbedder())
    opened = []

    class SlowClient:
        def get_collection(self, name):
            opened.append(name)
            time.sleep(0.05)  # widen the race window
            return object()

    monkeypatch.setattr(index_mod, "_chroma_client", lambda path: SlowClient())
    threads = [threading.Thread(target=lambda: idx.collection) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(opened) == 1
