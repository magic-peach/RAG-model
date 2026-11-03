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
