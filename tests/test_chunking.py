from thesis_rag.chunking import TABLE, TEXT, chunk_document, chunk_table, count_tokens
from thesis_rag.ingest import Table


def _chunks(parsed, size=120, overlap=20, max_parent=400):
    return chunk_document(parsed, size, overlap, max_parent)


def test_contextual_header_is_prepended(parsed):
    chunks, _ = _chunks(parsed)
    c = next(c for c in chunks if c.section == "1.1")
    assert c.header == "Chapter 1: Introduction › §1.1 Drought Stress"
    assert c.embed_text.startswith(c.header + "\n\n")
    assert not c.text.startswith("Chapter 1")  # header is not duplicated into the body


def test_chunks_respect_size_and_overlap(parsed):
    chunks, _ = _chunks(parsed, size=120, overlap=20)
    long = [c for c in chunks if c.section == "2.2"]
    assert len(long) > 3
    assert all(count_tokens(c.text) <= 120 for c in long)
    for a, b in zip(long, long[1:], strict=False):
        tail = " ".join(a.text.split()[-5:])
        assert tail in b.text, "neighbouring chunks should overlap"


def test_chunks_never_cross_sections(parsed):
    chunks, _ = _chunks(parsed)
    for c in chunks:
        if c.section == "1.1":
            assert "1.2" not in c.text


def test_chunk_metadata(parsed):
    chunks, parents = _chunks(parsed, size=60, overlap=10)
    s11 = [c for c in chunks if c.section == "1.1"]
    assert s11[0].page_start == 4 and s11[-1].page_end == 5
    for c in chunks:
        assert c.chunk_type in (TEXT, TABLE)
        assert c.parent_id in parents
        assert c.id in parents[c.parent_id].child_ids
        assert c.page_start <= c.page_end


def test_small_subsections_grouped_into_one_parent(parsed):
    chunks, parents = _chunks(parsed, max_parent=400)
    c = next(c for c in chunks if c.section == "2.1.1")
    parent = parents[c.parent_id]
    assert parent.section == "2.1"
    assert "Tiny" in parent.text and "Small part one" in parent.text


def test_oversized_group_is_not_merged(parsed):
    chunks, parents = _chunks(parsed, max_parent=50)
    c = next(c for c in chunks if c.section == "2.1.1")
    assert parents[c.parent_id].section == "2.1.1"


def test_table_chunk_has_caption_and_section(parsed):
    chunks, parents = _chunks(parsed)
    t = next(c for c in chunks if c.chunk_type == TABLE)
    assert t.text.startswith("Table 1 Genotype traits")
    assert t.header.endswith("› Table 1 Genotype traits")
    assert parents[t.parent_id].chunk_type == TABLE


def test_long_table_split_repeats_header(parsed):
    rows = "\n".join(f"| G{i} | {'High ' * 10} |" for i in range(60))
    table = Table("t", "Table 9 Big", "| Genotype | Drought |\n|---|---|\n" + rows, 5, 5, 2)
    chunks, parent = chunk_table(table, parsed, chunk_size=150)
    assert len(chunks) > 1
    assert all("| Genotype | Drought |" in c.text for c in chunks)
    assert len(parent.child_ids) == len(chunks)
