import pytest

from thesis_rag import index as index_mod
from thesis_rag.chunking import SUMMARY, make_summary_chunk
from thesis_rag.config import Settings
from thesis_rag.generate import expand_query
from thesis_rag.index import Index, build_index
from thesis_rag.retrieve import THESIS_SUMMARY_ID, Retriever, is_broad, rrf_fuse

from .fakes import FakeEmbedder, FakeReranker, fake_llm


def test_rrf_scores_and_order():
    fused = rrf_fuse([["a", "b", "c"], ["b", "c", "d"]], k=60)
    scores = dict(fused)
    # b is rank 2 and rank 1: 1/62 + 1/61
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["d"] == pytest.approx(1 / 63)
    assert [x for x, _ in fused] == ["b", "c", "a", "d"]


def test_rrf_single_list_keeps_order():
    assert [x for x, _ in rrf_fuse([["z", "y", "x"]])] == ["z", "y", "x"]


def test_rrf_agreement_beats_single_top_rank():
    # Found 2nd by both retrievers outranks found 1st by only one.
    fused = rrf_fuse([["solo", "both"], ["other", "both"]])
    assert fused[0][0] == "both"


@pytest.mark.parametrize(
    "question",
    [
        "What is the main idea of the thesis?",
        "Summarize the thesis",
        "What is the overall contribution?",
        "What are the key findings?",
        "What is this thesis about?",
        "what is this about",
        "What's it all about?",
        "Tell me about this document",
    ],
)
def test_broad_questions(question):
    assert is_broad(question)


@pytest.mark.parametrize(
    "question",
    [
        "Which tool was used for variant calling?",
        "What alignment percentage did Punjab Local show?",
        "How many SNPs were found in chromosome 3?",
    ],
)
def test_specific_questions(question):
    assert not is_broad(question)


def test_expand_query_parses_list_output():
    llm = fake_llm(
        [
            "Here are 3 rephrasings:\n1. Which GATK step finds variants?\n"
            "- How were SNPs called?\n\n3) Which tool was used for variant calling?\n"
            '"What software detected variants?"'
        ]
    )
    variants = expand_query(llm, "Which tool was used for variant calling?", n=3)
    assert variants == [
        "Which GATK step finds variants?",
        "How were SNPs called?",
        "What software detected variants?",
    ]


@pytest.fixture
def index(tmp_path, monkeypatch, parsed):
    """Synthetic index with one summary chunk injected (no LLM at build time)."""
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(index_mod, "parse_pdf", lambda path: parsed)
    monkeypatch.setattr(index_mod, "Embedder", lambda settings: FakeEmbedder())

    def fake_summaries(doc, parents, settings, progress=None):
        chunk, parent = make_summary_chunk(
            id=THESIS_SUMMARY_ID,
            text="Overall this thesis studies drought tolerance and SNP markers in mulberry.",
            header="Front Matter › Whole-thesis summary",
            chapter="",
            section="Whole thesis",
            page_start=4,
            page_end=6,
        )
        return [chunk], {parent.id: parent}

    monkeypatch.setattr(index_mod, "build_summaries", fake_summaries)
    settings = Settings(data_dir=tmp_path / "data", chunk_size=120, chunk_overlap=20)
    build_index(pdf, settings)
    return settings


def _retriever(settings, **overrides):
    s = settings.model_copy(update=overrides)
    return Retriever(s, Index(s, embedder=FakeEmbedder()), reranker=FakeReranker())


def test_specific_question_excludes_summaries(index):
    r = _retriever(index).retrieve("Which tool is used for variant calling with GATK?")
    assert not r.broad and r.score_kind == "rerank"
    assert r.passages and all(p.chunk_type != SUMMARY for p in r.passages)
    assert r.passages[0].section == "2.2"


def test_broad_question_pins_thesis_summary(index):
    r = _retriever(index).retrieve("What is the main contribution of this thesis?")
    assert r.broad
    assert r.passages[0].chunk.id == THESIS_SUMMARY_ID


def test_one_passage_per_parent(index):
    # With a budget large enough for §2.2, its many matching chunks collapse
    # into a single passage carrying the whole section.
    r = _retriever(index, top_k=5, max_passage_tokens=10_000).retrieve("variant calling GATK")
    parents = [p.chunk.parent_id for p in r.passages]
    assert len(parents) == len(set(parents))
    assert len({p.context for p in r.passages}) == len(r.passages)


def test_long_parent_falls_back_to_chunk(index):
    # §2.2 is ~3000 tokens: too long to send whole, so the matched chunk is sent.
    r = _retriever(index, max_passage_tokens=500).retrieve("variant calling GATK")
    top = r.passages[0]
    assert top.context == top.chunk.text


def test_stages_can_be_switched_off(index):
    reranker = FakeReranker()
    s = index.model_copy(update={"use_bm25": False, "use_reranker": False})
    r = Retriever(s, Index(s, embedder=FakeEmbedder()), reranker=reranker).retrieve("drought")
    assert r.score_kind == "rrf" and reranker.calls == 0 and r.passages


def test_multi_query_uses_rephrasings(index):
    s = index.model_copy(update={"use_multi_query": True})
    llm = fake_llm(["How is leaf yield affected?\nWhat reduces mulberry leaves?"])
    r = Retriever(s, Index(s, embedder=FakeEmbedder()), reranker=FakeReranker(), llm=llm)
    result = r.retrieve("What lowers leaf yield?")
    assert result.queries[0] == "What lowers leaf yield?" and len(result.queries) == 3


def test_failed_expansion_falls_back(index):
    class Broken:
        def invoke(self, *_):
            raise ConnectionError("ollama down")

    s = index.model_copy(update={"use_multi_query": True})
    r = Retriever(s, Index(s, embedder=FakeEmbedder()), reranker=FakeReranker(), llm=Broken())
    assert r.retrieve("leaf yield").queries == ["leaf yield"]


def test_allow_summaries_override(index):
    r = _retriever(index)
    assert not r.retrieve("What are the key findings?", allow_summaries=False).broad
    forced = r.retrieve("variant calling GATK", allow_summaries=True)
    assert forced.broad and forced.passages[0].chunk.id == THESIS_SUMMARY_ID
