import pytest

from thesis_rag.chunking import Chunk
from thesis_rag.generate import (
    NOT_FOUND,
    ThesisQA,
    build_messages,
    citation_tag,
    cited_passages,
    format_context,
    is_not_found,
    normalize_citations,
    source_dict,
)
from thesis_rag.retrieve import Passage, RetrievalResult

from .fakes import fake_llm


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (("4", "4.3", 62, 62, "text"), "[Ch.4 §4.3, p.62]"),
        (("4", "4.3", 62, 63, "text"), "[Ch.4 §4.3, pp.62–63]"),
        (("2", "Methods", 30, 30, "text"), "[Ch.2, p.30]"),
        (("", "Abstract", 3, 3, "text"), "[Abstract, p.3]"),
        (("3", "3.7", 55, 55, "table"), "[Ch.3 §3.7, p.55]"),
        (("1", "1.2", 16, 17, "summary"), "[Ch.1 §1.2 summary, pp.16–17]"),
        (("5", "", 80, 85, "summary"), "[Ch.5 summary, pp.80–85]"),
        (("", "Whole thesis", 16, 85, "summary"), "[Thesis summary, pp.16–85]"),
        (("1", "1.1", 0, 0, "text"), "[Ch.1 §1.1]"),
    ],
)
def test_citation_tag(args, expected):
    assert citation_tag(*args) == expected


def _passage(n: int, section: str, page: int, kind: str = "text") -> Passage:
    chunk = Chunk(
        id=f"c{n}",
        text=f"Body of chunk {n}. " * 40,
        header=f"Chapter 4: Results › §{section} Title {n}",
        chunk_type=kind,
        parent_id=f"p{n}",
        chapter="4",
        section=section,
        page_start=page,
        page_end=page,
    )
    return Passage(
        chunk=chunk,
        context=f"Context of passage {n}.",
        header=chunk.header,
        chapter="4",
        section=section,
        page_start=page,
        page_end=page,
        score=0.9 - n / 10,
    )


@pytest.fixture
def passages():
    return [_passage(1, "4.3", 62), _passage(2, "4.5", 66), _passage(3, "4.7", 70, "table")]


def test_format_context_numbers_and_tags(passages):
    ctx = format_context(passages)
    assert ctx.startswith(
        "Passage 1 (cite as [Ch.4 §4.3, p.62])\nChapter 4: Results › §4.3 Title 1"
    )
    assert "Passage 3 (cite as [Ch.4 §4.7, p.70])" in ctx
    assert ctx.count("---") == 2


def test_build_messages_has_rules_and_question(passages):
    system, user = build_messages("What changed?", passages)
    assert NOT_FOUND in system.content and "Never invent" in system.content
    assert user.content.endswith("Question: What changed?")


def test_cited_passages(passages):
    answer = "Yield fell [Ch.4 §4.3, p.62]. Also [Ch.4 §4.7, p.70] and [Ch.9 §9.9, p.1]."
    assert cited_passages(answer, passages) == [1, 3]


def test_is_not_found():
    assert is_not_found("Not found in the thesis.")
    assert is_not_found("  not found in the thesis")
    assert not is_not_found("The thesis found 12 SNPs.")


def test_source_dict_truncates_snippet(passages):
    src = source_dict(1, passages[0], snippet_chars=50)
    assert src["citation"] == "[Ch.4 §4.3, p.62]" and src["n"] == 1
    assert len(src["snippet"]) <= 51 and src["snippet"].endswith("…")


class StubRetriever:
    def __init__(self, passages):
        self.passages = passages

    def retrieve(self, question):
        return RetrievalResult(question, [question], False, self.passages, "rerank", {"total": 0.1})


def test_ask_returns_answer_with_sources(passages):
    llm = fake_llm(["Yield fell by 12% [Ch.4 §4.3, p.62]."])
    qa = ThesisQA(settings=None, index=object(), retriever=StubRetriever(passages), llm=llm)
    ans = qa.ask("How did yield change?")
    assert ans.answer == "Yield fell by 12% [Ch.4 §4.3, p.62]."
    assert ans.cited == [1] and not ans.not_found
    assert [s["n"] for s in ans.sources] == [1, 2, 3]


def test_ask_stream_yields_tokens_then_answer(passages):
    llm = fake_llm([NOT_FOUND])
    qa = ThesisQA(settings=None, index=object(), retriever=StubRetriever(passages), llm=llm)
    events = list(qa.ask_stream("Who funded it?"))
    tokens = [v for k, v in events if k == "token"]
    kind, answer = events[-1]
    assert kind == "done" and "".join(tokens) == NOT_FOUND
    assert answer.not_found and answer.cited == []


def test_no_passages_skips_llm():
    class ExplodingLLM:
        def invoke(self, *_):
            raise AssertionError("LLM must not be called")

    qa = ThesisQA(settings=None, index=object(), retriever=StubRetriever([]), llm=ExplodingLLM())
    assert qa.ask("anything").answer == NOT_FOUND


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Phytophthora [Passage 1, Ch.4 §4.3, p.62].", "Phytophthora [Ch.4 §4.3, p.62]."),
        ("QD is < 2.0. <claim> [Ch.4 §4.5, p.66].", "QD is < 2.0 [Ch.4 §4.5, p.66]."),
        ("Yield fell [Passage 3].", "Yield fell [Ch.4 §4.7, p.70]."),
        ("It is Low (cite as [Ch.4 §4.7, p.70]).", "It is Low [Ch.4 §4.7, p.70]."),
        ("See [Passage 9] and ref [4].", "See [Passage 9] and ref [4]."),
        ("Clean [Ch.4 §4.3, p.62].", "Clean [Ch.4 §4.3, p.62]."),
    ],
)
def test_normalize_citations(raw, expected, passages):
    assert normalize_citations(raw, passages) == expected


def test_answer_is_normalized_before_citations_are_counted(passages):
    llm = fake_llm(["Yield fell [Passage 2, Ch.4 §4.5, p.66]."])
    qa = ThesisQA(settings=None, index=object(), retriever=StubRetriever(passages), llm=llm)
    ans = qa.ask("How did yield change?")
    assert ans.answer == "Yield fell [Ch.4 §4.5, p.66]." and ans.cited == [2]
