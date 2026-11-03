"""LLM calls: hierarchical summaries at ingest time, grounded answers at query time.

All generation goes through :func:`get_llm`, a ``ChatOllama`` client for the
local Llama 3.1 model. Low temperature (0.1) because we want faithful
extraction, not creativity; ``num_ctx`` is raised from Ollama's small default
so several retrieved sections fit in one prompt.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from .config import Settings

if TYPE_CHECKING:
    from .index import Index
    from .retrieve import Passage, RetrievalResult, Retriever

log = logging.getLogger(__name__)


def get_llm(settings: Settings, **overrides) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    params = {
        "model": settings.llm_model,
        "base_url": settings.ollama_base_url,
        "temperature": settings.temperature,
        "num_ctx": settings.num_ctx,
    }
    params.update(overrides)
    return ChatOllama(**params)


def _complete(llm: BaseChatModel, system: str, user: str) -> str:
    msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return str(msg.content).strip()


# ---------------------------------------------------------------------------
# Hierarchical summaries (ingest time)
# ---------------------------------------------------------------------------
#
# Broad questions ("what is the main contribution?") have no single passage
# that answers them; similarity search returns a random-looking set of
# fragments. Summaries give those questions something to retrieve. They are
# built bottom-up (section -> chapter -> thesis) so each prompt stays small:
# 80 sections would not fit in one 8k context, but 5 chapter summaries do.

SUMMARY_PROMPT_VERSION = "v1"  # bump to invalidate cached summaries

SUMMARY_SYSTEM = (
    "You summarise sections of a PhD thesis for a search index. Write plain prose, "
    "no bullet points, no preamble such as 'This section'. Keep every specific name, "
    "method, tool, genotype, trait and number exactly as written. Do not add facts "
    "that are not in the text."
)


def summarize_section(llm: BaseChatModel, header: str, text: str) -> str:
    user = f"Section: {header}\n\n{text}\n\nSummarise this section in 2-4 sentences."
    return _complete(llm, SUMMARY_SYSTEM, user)


def summarize_chapter(llm: BaseChatModel, chapter: str, section_summaries: list[str]) -> str:
    joined = "\n".join(f"- {s}" for s in section_summaries)
    user = (
        f"{chapter}\n\nSummaries of its sections, in order:\n{joined}\n\n"
        "Summarise the whole chapter in one paragraph of 4-6 sentences."
    )
    return _complete(llm, SUMMARY_SYSTEM, user)


def summarize_thesis(
    llm: BaseChatModel, title: str, abstract: str, chapter_summaries: list[str]
) -> str:
    joined = "\n".join(f"- {s}" for s in chapter_summaries)
    user = (
        f"Thesis title: {title}\n\nAbstract:\n{abstract}\n\nChapter summaries:\n{joined}\n\n"
        "Write one paragraph (5-8 sentences) stating the thesis's main aim, objectives, "
        "methods, key findings and overall contribution."
    )
    return _complete(llm, SUMMARY_SYSTEM, user)


# ---------------------------------------------------------------------------
# Multi-query expansion (query time, optional)
# ---------------------------------------------------------------------------
#
# A user's wording often differs from the thesis's ("water stress" vs
# "drought tolerance"). Asking the LLM for a few rephrasings and retrieving
# with each one widens recall; RRF then rewards chunks found by several of
# them. It costs one extra LLM call per question, so it is off by default.

EXPAND_SYSTEM = (
    "You rewrite search queries for a PhD thesis on mulberry genomics. Given a "
    "question, write alternative phrasings that use different wording or likely "
    "technical terms, keeping the same meaning. Output one per line, with no "
    "numbering, quotes or commentary."
)
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def expand_query(llm: BaseChatModel, question: str, n: int = 3) -> list[str]:
    """Return up to ``n`` rephrasings of ``question`` (never the question itself)."""
    raw = _complete(llm, EXPAND_SYSTEM, f"Question: {question}\n\nWrite {n} rephrasings.")
    variants: list[str] = []
    for line in raw.splitlines():
        line = _LIST_MARKER.sub("", line).strip().strip('"').strip()
        # Skip blank lines and preambles like "Here are 3 rephrasings:".
        if not line or line.endswith(":") or line.lower() == question.lower():
            continue
        if line not in variants:
            variants.append(line)
    return variants[:n]


# ---------------------------------------------------------------------------
# Grounded answers (query time)
# ---------------------------------------------------------------------------
#
# The prompt does three jobs: restrict the model to the passages, make every
# claim traceable to a page, and give it a fixed phrase for "the thesis does
# not say", so unsupported questions are easy to detect in evaluation and
# in the UI. Each passage carries a ready-made citation tag, so the model
# copies a tag instead of composing chapter/section/page itself, which an 8B
# model gets wrong surprisingly often.

ANSWER_PROMPT_VERSION = "v1"
NOT_FOUND = "Not found in the thesis."

ANSWER_SYSTEM = f"""You answer questions about a PhD thesis using ONLY the numbered \
passages provided. Rules:
1. Use only facts stated in the passages. Do not use outside knowledge.
2. Cite every sentence that states a fact with the tag shown on the passage it came \
from, copied exactly, at the end of the sentence: "<claim> [Ch.X §Y, p.Z]." Use \
several tags if several passages support it. Never cite a passage that does not \
contain the fact.
3. If the passages do not contain the answer, reply exactly: {NOT_FOUND}
4. Never invent or estimate numbers, names, genotypes or results. Quote numbers \
exactly as written in the passages.
5. If the passages answer only part of the question, answer that part and say \
which part is not covered.
6. Be concise: a short paragraph or a few bullet points. No preamble."""


def _pages(start: int, end: int) -> str:
    if not start:
        return ""
    return f"p.{start}" if start == end or not end else f"pp.{start}\u2013{end}"


def citation_tag(chapter: str, section: str, page_start: int, page_end: int, kind: str = "") -> str:
    """Format a citation such as ``[Ch.4 §4.3, p.62]``.

    Numbered sections get "§"; unnumbered ones (Abstract, a chapter intro) are
    cited by chapter or by title; summary chunks say so, because a summary is
    weaker evidence than the page it summarises.
    """
    numbered = bool(re.match(r"^\d+(\.\d+)*$", section or ""))
    if kind == "summary":
        if chapter and numbered:
            where = f"Ch.{chapter} \u00a7{section} summary"
        else:
            where = f"Ch.{chapter} summary" if chapter else "Thesis summary"
    elif chapter and numbered:
        where = f"Ch.{chapter} \u00a7{section}"
    elif chapter:
        where = f"Ch.{chapter}"
    else:
        where = section or "Front matter"
    pages = _pages(page_start, page_end)
    return f"[{where}, {pages}]" if pages else f"[{where}]"


def passage_citation(p: Passage) -> str:
    return citation_tag(p.chapter, p.section, p.page_start, p.page_end, p.chunk_type)


def format_context(passages: list[Passage]) -> str:
    """Numbered passages, each with its citation tag and heading path."""
    blocks = []
    for i, p in enumerate(passages, start=1):
        blocks.append(f"Passage {i} {passage_citation(p)}\n{p.header}\n\n{p.context.strip()}")
    return "\n\n---\n\n".join(blocks)


def build_messages(question: str, passages: list[Passage]) -> list[BaseMessage]:
    user = f"Passages:\n\n{format_context(passages)}\n\n---\n\nQuestion: {question}"
    return [SystemMessage(content=ANSWER_SYSTEM), HumanMessage(content=user)]


_TAG = re.compile(r"\[[^\[\]]{2,80}\]")


def cited_passages(answer: str, passages: list[Passage]) -> list[int]:
    """1-based numbers of the passages whose citation tag appears in the answer."""
    found = set(_TAG.findall(answer))
    return [i for i, p in enumerate(passages, start=1) if passage_citation(p) in found]


def is_not_found(answer: str) -> bool:
    return answer.strip().lower().startswith(NOT_FOUND.lower().rstrip("."))


def source_dict(n: int, p: Passage, snippet_chars: int = 400) -> dict:
    """What the API and UI receive for each source."""
    text = p.chunk.text.strip()
    if len(text) > snippet_chars:
        text = text[:snippet_chars].rsplit(" ", 1)[0] + "\u2026"
    return {
        "n": n,
        "citation": passage_citation(p),
        "header": p.header,
        "chapter": p.chapter,
        "section": p.section,
        "page_start": p.page_start,
        "page_end": p.page_end,
        "chunk_type": p.chunk_type,
        "chunk_id": p.chunk.id,
        "score": p.score,
        "snippet": text,
    }


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[dict]
    cited: list[int] = field(default_factory=list)
    not_found: bool = False
    broad: bool = False
    queries: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ThesisQA:
    """Retrieve, then answer with citations. Shared by the API, the CLI and the eval."""

    def __init__(
        self,
        settings: Settings,
        index: Index | None = None,
        retriever: Retriever | None = None,
        llm: BaseChatModel | None = None,
    ):
        from .index import Index
        from .retrieve import Retriever

        self.settings = settings
        self.index = index or Index(settings)
        self.retriever = retriever or Retriever(settings, self.index)
        # Answers can be a few paragraphs; cap them so a runaway generation ends.
        self.llm = llm or get_llm(settings, num_predict=700)

    def _finish(self, question: str, text: str, result: RetrievalResult) -> Answer:
        passages = result.passages
        return Answer(
            question=question,
            answer=text,
            sources=[source_dict(i, p) for i, p in enumerate(passages, start=1)],
            cited=cited_passages(text, passages),
            not_found=is_not_found(text),
            broad=result.broad,
            queries=result.queries,
            timings={k: round(v, 3) for k, v in result.timings.items()},
        )

    def ask(self, question: str) -> Answer:
        return self.answer(question, self.retriever.retrieve(question))

    def answer(self, question: str, result: RetrievalResult) -> Answer:
        """Generate from an existing retrieval (the eval scores retrieval separately)."""
        if not result.passages:
            return self._finish(question, NOT_FOUND, result)
        msg = self.llm.invoke(build_messages(question, result.passages))
        return self._finish(question, str(msg.content).strip(), result)

    def ask_stream(self, question: str) -> Iterator[tuple[str, object]]:
        """Yield ("token", str) events while generating, then one ("done", Answer)."""
        result = self.retriever.retrieve(question)
        if not result.passages:
            yield "token", NOT_FOUND
            yield "done", self._finish(question, NOT_FOUND, result)
            return
        parts: list[str] = []
        for piece in self.llm.stream(build_messages(question, result.passages)):
            token = str(piece.content)
            if token:
                parts.append(token)
                yield "token", token
        yield "done", self._finish(question, "".join(parts).strip(), result)


def main() -> None:
    """``python -m thesis_rag.generate "question"``: ask from the terminal."""
    import argparse
    import json

    from .config import get_settings

    parser = argparse.ArgumentParser(description="Ask the indexed thesis a question.")
    parser.add_argument("question")
    parser.add_argument("--json", action="store_true", help="print the full answer object")
    args = parser.parse_args()

    qa = ThesisQA(get_settings())
    if args.json:
        print(json.dumps(qa.ask(args.question).to_dict(), indent=2, ensure_ascii=False))
        return
    answer = None
    for kind, value in qa.ask_stream(args.question):
        if kind == "token":
            print(value, end="", flush=True)
        else:
            answer = value
    print("\n\nSources:")
    for src in answer.sources:
        mark = "*" if src["n"] in answer.cited else " "
        print(f" {mark}[{src['n']}] {src['citation']} {src['header']}  (score {src['score']})")


if __name__ == "__main__":
    main()
