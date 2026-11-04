"""LLM calls: hierarchical summaries at ingest time, grounded answers at query time.

All generation goes through :func:`get_llm`, a ``ChatOllama`` client for the
local Llama 3.1 model. Low temperature (0.1) because we want faithful
extraction, not creativity; ``num_ctx`` is raised from Ollama's small default
so several retrieved sections fit in one prompt.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .config import Settings

if TYPE_CHECKING:
    from .index import Index
    from .retrieve import Passage, RetrievalResult, Retriever

log = logging.getLogger(__name__)


def get_llm(settings: Settings, **overrides) -> BaseChatModel:
    """Chat model for the configured provider. Ollama-style overrides
    (``num_predict``, ``format="json"``) are translated for Groq."""
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq  # optional dependency: pip install ".[deploy]"

        params: dict = {
            "model": settings.groq_model,
            "temperature": overrides.get("temperature", settings.temperature),
            "max_tokens": overrides.get("num_predict", settings.max_answer_tokens),
        }
        if overrides.get("format") == "json":
            params["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatGroq(**params)

    from langchain_ollama import ChatOllama

    params = {
        "model": settings.llm_model,
        "base_url": settings.ollama_base_url,
        "temperature": settings.temperature,
        "num_ctx": settings.num_ctx,
        # Keep the model in memory between questions: reloading 5 GB costs seconds.
        "keep_alive": settings.keep_alive,
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
# Answers (query time)
# ---------------------------------------------------------------------------
#
# Each reply has up to three parts:
#
#   1. a mode: "thesis" (answered from the passages, with citations) or
#      "general" (the thesis does not cover it, so the model answers from its
#      own knowledge and the UI labels it "Not from the thesis");
#   2. the answer, direct first sentence, plain language;
#   3. "Reasoning:" with a few bullets saying which passages were used and
#      how, so a reader can check the answer instead of trusting it.
#
# The mode is a fixed first line ("GENERAL:") rather than a judgement made
# afterwards, so it can be read while the answer streams. Each passage carries
# a ready-made citation tag that the model copies, because an 8B model gets
# chapter/section/page wrong when it has to compose them itself.

ANSWER_PROMPT_VERSION = "v3"
NOT_FOUND = "Not found in the thesis."
GENERAL_MARKER = "GENERAL:"
REASONING_MARKER = "Reasoning:"

ANSWER_SYSTEM = f"""You are a friendly research assistant for a PhD thesis on mulberry \
genomics. You receive numbered passages from the thesis and a question.

First decide whether the passages answer the question.

If they do:
- Start with a direct one-sentence answer, then add detail in plain language. Explain \
technical terms briefly so a non-specialist can follow.
- End every sentence that states a fact from the thesis with the citation tag of its \
passage, copied exactly as shown after "cite as" (square brackets included).
- You may add one or two sentences of general background if it helps understanding; \
end those with "(general knowledge)" instead of a tag, and never present them as the \
thesis's findings.
- Never invent or estimate numbers, names, genotypes or results. Quote numbers exactly.

If they do not (the topic is not in the passages, or it is a general question):
- Write the line {GENERAL_MARKER} first, then answer helpfully from general knowledge, \
without citations, and do not claim the thesis says anything about it.

Always finish with a line "{REASONING_MARKER}" followed by 1-3 short bullet points \
explaining how you reached the answer: which passages you used and what they say, or \
why the thesis does not cover the question.

Keep the whole reply under 250 words. No preamble."""

GENERAL_SYSTEM = f"""You are a friendly research assistant attached to a PhD thesis on \
mulberry genomics. This message does not need the thesis (a greeting, thanks, or a \
general question). Reply briefly and helpfully from general knowledge. Start with the \
line {GENERAL_MARKER} and finish with a line "{REASONING_MARKER}" and one short bullet."""

_SMALL_TALK = re.compile(
    r"^\s*(hi|hello|hey|hiya|yo|thanks|thank you|thx|ok|okay|cool|great|bye|good (morning|"
    r"afternoon|evening)|who are you|what can you do)\b[\s!.?]*$",
    re.IGNORECASE,
)
_FOLLOW_UP = re.compile(
    r"\b(it|its|this|that|these|those|they|them|he|she|more|why|how so|explain|elaborate|"
    r"example|simpler|again)\b",
    re.IGNORECASE,
)


def is_small_talk(question: str) -> bool:
    return bool(_SMALL_TALK.match(question))


def retrieval_query(question: str, history: list[tuple[str, str]]) -> str:
    """Make a short follow-up ("why is that?") searchable by adding the previous question."""
    if history and len(question.split()) <= 8 and _FOLLOW_UP.search(question):
        return f"{history[-1][0]} {question}"
    return question


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
        blocks.append(
            f"Passage {i} (cite as {passage_citation(p)})\n{p.header}\n\n{p.context.strip()}"
        )
    return "\n\n---\n\n".join(blocks)


def _history_messages(history: list[tuple[str, str]]) -> list[BaseMessage]:
    msgs: list[BaseMessage] = []
    for q, a in history:
        msgs.append(HumanMessage(content=q))
        msgs.append(AIMessage(content=a[:800]))  # earlier answers only for context
    return msgs


def build_messages(
    question: str, passages: list[Passage], history: list[tuple[str, str]] | None = None
) -> list[BaseMessage]:
    user = f"Passages:\n\n{format_context(passages)}\n\n---\n\nQuestion: {question}"
    return [
        SystemMessage(content=ANSWER_SYSTEM),
        *_history_messages(history or []),
        HumanMessage(content=user),
    ]


def build_general_messages(
    question: str, history: list[tuple[str, str]] | None = None
) -> list[BaseMessage]:
    return [
        SystemMessage(content=GENERAL_SYSTEM),
        *_history_messages(history or []),
        HumanMessage(content=question),
    ]


def split_reply(raw: str) -> tuple[str, str, str]:
    """Raw model output -> (mode, answer, reasoning)."""
    text = raw.strip()
    mode = "thesis"
    if text.upper().startswith(GENERAL_MARKER):
        mode, text = "general", text[len(GENERAL_MARKER) :].lstrip()
    elif is_not_found(text):  # an older-style refusal still counts as off-thesis
        mode = "general"
    m = re.search(r"\n?\s*\**\s*reasoning\s*\**\s*:\s*\**", text, re.IGNORECASE)
    if m:
        return mode, text[: m.start()].strip(), text[m.end() :].strip()
    return mode, text, ""


class StreamSplitter:
    """Turns raw streamed tokens into ("mode" | "token" | "reasoning", text) events.

    Holds back just enough text to recognise the GENERAL: prefix and the
    Reasoning: marker, so neither ever flashes on screen while streaming.
    """

    def __init__(self):
        self.buffer = ""
        self.mode: str | None = None
        self.in_reasoning = False

    def feed(self, token: str) -> list[tuple[str, str]]:
        self.buffer += token
        events: list[tuple[str, str]] = []
        if self.mode is None:
            head = self.buffer.lstrip()
            if len(head) < len(GENERAL_MARKER) and GENERAL_MARKER.startswith(head.upper()):
                return events  # could still become "GENERAL:"
            if head.upper().startswith(GENERAL_MARKER):
                self.mode, self.buffer = "general", head[len(GENERAL_MARKER) :].lstrip()
            else:
                self.mode, self.buffer = "thesis", head
            events.append(("mode", self.mode))
        if not self.in_reasoning:
            m = re.search(r"\n?\s*\**\s*reasoning\s*\**\s*:\s*\**", self.buffer, re.IGNORECASE)
            if m:
                if m.start():
                    events.append(("token", self.buffer[: m.start()]))
                self.buffer, self.in_reasoning = self.buffer[m.end() :], True
            else:
                safe = len(self.buffer) - (len(REASONING_MARKER) + 6)
                if safe > 0:
                    events.append(("token", self.buffer[:safe]))
                    self.buffer = self.buffer[safe:]
                return events
        if self.buffer:
            events.append(("reasoning", self.buffer))
            self.buffer = ""
        return events

    def flush(self) -> list[tuple[str, str]]:
        events = []
        if self.mode is None:
            self.mode = "thesis"
            events.append(("mode", self.mode))
        if self.buffer:
            events.append(("reasoning" if self.in_reasoning else "token", self.buffer))
            self.buffer = ""
        return events


_TAG = re.compile(r"\[[^\[\]]{2,80}\]")


_PASSAGE_ONLY = re.compile(r"\[(?:passage|source)\s*(\d+)\]", re.IGNORECASE)
_PASSAGE_PREFIX = re.compile(r"\[(?:passage|source)\s*\d+\s*[,:;\u2013-]\s*", re.IGNORECASE)


def normalize_citations(answer: str, passages: list[Passage]) -> str:
    """Repair the citation slips an 8B model makes, so tags match the sources.

    "[Passage 4, Ch.4 §4.11, p.76]" -> "[Ch.4 §4.11, p.76]", "[Passage 2]" ->
    that passage's tag, "(cite as [tag])" -> "[tag]", and a stray "<claim>"
    placeholder is dropped. Bare numbers like "[4]" are left alone: the
    thesis's own reference list uses them.
    """

    def by_number(m: re.Match) -> str:
        n = int(m.group(1))
        return passage_citation(passages[n - 1]) if 1 <= n <= len(passages) else m.group(0)

    text = re.sub(r"<\s*claim\s*>", "", answer, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*cite as\s*(\[[^\[\]]+\])\s*\)", r"\1", text, flags=re.IGNORECASE)
    text = _PASSAGE_PREFIX.sub("[", text)
    text = _PASSAGE_ONLY.sub(by_number, text)
    text = re.sub(r"\.\s+\[", " [", text)  # "fact. [tag]." -> "fact [tag]."
    return re.sub(r"[ \t]{2,}", " ", text).strip()


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
    mode: str = "thesis"  # "thesis" (from the passages) or "general" (not from the thesis)
    reasoning: str = ""
    not_found: bool = False  # kept for the eval: True when the thesis did not answer
    broad: bool = False
    queries: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ThesisQA:
    """Retrieve, then answer. Shared by the API, the CLI and the eval."""

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
        # Cap answer length: the prompt asks for < 250 words plus reasoning.
        self.llm = llm or get_llm(settings, num_predict=settings.max_answer_tokens)

    @property
    def _general_allowed(self) -> bool:
        return getattr(self.settings, "allow_general_knowledge", True)

    def _finish(
        self, question: str, raw: str, result: RetrievalResult, timings: dict | None = None
    ) -> Answer:
        passages = result.passages
        mode, text, reasoning = split_reply(raw)
        if mode == "general" and not self._general_allowed:
            text = NOT_FOUND
        text = normalize_citations(text, passages)
        reasoning = normalize_citations(reasoning, passages)
        cited = cited_passages(text + "\n" + reasoning, passages) if mode == "thesis" else []
        return Answer(
            question=question,
            answer=text,
            sources=[source_dict(i, p) for i, p in enumerate(passages, start=1)],
            cited=cited,
            mode=mode,
            reasoning=reasoning,
            not_found=mode == "general",
            broad=result.broad,
            queries=result.queries,
            timings={k: round(v, 3) for k, v in {**result.timings, **(timings or {})}.items()},
        )

    def _plan(self, question: str, history: list[tuple[str, str]]):
        """Retrieve (unless it is small talk) and build the prompt."""
        from .retrieve import RetrievalResult

        history = history[-self.settings.history_turns :] if history else []
        if is_small_talk(question):
            result = RetrievalResult(question, [question], False, [], "none", {})
            return result, build_general_messages(question, history)
        result = self.retriever.retrieve(retrieval_query(question, history))
        if not result.passages:
            return result, build_general_messages(question, history)
        return result, build_messages(question, result.passages, history)

    def ask(self, question: str, history: list[tuple[str, str]] | None = None) -> Answer:
        result, messages = self._plan(question, history or [])
        msg = self.llm.invoke(messages)
        return self._finish(question, str(msg.content), result)

    def answer(self, question: str, result: RetrievalResult) -> Answer:
        """Generate from an existing retrieval (the eval scores retrieval separately)."""
        messages = (
            build_messages(question, result.passages)
            if result.passages
            else build_general_messages(question)
        )
        msg = self.llm.invoke(messages)
        return self._finish(question, str(msg.content), result)

    def ask_stream(
        self, question: str, history: list[tuple[str, str]] | None = None
    ) -> Iterator[tuple[str, object]]:
        """Yield ("mode" | "token" | "reasoning", str) events, then one ("done", Answer)."""
        t0 = time.perf_counter()
        result, messages = self._plan(question, history or [])
        splitter, parts, first = StreamSplitter(), [], None
        for piece in self.llm.stream(messages):
            token = str(piece.content)
            if not token:
                continue
            if first is None:
                first = time.perf_counter() - t0
            parts.append(token)
            yield from splitter.feed(token)
        yield from splitter.flush()
        timings = {"first_token": first or 0.0, "answer_total": time.perf_counter() - t0}
        yield "done", self._finish(question, "".join(parts), result, timings)


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
    answer, in_reasoning = None, False
    for kind, value in qa.ask_stream(args.question):
        if kind == "mode" and value == "general":
            print("[Not from the thesis: general knowledge]\n")
        elif kind == "token":
            print(value, end="", flush=True)
        elif kind == "reasoning":
            if not in_reasoning:
                print("\n\nReasoning:", end=" ")
                in_reasoning = True
            print(value, end="", flush=True)
        elif kind == "done":
            answer = value
    print("\n\nSources:")
    for src in answer.sources:
        mark = "*" if src["n"] in answer.cited else " "
        print(f" {mark}[{src['n']}] {src['citation']} {src['header']}  (score {src['score']})")
    print(f"\n{answer.timings}")


if __name__ == "__main__":
    main()
