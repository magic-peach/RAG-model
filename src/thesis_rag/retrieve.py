"""Hybrid retrieval: BM25 + dense fused with RRF, optional multi-query, reranking.

Pipeline for one question (each stage can be switched off in settings, which
is how the evaluation compares configurations)::

    question ──► [multi-query: + 3 LLM rephrasings]
             ──► for each query: dense top-20 (Chroma) and BM25 top-20
             ──► Reciprocal Rank Fusion  ──► top-20 candidates
             ──► cross-encoder rerank (MiniLM ms-marco) against the question
             ──► one passage per parent  ──► top-5 passages

Why hybrid: dense embeddings match meaning ("water deficit" ~ "drought") but
are weak on exact identifiers (gene IDs, "GATK", "Table 3"); BM25 is the
opposite. Fusing both covers each other's misses.

Why RRF instead of adding scores: BM25 scores are unbounded and cosine
similarities sit in a narrow band, so a weighted sum needs per-corpus tuning.
RRF uses only ranks, ``score = sum(1 / (k + rank))``, which is scale-free and
has one robust parameter (k=60, from the original paper).

Why rerank: bi-encoders embed query and passage separately; a cross-encoder
reads them together and is much more accurate, but too slow to run over the
whole corpus. So it only re-scores the 20 fused candidates.

Routing: broad questions ("what is the main contribution?") have no single
answering passage, so they also search the LLM-written summary chunks, and
the whole-thesis summary is always included. Specific questions search only
original text and tables, so answers cite the thesis itself, not a summary
of it.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache

from .chunking import SUMMARY, TABLE, TEXT, Chunk, count_tokens
from .config import Settings
from .index import Index

log = logging.getLogger(__name__)

THESIS_SUMMARY_ID = "summary-thesis"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion of several ranked id lists (best first).

    An id's score is the sum over lists of ``1 / (k + rank)``, rank starting
    at 1. Ties keep first-seen order, so a single list comes back unchanged.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


_BROAD = re.compile(
    r"\b("
    r"main (idea|point|aim|goal|contribution|finding|theme|argument)s?"
    r"|summar(y|ise|ize|ising|izing)"
    r"|overall|overview|in (short|brief|a nutshell)"
    r"|(key|major|principal) (finding|contribution|result|takeaway)s?"
    r"|contributions? of (the|this) (thesis|work|study|research)"
    r"|what is (the|this) (thesis|work|study|research) about"
    r"|what('s| is) (this|it) (\w+ )?about"
    r"|tell me about (the|this) (thesis|document|work|study|research)"
    r"|give me an overview"
    r"|(aim|objective|goal)s? of (the|this) (thesis|work|study|research)"
    r"|conclusions? of (the|this) (thesis|work|study|research)"
    r")\b",
    re.IGNORECASE,
)


def is_broad(question: str) -> bool:
    """Whole-document questions that should also retrieve summaries."""
    return bool(_BROAD.search(question))


class CrossEncoderReranker:
    """Cross-encoder reranker via sentence-transformers, loaded on first use.

    The model is downloaded from Hugging Face once and then runs
    locally; tests replace this class with a fake so CI never downloads it.
    """

    def __init__(self, model_name: str, max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder

            # Force a sigmoid so every reranker reports a 0-1 relevance score
            # (bge applies one by default, the MiniLM ms-marco models do not).
            self._model = CrossEncoder(
                self.model_name, max_length=self.max_length, activation_fn=torch.nn.Sigmoid()
            )
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self._load().predict([(query, t) for t in texts])]


@lru_cache(maxsize=2)
def get_reranker(model_name: str) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


@dataclass
class Passage:
    """One numbered passage handed to the LLM, plus what the UI shows as a source."""

    chunk: Chunk  # the child chunk that matched
    context: str  # text given to the LLM: the parent section, or the chunk itself
    header: str  # "Chapter 2: ... › §2.5 Best Practices"
    chapter: str
    section: str
    page_start: int
    page_end: int
    score: float

    @property
    def chunk_type(self) -> str:
        return self.chunk.chunk_type


@dataclass
class RetrievalResult:
    question: str
    queries: list[str]  # the question plus any multi-query rephrasings
    broad: bool
    passages: list[Passage]
    score_kind: str  # "rerank" (0-1 relevance) or "rrf"
    timings: dict[str, float] = field(default_factory=dict)


class Retriever:
    def __init__(self, settings: Settings, index: Index, reranker=None, llm=None):
        self.settings = settings
        self.index = index
        self._reranker = reranker
        self._llm = llm
        self._parent_tokens: dict[str, int] = {}

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = get_reranker(self.settings.reranker_model)
        return self._reranker

    @property
    def llm(self):
        if self._llm is None:
            from .generate import get_llm

            # A little temperature so the rephrasings actually differ.
            self._llm = get_llm(self.settings, temperature=0.3, num_predict=200)
        return self._llm

    def expand(self, question: str) -> list[str]:
        from .generate import expand_query

        try:
            return expand_query(self.llm, question, self.settings.num_query_variants)
        except Exception:  # expansion is an optimisation; never fail the question over it
            log.exception("Query expansion failed; using the original question only")
            return []

    def retrieve(self, question: str, allow_summaries: bool | None = None) -> RetrievalResult:
        """``allow_summaries`` overrides routing: False keeps to original text only."""
        s, idx = self.settings, self.index
        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        if idx.has_summaries():
            broad = is_broad(question) if allow_summaries is None else allow_summaries
            types: list[str] | None = [TEXT, TABLE, SUMMARY] if broad else [TEXT, TABLE]
        else:
            broad, types = False, None  # baseline index: plain text only, nothing to filter

        queries = [question]
        if s.use_multi_query:
            queries += self.expand(question)
            timings["expand"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        rankings = []
        for q in queries:
            rankings.append([h.chunk_id for h in idx.dense_search(q, s.retrieve_k, types)])
            if s.use_bm25:
                rankings.append([h.chunk_id for h in idx.bm25_search(q, s.retrieve_k, types)])
        fused = rrf_fuse(rankings, s.rrf_k)[: s.retrieve_k]
        timings["search"] = time.perf_counter() - t1

        # A broad question always sees the whole-thesis summary, ranked first.
        pinned = [THESIS_SUMMARY_ID] if broad and THESIS_SUMMARY_ID in idx.chunks else []
        candidates = pinned + [cid for cid, _ in fused if cid not in pinned]
        scores = dict(fused)

        if s.use_reranker and candidates:
            t2 = time.perf_counter()
            texts = [idx.chunks[cid].embed_text for cid in candidates]
            scores = dict(zip(candidates, self.reranker.score(question, texts), strict=True))
            rest = sorted(candidates[len(pinned) :], key=lambda cid: -scores[cid])
            candidates = pinned + rest
            timings["rerank"] = time.perf_counter() - t2
            score_kind = "rerank"
        else:
            score_kind = "rrf"

        passages = self._to_passages(candidates, scores)
        timings["total"] = time.perf_counter() - t0
        return RetrievalResult(question, queries, broad, passages, score_kind, timings)

    def _to_passages(self, ranked_ids: list[str], scores: dict[str, float]) -> list[Passage]:
        """Keep the best child per parent, so the LLM never sees one section twice."""
        s, idx = self.settings, self.index
        passages: list[Passage] = []
        seen: set[str] = set()
        for cid in ranked_ids:
            chunk = idx.chunks[cid]
            parent = idx.parents.get(chunk.parent_id) if s.use_parent_context else None
            if parent is not None and self._fits(parent.id, parent.text):
                key = parent.id
                context, header, section = parent.text, parent.header, parent.section
                pages = (parent.page_start, parent.page_end)
            else:
                # No parent, or one too long for the context budget: send the chunk.
                key = cid
                context, header, section = chunk.text, chunk.header, chunk.section
                pages = (chunk.page_start, chunk.page_end)
            if key in seen:
                continue
            seen.add(key)
            passages.append(
                Passage(
                    chunk=chunk,
                    context=context,
                    header=header,
                    chapter=chunk.chapter,
                    section=section,
                    page_start=pages[0],
                    page_end=pages[1],
                    score=round(float(scores.get(cid, 0.0)), 4),
                )
            )
            if len(passages) == s.top_k:
                break
        return passages

    def _fits(self, parent_id: str, text: str) -> bool:
        if parent_id not in self._parent_tokens:
            self._parent_tokens[parent_id] = count_tokens(text)
        return self._parent_tokens[parent_id] <= self.settings.max_passage_tokens
