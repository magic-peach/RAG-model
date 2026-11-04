"""Persistent index: build it once per (PDF, configuration), load it fast afterwards.

On-disk layout (``settings.index_dir``, e.g. ``data/index/structured-600-72/``)::

    chroma/          Chroma persistent client: one vector per chunk (cosine)
    docstore.json    every chunk and parent, with metadata (the source of truth)
    manifest.json    file hash + settings fingerprint; written last

and, shared across configurations, ``data/cache/``::

    <file-hash>/parsed.json   parser output (parsing takes ~30 s)
    summaries.json            LLM summaries keyed by a hash of their input

Idempotency: :func:`build_index` computes a fingerprint from the PDF's SHA-256
and every setting that changes the index (chunk size, overlap, embedding
model, ...). If ``manifest.json`` already has that fingerprint, nothing is
rebuilt. Because the manifest is written only after Chroma and the docstore,
an interrupted build is simply redone next time.

BM25 is not persisted: building it from the docstore takes milliseconds for a
document of this size, and it avoids a second copy of the corpus going stale.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .chunking import (
    SUMMARY,
    TEXT,
    Chunk,
    Parent,
    baseline_chunks,
    chapter_label,
    chunk_document,
    count_tokens,
    make_summary_chunk,
)
from .config import Settings
from .ingest import ParsedDocument, file_sha256, parse_pdf

log = logging.getLogger(__name__)

PARSER_VERSION = "2"  # bump when ingest/chunking output changes
COLLECTION = "chunks"
ProgressFn = Callable[[str, float], None]


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def with_retry(fn: Callable, attempts: int = 4, wait: float = 15.0):
    """Call ``fn()``, retrying transient Ollama failures with linear backoff.

    Under memory pressure Ollama can fail to (re)load a model ("timed out
    waiting for llama-server to start"); a later attempt usually succeeds.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # ollama raises ResponseError, httpx errors, ...
            if attempt == attempts:
                raise
            log.warning("Ollama call failed (%s); retry %d/%d", exc, attempt, attempts - 1)
            time.sleep(wait * attempt)


class Embedder:
    """nomic-embed-text with the task prefixes it was trained with.

    nomic-embed-text expects "search_document: " on indexed text and
    "search_query: " on queries; without them retrieval quality drops
    noticeably. Locally the model is served by Ollama; the hosted demo runs
    the same weights through fastembed (ONNX), which needs no model server.
    """

    def __init__(self, settings: Settings):
        self._fastembed = None
        if settings.embedding_provider == "fastembed":
            from fastembed import TextEmbedding  # optional dependency: pip install ".[deploy]"

            self._fastembed = TextEmbedding(settings.fastembed_model)
            self._prefixed = True
        else:
            from langchain_ollama import OllamaEmbeddings

            self._model = OllamaEmbeddings(
                model=settings.embedding_model, base_url=settings.ollama_base_url
            )
            self._prefixed = "nomic" in settings.embedding_model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._fastembed is not None:
            return [v.tolist() for v in self._fastembed.embed(texts)]
        return with_retry(lambda: self._model.embed_documents(texts))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._prefixed:
            texts = [f"search_document: {t}" for t in texts]
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([f"search_query: {text}" if self._prefixed else text])[0]


# ---------------------------------------------------------------------------
# BM25 tokenisation
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "which",
        "with",
        "what",
        "how",
        "why",
        "when",
        "who",
        "does",
        "do",
        "did",
        "their",
        "there",
        "these",
        "those",
    ]
)
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def bm25_tokenize(text: str) -> list[str]:
    """Lowercase word tokens; keeps identifiers like 'mi00022', 'go:...' parts, '3.10'."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def fingerprint(file_hash: str, settings: Settings) -> str:
    relevant = {
        "file": file_hash,
        "parser": PARSER_VERSION,
        "structured": settings.structured_chunking,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "max_parent_tokens": settings.max_parent_tokens,
        "embedding_model": settings.embedding_model,
        "summaries": settings.summarize_sections and settings.structured_chunking,
        "llm": settings.llm_model if settings.summarize_sections else "",
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]


def _cache_dir(settings: Settings) -> Path:
    return settings.data_dir / "cache"


def load_or_parse(pdf_path: Path, file_hash: str, settings: Settings) -> ParsedDocument:
    cache = _cache_dir(settings) / file_hash[:16] / f"parsed-v{PARSER_VERSION}.json"
    if cache.is_file():
        return ParsedDocument.from_dict(json.loads(cache.read_text()))
    doc = parse_pdf(pdf_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(doc.to_dict()))
    return doc


class SummaryCache:
    """JSON cache so re-chunking or re-indexing never re-runs the LLM on the same input."""

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, str] = json.loads(path.read_text()) if path.is_file() else {}

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()

    def get_or_make(self, key: str, make: Callable[[], str]) -> str:
        if key not in self._data:
            self._data[key] = make()
            self.save()  # save as we go: summarising takes minutes
        return self._data[key]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=1))


def build_summaries(
    doc: ParsedDocument,
    parents: dict[str, Parent],
    settings: Settings,
    progress: ProgressFn | None = None,
    min_tokens: int = 120,
) -> tuple[list[Chunk], dict[str, Parent]]:
    """Section -> chapter -> thesis summaries, each stored as a ``summary`` chunk."""
    from .generate import (
        SUMMARY_PROMPT_VERSION,
        get_llm,
        summarize_chapter,
        summarize_section,
        summarize_thesis,
    )

    # Cap output length: summaries are 2-6 sentences, and an 8B model occasionally rambles.
    llm = get_llm(settings, num_predict=300)
    cache = SummaryCache(_cache_dir(settings) / "summaries.json")
    tag = f"{SUMMARY_PROMPT_VERSION}|{settings.llm_model}"
    chunks: list[Chunk] = []
    new_parents: dict[str, Parent] = {}

    def add(chunk_parent: tuple[Chunk, Parent]) -> None:
        chunks.append(chunk_parent[0])
        new_parents[chunk_parent[1].id] = chunk_parent[1]

    # Level 1: one summary per text parent in the body chapters. Front matter
    # (declaration, acknowledgements...) is skipped; the abstract is used below.
    targets = [
        p
        for p in parents.values()
        if p.chunk_type == TEXT and p.chapter and count_tokens(p.text) >= min_tokens
    ]
    by_chapter: dict[str, list[str]] = {}
    chapter_titles: dict[str, str] = {}
    chapter_pages: dict[str, tuple[int, int]] = {}
    for i, p in enumerate(targets):
        if progress:
            progress(f"Summarising {p.header}", i / max(len(targets), 1))
        key = cache.key(tag, "section", p.header, p.text)
        text = cache.get_or_make(key, lambda p=p: summarize_section(llm, p.header, p.text))
        add(
            make_summary_chunk(
                id=f"summary-{p.id}",
                text=text,
                header=f"{p.header} › Summary",
                chapter=p.chapter,
                section=p.section,
                page_start=p.page_start,
                page_end=p.page_end,
            )
        )
        by_chapter.setdefault(p.chapter, []).append(text)
        chapter_titles[p.chapter] = p.header.split(" › ")[0]
        lo, hi = chapter_pages.get(p.chapter, (p.page_start, p.page_end))
        chapter_pages[p.chapter] = (min(lo, p.page_start), max(hi, p.page_end))

    # Level 2: chapters, from their section summaries.
    chapter_summaries = []
    for ch, summaries in by_chapter.items():
        if progress:
            progress(f"Summarising {chapter_titles[ch]}", 0.95)
        key = cache.key(tag, "chapter", chapter_titles[ch], *summaries)
        text = cache.get_or_make(
            key, lambda ch=ch, s=summaries: summarize_chapter(llm, chapter_titles[ch], s)
        )
        chapter_summaries.append(f"{chapter_titles[ch]}: {text}")
        add(
            make_summary_chunk(
                id=f"summary-ch{ch}",
                text=text,
                header=f"{chapter_titles[ch]} › Chapter summary",
                chapter=ch,
                section="",
                page_start=chapter_pages[ch][0],
                page_end=chapter_pages[ch][1],
            )
        )

    # Level 3: the whole thesis, from chapter summaries plus the abstract.
    abstract = next((s.text for s in doc.sections if s.title.lower() == "abstract"), "")
    if chapter_summaries:
        key = cache.key(tag, "thesis", doc.title, abstract, *chapter_summaries)
        text = cache.get_or_make(
            key, lambda: summarize_thesis(llm, doc.title, abstract, chapter_summaries)
        )
        pages = [p for lo_hi in chapter_pages.values() for p in lo_hi]
        add(
            make_summary_chunk(
                id="summary-thesis",
                text=text,
                header=f"{chapter_label('', '')} › Whole-thesis summary",
                chapter="",
                section="Whole thesis",
                page_start=min(pages),
                page_end=max(pages),
            )
        )
    return chunks, new_parents


def build_index(
    pdf_path: str | Path,
    settings: Settings,
    force: bool = False,
    progress: ProgressFn | None = None,
) -> dict:
    """Parse, chunk, summarise, embed and persist ``pdf_path``. Returns the manifest."""
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    say = progress or (lambda msg, frac: log.info("%s (%.0f%%)", msg, frac * 100))

    file_hash = file_sha256(pdf_path)
    fp = fingerprint(file_hash, settings)
    index_dir = settings.index_dir
    manifest_path = index_dir / "manifest.json"
    if not force and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("fingerprint") == fp:
            say("Index is up to date", 1.0)
            return {**manifest, "cached": True}

    started = time.time()
    say("Parsing PDF", 0.0)
    if settings.structured_chunking:
        doc = load_or_parse(pdf_path, file_hash, settings)
        title = doc.title
        chunks, parents = chunk_document(
            doc, settings.chunk_size, settings.chunk_overlap, settings.max_parent_tokens
        )
        if settings.summarize_sections:
            s_chunks, s_parents = build_summaries(doc, parents, settings, progress=say)
            chunks += s_chunks
            parents.update(s_parents)
    else:
        title = pdf_path.stem
        chunks, parents = baseline_chunks(str(pdf_path))

    say(f"Embedding {len(chunks)} chunks", 0.97)
    embedder = Embedder(settings)
    vectors = []
    for i in range(0, len(chunks), 32):
        vectors += embedder.embed_documents([c.embed_text for c in chunks[i : i + 32]])

    # Rebuild from scratch: the index holds exactly one document (see README).
    # Invalidate the manifest first so a crash mid-write cannot look "up to date".
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.unlink(missing_ok=True)
    _write_chroma(index_dir, chunks, vectors)
    (index_dir / "docstore.json").write_text(
        json.dumps(
            {
                "chunks": [c.to_dict() for c in chunks],
                "parents": {k: p.to_dict() for k, p in parents.items()},
            }
        )
    )
    counts: dict[str, int] = {}
    for c in chunks:
        counts[c.chunk_type] = counts.get(c.chunk_type, 0) + 1
    manifest = {
        "fingerprint": fp,
        "file_hash": file_hash,
        "source": pdf_path.name,
        "title": title,
        "chunk_counts": counts,
        "num_parents": len(parents),
        "settings": {
            "structured_chunking": settings.structured_chunking,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model,
            "summaries": settings.summarize_sections and settings.structured_chunking,
        },
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "build_seconds": round(time.time() - started, 1),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    say("Indexed", 1.0)
    return {**manifest, "cached": False}


def _chroma_client(index_dir: Path):
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    return chromadb.PersistentClient(
        path=str(index_dir / "chroma"), settings=ChromaSettings(anonymized_telemetry=False)
    )


def _write_chroma(index_dir: Path, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    client = _chroma_client(index_dir)
    # Drop and recreate the collection rather than deleting the folder: Chroma
    # caches clients per path in-process, and a deleted folder leaves that cache
    # pointing at a removed database file.
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    col = client.create_collection(
        COLLECTION, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None
    )
    for i in range(0, len(chunks), 256):
        batch = chunks[i : i + 256]
        col.add(
            ids=[c.id for c in batch],
            embeddings=vectors[i : i + 256],
            documents=[c.embed_text for c in batch],
            metadatas=[{"chunk_type": c.chunk_type, "page_start": c.page_start} for c in batch],
        )


# ---------------------------------------------------------------------------
# Load / query
# ---------------------------------------------------------------------------


@dataclass
class Hit:
    chunk_id: str
    score: float


class Index:
    """Read side of the index: chunk lookup, dense search and BM25 search."""

    def __init__(self, settings: Settings, embedder: Embedder | None = None):
        from rank_bm25 import BM25Okapi

        self.settings = settings
        self.dir = settings.index_dir
        manifest_path = self.dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"No index at {self.dir}; run ingestion first")
        self.manifest = json.loads(manifest_path.read_text())
        store = json.loads((self.dir / "docstore.json").read_text())
        self.chunks: dict[str, Chunk] = {c["id"]: Chunk(**c) for c in store["chunks"]}
        self.parents: dict[str, Parent] = {k: Parent(**p) for k, p in store["parents"].items()}
        self._ids = list(self.chunks)
        self._bm25 = BM25Okapi([bm25_tokenize(self.chunks[i].embed_text) for i in self._ids])
        self._embedder = embedder
        self._collection = None
        # The API's warm-up thread and the first request can arrive together;
        # loading the model or opening Chroma twice at once fails, so serialise it.
        self._load_lock = threading.RLock()

    @classmethod
    def exists(cls, settings: Settings) -> bool:
        return (settings.index_dir / "manifest.json").is_file()

    @property
    def embedder(self) -> Embedder:
        with self._load_lock:
            if self._embedder is None:
                self._embedder = Embedder(self.settings)
            return self._embedder

    @property
    def collection(self):
        with self._load_lock:
            if self._collection is None:
                self._collection = _chroma_client(self.dir).get_collection(COLLECTION)
            return self._collection

    def dense_search(self, query: str, k: int, types: list[str] | None = None) -> list[Hit]:
        where = {"chunk_type": {"$in": types}} if types else None
        res = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(k, len(self._ids)),
            where=where,
            include=["distances"],
        )
        # Cosine distance -> similarity, so higher is better everywhere.
        return [
            Hit(cid, 1.0 - dist)
            for cid, dist in zip(res["ids"][0], res["distances"][0], strict=True)
        ]

    def bm25_search(self, query: str, k: int, types: list[str] | None = None) -> list[Hit]:
        scores = self._bm25.get_scores(bm25_tokenize(query))
        ranked = sorted(zip(self._ids, scores, strict=True), key=lambda x: -x[1])
        hits = []
        for cid, score in ranked:
            if score <= 0:
                break
            if types and self.chunks[cid].chunk_type not in types:
                continue
            hits.append(Hit(cid, float(score)))
            if len(hits) == k:
                break
        return hits

    def has_summaries(self) -> bool:
        return any(c.chunk_type == SUMMARY for c in self.chunks.values())
