"""Chunking: turn parsed sections and tables into retrievable chunks.

Design
------
*Small chunks for search, large parents for the answer.* Embeddings of
~600-token chunks are precise (one idea per vector), but an LLM answers better
with the surrounding section. So every child chunk carries a ``parent_id``;
retrieval scores children, and generation receives the parent's text.

*Parents are the largest enclosing section that fits a budget.* Many thesis
sections are tiny (e.g. §2.5.1-§2.5.11 are ~30 words each). Handing the LLM
one 30-word fragment wastes the retrieval hit, so leaf sections are grouped
under their two-level number (§2.5) when the group fits ``max_parent_tokens``;
otherwise each leaf section is its own parent.

*Section-aware splitting.* Chunks never cross a section boundary because each
section is split separately. Inside a section, a recursive splitter prefers to
break at sub-headings, then paragraphs, then sentences, with ~12% overlap so a
sentence cut at a boundary still appears whole in one of the two chunks.

*Contextual header.* Before embedding, each chunk is prefixed with
"Chapter 4: Results and Discussion › §4.3 Gapped Alignment". A chunk that
says "the alignment rate was 87%" then still matches a query about
"gapped alignment results" even though those words are only in the heading.

Token counts use tiktoken's ``cl100k_base`` as an approximation; Llama 3.1 and
nomic-embed-text use different tokenizers, but the difference (~10-20%) is
well inside the safety margins used here.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .ingest import ParsedDocument, Section, Table

TEXT, TABLE, SUMMARY = "text", "table", "summary"


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))


@dataclass
class Chunk:
    id: str
    text: str  # body text shown to users and passed to the LLM when no parent is used
    header: str  # "Chapter 1: Introduction › §1.10.2 Drought Stress"
    chunk_type: str  # text | table | summary
    parent_id: str
    chapter: str = ""
    section: str = ""  # "1.10.2", or a title for unnumbered sections
    section_title: str = ""
    page_start: int = 0
    page_end: int = 0
    seq: int = 0  # position among the parent's children

    @property
    def embed_text(self) -> str:
        """What actually gets embedded and BM25-indexed: header + body."""
        return f"{self.header}\n\n{self.text}" if self.header else self.text

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Parent:
    id: str
    text: str
    header: str
    chunk_type: str
    chapter: str = ""
    section: str = ""
    section_title: str = ""
    page_start: int = 0
    page_end: int = 0
    child_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------


def chapter_label(chapter: str, chapter_title: str) -> str:
    if not chapter:
        return "Front Matter"
    return f"Chapter {chapter}: {chapter_title}" if chapter_title else f"Chapter {chapter}"


def section_label(number: str, title: str) -> str:
    return f"§{number} {title}".strip() if number else title


def make_header(chapter: str, chapter_title: str, number: str, title: str) -> str:
    """'Chapter 1: Introduction › §1.10.2 Drought Stress'."""
    parts = [chapter_label(chapter, chapter_title)]
    sec = section_label(number, title)
    if sec and sec != chapter_title:
        parts.append(sec)
    return " › ".join(parts)


# ---------------------------------------------------------------------------
# Parent grouping
# ---------------------------------------------------------------------------


def _group_key(section: Section) -> str:
    """Two-level number ("2.5" for "2.5.7"); unnumbered sections stand alone."""
    if not section.number:
        return section.id
    return ".".join(section.number.split(".")[:2])


def group_parents(sections: list[Section], max_parent_tokens: int) -> list[list[Section]]:
    """Group consecutive leaf sections that share a two-level number, if they fit."""
    groups: list[list[Section]] = []
    for sec in sections:
        if groups and _group_key(groups[-1][0]) == _group_key(sec):
            groups[-1].append(sec)
        else:
            groups.append([sec])

    result: list[list[Section]] = []
    for g in groups:
        total = sum(count_tokens(s.text) for s in g)
        if len(g) > 1 and total > max_parent_tokens:
            result.extend([s] for s in g)  # too big together: each leaf is its own parent
        else:
            result.append(g)
    return result


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens,
        # Sub-headings first, then paragraphs, lines, sentences, words.
        separators=["\n#### ", "\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
        add_start_index=True,
        strip_whitespace=True,
    )


def split_section(section: Section, splitter: RecursiveCharacterTextSplitter) -> list[dict]:
    """Split one section into pieces, each with the page range it came from."""
    pieces = []
    for d in splitter.create_documents([section.text]):
        start = max(d.metadata.get("start_index", 0), 0)
        page_start, page_end = section.pages_for_span(start, start + len(d.page_content))
        pieces.append({"text": d.page_content, "page_start": page_start, "page_end": page_end})
    return pieces


def _clean_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_document(
    doc: ParsedDocument, chunk_size: int, chunk_overlap: int, max_parent_tokens: int
) -> tuple[list[Chunk], dict[str, Parent]]:
    """Structured chunking: section-aware text chunks + table chunks, with parents."""
    splitter = make_splitter(chunk_size, chunk_overlap)
    chunks: list[Chunk] = []
    parents: dict[str, Parent] = {}

    for group in group_parents(doc.sections, max_parent_tokens):
        first, last = group[0], group[-1]
        if len(group) == 1:
            p_number, p_title = first.number, first.title
        else:
            # A group is labelled by its two-level number and that heading's title.
            p_number = _group_key(first)
            p_title = doc.heading_titles.get(p_number, "")
        parent_id = f"parent-{first.id}"
        if len(group) == 1:
            parent_text = first.text
        else:
            parent_text = "\n\n".join(
                f"#### {section_label(s.number, s.title)}\n{s.text}" for s in group
            )
        parent = Parent(
            id=parent_id,
            text=_clean_text(parent_text),
            header=make_header(first.chapter, first.chapter_title, p_number, p_title),
            chunk_type=TEXT,
            chapter=first.chapter,
            section=p_number or first.title,
            section_title=p_title,
            page_start=first.page_start,
            page_end=last.page_end,
        )
        parents[parent_id] = parent

        for sec in group:
            header = make_header(sec.chapter, sec.chapter_title, sec.number, sec.title)
            for piece in split_section(sec, splitter):
                chunk = Chunk(
                    id=f"{sec.id}-c{len(parent.child_ids)}",
                    text=_clean_text(piece["text"]),
                    header=header,
                    chunk_type=TEXT,
                    parent_id=parent_id,
                    chapter=sec.chapter,
                    section=sec.number or sec.title,
                    section_title=sec.title,
                    page_start=piece["page_start"],
                    page_end=piece["page_end"],
                    seq=len(parent.child_ids),
                )
                parent.child_ids.append(chunk.id)
                chunks.append(chunk)

    for table in doc.tables:
        table_chunks, table_parent = chunk_table(table, doc, chunk_size)
        chunks.extend(table_chunks)
        parents[table_parent.id] = table_parent

    return chunks, parents


def chunk_table(table: Table, doc: ParsedDocument, chunk_size: int) -> tuple[list[Chunk], Parent]:
    """A table is its own parent; very long tables are split by rows, repeating the header."""
    sec = doc.section_for_table(table)
    chapter = sec.chapter if sec else ""
    number = sec.number if sec else ""
    title = sec.title if sec else ""
    header = make_header(chapter, sec.chapter_title if sec else "", number, title)
    header += f" › {table.caption}"
    parent = Parent(
        id=f"parent-{table.id}",
        text=f"{table.caption}\n\n{table.markdown}",
        header=header,
        chunk_type=TABLE,
        chapter=chapter,
        section=number or title,
        section_title=title,
        page_start=table.page_start,
        page_end=table.page_end,
    )

    lines = table.markdown.splitlines()
    head, body = lines[:2], lines[2:]
    pieces: list[list[str]] = [[]]
    for row in body:
        if pieces[-1] and count_tokens("\n".join(head + pieces[-1] + [row])) > chunk_size:
            pieces.append([])
        pieces[-1].append(row)

    chunks = []
    for i, rows in enumerate(pieces):
        chunk = Chunk(
            id=f"{table.id}-c{i}",
            text=f"{table.caption}\n\n" + "\n".join(head + rows),
            header=header,
            chunk_type=TABLE,
            parent_id=parent.id,
            chapter=chapter,
            section=number or title,
            section_title=title,
            page_start=table.page_start,
            page_end=table.page_end,
            seq=i,
        )
        parent.child_ids.append(chunk.id)
        chunks.append(chunk)
    return chunks, parent


def make_summary_chunk(
    id: str, text: str, header: str, chapter: str, section: str, page_start: int, page_end: int
) -> tuple[Chunk, Parent]:
    """Summaries are short, so each is a single chunk that is also its own parent."""
    parent = Parent(
        id=f"parent-{id}",
        text=text,
        header=header,
        chunk_type=SUMMARY,
        chapter=chapter,
        section=section,
        page_start=page_start,
        page_end=page_end,
        child_ids=[id],
    )
    chunk = Chunk(
        id=id,
        text=text,
        header=header,
        chunk_type=SUMMARY,
        parent_id=parent.id,
        chapter=chapter,
        section=section,
        page_start=page_start,
        page_end=page_end,
    )
    return chunk, parent


def baseline_chunks(pdf_path: str) -> tuple[list[Chunk], dict[str, Parent]]:
    """Reproduce the v1 notebook's chunking for the evaluation baseline.

    ``PyPDFLoader.load_and_split()`` extracts plain text per page and applies
    LangChain's default ``RecursiveCharacterTextSplitter`` (4000 characters,
    200 overlap). No headings, no reference filtering, no tables, no parents.
    """
    import pymupdf

    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    chunks: list[Chunk] = []
    parents: dict[str, Parent] = {}
    with pymupdf.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf, start=1):
            for i, piece in enumerate(splitter.split_text(page.get_text())):
                cid = f"page{page_no}-c{i}"
                chunk = Chunk(
                    id=cid,
                    text=piece,
                    header="",
                    chunk_type=TEXT,
                    parent_id=f"parent-{cid}",
                    page_start=page_no,
                    page_end=page_no,
                )
                chunks.append(chunk)
                parents[chunk.parent_id] = Parent(
                    id=chunk.parent_id,
                    text=piece,
                    header="",
                    chunk_type=TEXT,
                    page_start=page_no,
                    page_end=page_no,
                    child_ids=[cid],
                )
    return chunks, parents
