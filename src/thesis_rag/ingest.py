"""PDF parsing: turn a thesis PDF into clean, structured sections and tables.

Pipeline
--------
1. ``pymupdf4llm`` converts each page to Markdown. Unlike plain text
   extraction, it keeps headings (``#``), so the chapter/section structure
   survives. OCR is disabled: the thesis has a text layer, and OCR on figure
   pages only produced noise.
2. Each page is cleaned: highlight/HTML tags are removed, Markdown tables are
   dropped (tables are extracted separately by pdfplumber, see step 5), and
   lines that repeat on many pages (running headers/footers) or are bare page
   numbers are stripped.
3. Headings are classified into chapters ("CHAPTER 3"), numbered sections
   ("3.10.2 Backend Development") or unnumbered top-level sections
   ("ABSTRACT"). Body lines are attached to the current section, keeping
   track of the page each line came from.
4. Everything from a "REFERENCES"/"BIBLIOGRAPHY" heading onward is excluded
   (until an appendix, if any): reference lists match almost any query
   lexically and crowd out real content. Table-of-contents style sections
   are excluded for the same reason.
5. ``pdfplumber`` extracts real tables (>= 2 columns), finds their
   "Table N ..." captions, and merges tables that continue across a page
   break. Each table later becomes its own chunk.

The output (:class:`ParsedDocument`) is plain data with no LLM involvement, so
it is deterministic and cheap to unit test.

Run ``python -m thesis_rag.ingest data/raw/thesis.pdf`` to parse, chunk,
summarise and index a PDF (see :func:`thesis_rag.index.build_index`).
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Section:
    """A leaf section of the document (the unit we use as a retrieval "parent")."""

    id: str
    chapter: str  # "3", or "" for front matter
    chapter_title: str  # "Research Methodology"
    number: str  # "3.10.2", or "" for unnumbered sections such as "Abstract"
    title: str  # "Backend Development"
    page_start: int
    page_end: int
    text: str = ""
    # (character offset into ``text``, page number) at every page change, so a
    # chunk cut from ``text`` can be mapped back to the pages it spans.
    page_offsets: list[tuple[int, int]] = field(default_factory=list)

    def pages_for_span(self, start: int, end: int) -> tuple[int, int]:
        """Pages covered by characters ``start:end`` of this section's text."""
        first = last = self.page_start
        for offset, page in self.page_offsets:
            if offset <= start:
                first = page
            if offset < end:
                last = page
        return first, max(first, last)


@dataclass
class Table:
    id: str
    caption: str
    markdown: str
    page_start: int
    page_end: int
    n_cols: int


@dataclass
class ParsedDocument:
    source: str
    file_hash: str
    title: str
    sections: list[Section]
    tables: list[Table]
    excluded_pages: list[int]  # pages dropped as references / table of contents
    # Every numbered heading's title, including headings with no body text of
    # their own (e.g. "1.6 Importance of Morus indica" directly followed by 1.6.1).
    heading_titles: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ParsedDocument:
        sections = [
            Section(**{**s, "page_offsets": [tuple(p) for p in s["page_offsets"]]})
            for s in d["sections"]
        ]
        tables = [Table(**t) for t in d["tables"]]
        return cls(**{**d, "sections": sections, "tables": tables})

    def section_for_table(self, table: Table) -> Section | None:
        """The section a table belongs to.

        Among sections that cover the table's page, prefer the one whose text
        refers to it ("... shown in Table 3"); otherwise take the first one.
        Falls back to the last section starting before the page.
        """
        m = re.match(r"table\s*[-:]?\s*(\d+)", table.caption, re.I)
        on_page = [s for s in self.sections if s.page_start <= table.page_start <= s.page_end]
        if m and on_page:
            mention = re.compile(rf"\btable\s*{m.group(1)}\b", re.I)
            for s in on_page:
                if mention.search(s.text):
                    return s
        if on_page:
            return on_page[0]
        before = [s for s in self.sections if s.page_start <= table.page_start]
        return before[-1] if before else None


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Line cleaning
# ---------------------------------------------------------------------------

_HTML_TAG = re.compile(r"</?(mark|u|b|i|em|strong|sup|sub|span)\b[^>]*>", re.I)
_BR = re.compile(r"<br\s*/?>", re.I)
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![\w*])_(.+?)_(?![\w*])")
_PAGE_NUMBER = re.compile(r"^\s*(page\s+)?\(?[0-9ivxlcdm]{1,6}\)?\s*$", re.I)


def clean_inline(text: str) -> str:
    """Remove markup noise from one line while keeping its words."""
    text = _BR.sub(" ", text)
    text = _HTML_TAG.sub("", text)
    text = _BOLD.sub(r"\1", text)
    text = _ITALIC.sub(r"\1", text)
    text = text.replace("**", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def _normalise_for_repeat(line: str) -> str:
    """Digits vary between pages ("Page 12 of 90"), so compare with them masked."""
    return re.sub(r"\d+", "#", clean_inline(line).lower())


def find_repeated_lines(pages: list[str], min_fraction: float = 0.4) -> set[str]:
    """Normalised lines that occur on a large share of pages: running headers/footers."""
    if len(pages) < 4:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        # Headings are never running headers; and with digits masked, "CHAPTER 1",
        # "CHAPTER 2", ... would otherwise look like one repeated line.
        lines = [ln for ln in page.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        counts.update({_normalise_for_repeat(line) for line in lines})
    threshold = max(4, int(len(pages) * min_fraction))
    return {line for line, n in counts.items() if n >= threshold and line}


def clean_page(markdown: str, repeated: set[str]) -> list[str]:
    """Return the useful lines of one page (headings kept, tables and noise dropped)."""
    out: list[str] = []
    for raw in markdown.splitlines():
        stripped = raw.strip()
        if not stripped:
            out.append("")
            continue
        if stripped.startswith("|"):  # Markdown table row: handled by pdfplumber instead
            continue
        if stripped.startswith(("==>", "<!--", "![")):  # image placeholders
            continue
        if _PAGE_NUMBER.match(clean_inline(stripped)):
            continue
        if _normalise_for_repeat(stripped) in repeated:
            continue
        out.append(raw.rstrip())
    return out


# ---------------------------------------------------------------------------
# Heading classification
# ---------------------------------------------------------------------------

# "CHAPTER 3" / "Chapter 5 CONCLUSION". Whitespace is required after the word so
# that summary lines such as "Chapter-1 Introduction" are not taken as chapters.
_CHAPTER = re.compile(r"^chapter\s+(\d+)\s*[:.\-–]?\s*(.*)$", re.I)
# Matches "3.10.2 Title", "2. 2 TITLE", "2.5Best practices", "...: 3.1 Whole genome".
_NUMBERED = re.compile(r"(?<![\d.])(\d{1,2})\s?\.\s?(\d{1,2}(?:\.\d{1,2})*)\.?\s*([A-Za-z(].*)$")
_REFERENCES = re.compile(r"^(references|bibliography|works cited|literature cited)$", re.I)
_APPENDIX = re.compile(r"^(appendix|appendices|annexure)\b", re.I)
_TOC_LIKE = re.compile(
    r"^(table of contents|contents|list of (figures|tables|abbreviations|symbols))$", re.I
)


_SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to"}


def title_case(text: str) -> str:
    """'LEAF YIELD OF MULBERRY' -> 'Leaf Yield of Mulberry'; mixed case is left alone."""
    text = text.strip(" :.-–")
    if not text.isupper():
        return text
    words = text.lower().split()
    return " ".join(
        w if (i and w in _SMALL_WORDS) else w[:1].upper() + w[1:] for i, w in enumerate(words)
    )


@dataclass
class Heading:
    kind: str  # "chapter" | "numbered" | "unnumbered" | "inline"
    text: str
    chapter: str = ""
    number: str = ""


def classify_heading(line: str, current_chapter: str) -> Heading | None:
    """Classify a Markdown heading line, or return None if ``line`` is not a heading."""
    if not line.lstrip().startswith("#"):
        return None
    text = clean_inline(line.lstrip().lstrip("#"))
    if not text:
        return None

    m = _CHAPTER.match(text)
    # Chapter numbers only go up; a lower number is a cross-reference, not a chapter.
    if m and int(m.group(1)) > int(current_chapter or 0):
        return Heading("chapter", title_case(m.group(2)), chapter=m.group(1))

    m = _NUMBERED.search(text)
    # The first number must match the current chapter; this rejects things
    # like a stray "Figure 4.2" heading appearing inside chapter 3.
    if m and (not current_chapter or m.group(1) == current_chapter):
        number = f"{m.group(1)}.{m.group(2)}"
        return Heading("numbered", title_case(m.group(3)), chapter=m.group(1), number=number)

    # Unnumbered headings: ALL-CAPS ones ("ABSTRACT", "OBJECTIVES OF THE STUDY")
    # start a new section; mixed-case ones ("Architecture", "Table 1 ...") are
    # sub-headings and stay inline in the current section's text.
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 4 and text.isupper():
        return Heading("unnumbered", title_case(text))
    return Heading("inline", text)


# ---------------------------------------------------------------------------
# Sectioning
# ---------------------------------------------------------------------------


class _SectionBuilder:
    """Accumulates (line, page) pairs into Section objects."""

    def __init__(self) -> None:
        self.sections: list[Section] = []
        self.chapter = ""
        self.chapter_title = ""
        self._current: Section | None = None
        self._buf: list[str] = []
        self._offset = 0
        self._last_page = -1

    def start(self, number: str, title: str, page: int) -> None:
        self.flush()
        sid = f"ch{self.chapter or 0}-{number or _slug(title)}-p{page}"
        self._current = Section(
            id=sid,
            chapter=self.chapter,
            chapter_title=self.chapter_title,
            number=number,
            title=title,
            page_start=page,
            page_end=page,
        )
        self._buf, self._offset, self._last_page = [], 0, -1

    def add(self, line: str, page: int) -> None:
        if self._current is None:  # text before the first heading
            self.start("", "Front Matter", page)
        assert self._current is not None
        if line.strip() and page != self._last_page:
            self._current.page_offsets.append((self._offset, page))
            self._last_page = page
            self._current.page_end = page
        self._buf.append(line)
        self._offset += len(line) + 1

    def flush(self) -> None:
        sec = self._current
        if sec is not None:
            # Joining with "\n" keeps page_offsets valid; blank runs are left as-is
            # rather than collapsed for the same reason.
            sec.text = "\n".join(self._buf).rstrip()
            if _has_content(sec):
                self.sections.append(sec)
        self._current = None


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "section"


def _has_content(section: Section, min_words: int = 8) -> bool:
    return len(re.findall(r"[A-Za-z]{2,}", section.text)) >= min_words


def _is_chapter_overview(lines: list[str]) -> bool:
    """A page with several "Chapter N" headings is a chapter-by-chapter overview,
    not the start of a chapter; its headings must not change the current chapter."""
    n = sum(
        1
        for line in lines
        if line.lstrip().startswith("#")
        and re.match(r"chapter\b", clean_inline(line.lstrip("# ")), re.I)
    )
    return n >= 3


def build_sections(
    pages: list[list[str]],
) -> tuple[list[Section], list[int], dict[str, str]]:
    """Group cleaned page lines into sections.

    Returns the sections, the 1-based page numbers that were excluded
    (references, table of contents, lists of figures/tables), and a map of
    section number -> heading title.
    """
    b = _SectionBuilder()
    titles: dict[str, str] = {}
    skip: str | None = None  # None, "references" or "toc"
    excluded: set[int] = set()
    awaiting_chapter_title = False

    for page_no, lines in enumerate(pages, start=1):
        overview = _is_chapter_overview(lines)
        for line in lines:
            h = classify_heading(line, b.chapter)
            if overview and h is not None and h.kind == "chapter":
                h = Heading("inline", clean_inline(line.lstrip().lstrip("#")))

            # --- body text (or an inline sub-heading kept as part of the text) ---
            if h is None or h.kind == "inline":
                if skip:
                    if line.strip():
                        excluded.add(page_no)
                    continue
                b.add(clean_inline(line) if h is None else f"#### {h.text}", page_no)
                continue

            # --- structural headings ---
            if h.kind == "chapter":
                b.flush()
                b.chapter, b.chapter_title = h.chapter, h.text
                skip = None
                awaiting_chapter_title = not h.text
                if awaiting_chapter_title:
                    continue  # title is on the next heading, e.g. "CHAPTER 2" / "REVIEW..."
                h = Heading("unnumbered", h.text)
            elif awaiting_chapter_title:
                awaiting_chapter_title = False
                # "REVIEW OF LITERATURE" or "1.0 INTRODUCTION" names the chapter.
                if h.kind == "unnumbered" or h.number.endswith(".0"):
                    b.chapter_title = h.text

            if _REFERENCES.match(h.text):
                b.flush()
                skip = "references"
                excluded.add(page_no)
                continue
            if _TOC_LIKE.match(h.text):
                b.flush()
                skip = "toc"
                excluded.add(page_no)
                continue
            if skip == "references" and not _APPENDIX.match(h.text):
                excluded.add(page_no)
                continue
            skip = None  # any other heading ends a TOC; an appendix ends references
            if h.number:
                titles.setdefault(h.number, h.text)
            b.start(h.number, h.text, page_no)

    b.flush()
    return b.sections, sorted(excluded), titles


# ---------------------------------------------------------------------------
# Tables (pdfplumber)
# ---------------------------------------------------------------------------

_CAPTION = re.compile(r"^\s*table\s*[-:]?\s*(\d+)\b", re.I)
_PAGE_REF = re.compile(r"^([0-9]+|[ivxlcdm]+)$", re.I)


def _clean_cell(cell: str | None) -> str:
    return re.sub(r"\s+", " ", cell or "").replace("|", "/").strip()


def is_real_table(rows: list[list[str]]) -> bool:
    """Reject text boxes (1 column), mostly-empty grids and table-of-contents tables."""
    if len(rows) < 2 or max(len(r) for r in rows) < 2:
        return False
    cells = [c for r in rows for c in r]
    if sum(1 for c in cells if c) < 0.5 * len(cells):
        return False
    last = [next((c for c in reversed(r) if c), "") for r in rows]
    page_refs = sum(1 for c in last if _PAGE_REF.match(c))
    return page_refs < 0.6 * len(rows)


def rows_to_markdown(rows: list[list[str]]) -> str:
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _caption_near(page, bbox, max_gap: float = 70.0) -> str:
    """Find a 'Table N ...' line just above (preferred) or just below a table."""
    rows: dict[int, list[dict]] = {}
    for w in page.extract_words():
        rows.setdefault(round(w["top"] / 3), []).append(w)
    above, below = [], []
    for words in rows.values():
        words.sort(key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in words)
        if not _CAPTION.match(text):
            continue
        top, bottom = words[0]["top"], words[0]["bottom"]
        if 0 <= bbox[1] - bottom <= max_gap:
            above.append((bbox[1] - bottom, text))
        elif 0 <= top - bbox[3] <= max_gap:
            below.append((top - bbox[3], text))
    for candidates in (above, below):
        if candidates:
            return clean_inline(min(candidates)[1])
    return ""


def extract_tables(pdf_path: str | Path, skip_pages: set[int]) -> list[Table]:
    import pdfplumber

    tables: list[Table] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            if page_no in skip_pages:
                continue
            for found in page.find_tables():
                rows = [[_clean_cell(c) for c in r] for r in found.extract()]
                if not is_real_table(rows):
                    continue
                caption = _caption_near(page, found.bbox)
                prev = tables[-1] if tables else None
                # A caption-less table at the top of a page with the same column
                # count as a table ending on the previous page is its continuation.
                if (
                    prev is not None
                    and not caption
                    and prev.page_end == page_no - 1
                    and prev.n_cols == len(rows[0])
                    and found.bbox[1] < page.height * 0.25
                ):
                    prev.markdown += "\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows)
                    prev.page_end = page_no
                    continue
                tables.append(
                    Table(
                        id=f"table-p{page_no}-{len(tables)}",
                        caption=caption or f"Table on page {page_no}",
                        markdown=rows_to_markdown(rows),
                        page_start=page_no,
                        page_end=page_no,
                        n_cols=len(rows[0]),
                    )
                )
    return tables


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def pdf_to_markdown_pages(pdf_path: str | Path) -> list[str]:
    import pymupdf4llm

    chunks = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
        use_ocr=False,
        header=False,  # layout mode: drop detected running headers/footers
        footer=False,
        show_progress=False,
    )
    return [c["text"] for c in chunks]


def parse_markdown_pages(
    md_pages: list[str],
) -> tuple[list[Section], list[int], dict[str, str]]:
    """The pure-text half of parsing; separated from PDF I/O so it is unit-testable."""
    repeated = find_repeated_lines(md_pages)
    cleaned = [clean_page(p, repeated) for p in md_pages]
    return build_sections(cleaned)


_ACRONYM = re.compile(
    r"\b(snp|snps|ssr|ssrs|dna|rna|pcr|qtl|gwas|gatk|bwa|nue|ngs)\b", re.IGNORECASE
)


def _upper_acronym(word: str) -> str:
    return word[:-1].upper() + "s" if word.lower().endswith("s") and len(word) > 3 else word.upper()


def guess_title(md_pages: list[str]) -> str:
    """Thesis title pages usually quote the title; return it if found."""
    if not md_pages:
        return ""
    first = clean_inline(" ".join(md_pages[0].split()))
    m = re.search(r"[“\"](.{15,250}?)[”\"]", first)
    if not m:
        return ""
    # Title-casing an all-caps title turns "SNP" into "Snp"; put known acronyms back.
    return _ACRONYM.sub(lambda a: _upper_acronym(a.group(0)), title_case(m.group(1)))


def parse_pdf(pdf_path: str | Path) -> ParsedDocument:
    pdf_path = Path(pdf_path)
    md_pages = pdf_to_markdown_pages(pdf_path)
    sections, excluded, titles = parse_markdown_pages(md_pages)
    tables = extract_tables(pdf_path, skip_pages=set(excluded))
    return ParsedDocument(
        source=pdf_path.name,
        file_hash=file_sha256(pdf_path),
        title=guess_title(md_pages) or pdf_path.stem,
        sections=sections,
        tables=tables,
        excluded_pages=excluded,
        heading_titles=titles,
    )


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json
    import logging

    from .config import get_settings
    from .index import build_index

    parser = argparse.ArgumentParser(description="Parse, chunk, summarise and index a PDF.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    parser.add_argument("--no-summaries", action="store_true", help="skip LLM summaries")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    if args.no_summaries:
        settings = settings.model_copy(update={"summarize_sections": False})
    manifest = build_index(args.pdf, settings, force=args.force)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
