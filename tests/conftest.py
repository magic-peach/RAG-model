"""Shared fixtures: a small synthetic thesis, as pymupdf4llm-style Markdown pages."""

import pytest

from thesis_rag.ingest import ParsedDocument, Table, parse_markdown_pages

RUNNING_HEADER = "A Study of Mulberry Genomics"

LOREM = (
    "Mulberry leaves feed silkworms and drought reduces leaf yield. "
    "Genotypes differ in nitrogen use efficiency and root rot resistance. "
)


def _page(n: int, body: str) -> str:
    # Every page has a running header and a page number, like a real thesis.
    return f"{RUNNING_HEADER}\n\n{body}\n\n{n}\n"


@pytest.fixture
def md_pages() -> list[str]:
    long_text = " ".join(f"Sentence {i} about variant calling with GATK." for i in range(300))
    return [
        _page(
            1,
            "## **ABSTRACT**\n\nThis thesis studies <mark>SNP markers</mark> in mulberry " + LOREM,
        ),
        _page(
            2,
            "## **CHAPTER-WISE EXPLANATION**\n\n#### **Chapter 1- Introduction**\nIntro.\n"
            "#### **Chapter 2- Methods**\nMethods.\n#### **Chapter 3- References**\nRefs.",
        ),
        _page(3, "## **LIST OF FIGURES**\n\nFigure 1 Something 12\nFigure 2 Other 13"),
        _page(
            4,
            "## **CHAPTER 1**\n\n## **1.0 INTRODUCTION**\n\n" + LOREM * 3 + "\n\n"
            "### **1.1 DROUGHT STRESS**\n\n" + LOREM * 4,
        ),
        _page(5, LOREM * 2 + "\n\n### **1.2 LEAF YIELD**\n\n" + LOREM * 2),
        _page(
            6,
            "## **CHAPTER 2**\n\n## **METHODS**\n\n#### **2.1 Small part one**\n\n"
            + LOREM
            + "\n\n#### **2.1.1 Tiny**\n\n"
            + LOREM
            + "\n\n#### **Architecture**\n\nInline sub."
            + "\n\n#### **2.2 Variant calling**\n\n"
            + long_text,
        ),
        _page(7, "## **CHAPTER 3**\n\n## **REFERENCES**\n\n1) Smith, J. (2020). A paper."),
        _page(8, "2) Doe, A. (2021). Another paper about drought and mulberry."),
    ]


@pytest.fixture
def parsed(md_pages) -> ParsedDocument:
    sections, excluded, titles = parse_markdown_pages(md_pages)
    table = Table(
        id="table-p5-0",
        caption="Table 1 Genotype traits",
        markdown="| Genotype | Drought |\n|---|---|\n| K2 | Medium |\n| V1 | High |",
        page_start=5,
        page_end=5,
        n_cols=2,
    )
    return ParsedDocument(
        source="synthetic.pdf",
        file_hash="0" * 64,
        title="Synthetic thesis",
        sections=sections,
        tables=[table],
        excluded_pages=excluded,
        heading_titles=titles,
    )
