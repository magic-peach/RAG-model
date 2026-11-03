from thesis_rag.ingest import (
    classify_heading,
    clean_inline,
    clean_page,
    find_repeated_lines,
    is_real_table,
    title_case,
)

from .conftest import RUNNING_HEADER


def test_clean_inline_strips_markup():
    assert (
        clean_inline("<mark>Whole</mark> **genome** _Morus indica_") == "Whole genome Morus indica"
    )


def test_running_headers_and_page_numbers_removed(md_pages):
    repeated = find_repeated_lines(md_pages)
    for page in md_pages:
        lines = [line for line in clean_page(page, repeated) if line.strip()]
        assert RUNNING_HEADER not in lines
        assert not any(line.strip().isdigit() for line in lines)


def test_classify_heading_variants():
    assert classify_heading("## **CHAPTER 2**", "1").kind == "chapter"
    h = classify_heading("#### **2. 2 EXPLORING POLYPLOID GENOMES**", "2")
    assert (h.kind, h.number, h.text) == ("numbered", "2.2", "Exploring Polyploid Genomes")
    assert classify_heading("#### **2.5Best practices**", "2").number == "2.5"
    h = classify_heading("### **OBJECTIVE 1: Database 3.1 Whole genome sequencing**", "3")
    assert (h.number, h.text) == ("3.1", "Whole genome sequencing")
    assert classify_heading("#### **Architecture**", "3").kind == "inline"
    assert classify_heading("## **ABSTRACT**", "").kind == "unnumbered"
    # A section number from another chapter is not a section heading here.
    assert classify_heading("#### Figure 4.2 Something", "3").kind == "inline"
    # Chapter numbers never go backwards.
    assert classify_heading("## CHAPTER 1", "3").kind != "chapter"
    assert classify_heading("plain text line", "1") is None


def test_title_case_keeps_small_words_lowercase():
    assert title_case("IMPORTANCE OF FEED IN SERICULTURE") == "Importance of Feed in Sericulture"
    assert title_case("Mixed case Title") == "Mixed case Title"


def test_sections_structure(parsed):
    labels = [(s.chapter, s.number, s.title) for s in parsed.sections]
    assert ("", "", "Abstract") in labels
    assert ("1", "1.1", "Drought Stress") in labels
    assert ("2", "2.2", "Variant calling") in labels
    # Chapter titles come from the heading after "CHAPTER N".
    ch2 = [s for s in parsed.sections if s.chapter == "2"]
    assert {s.chapter_title for s in ch2} == {"Methods"}


def test_chapter_overview_page_does_not_create_chapters(parsed):
    overview = [s for s in parsed.sections if s.title == "Chapter-wise Explanation"]
    assert overview and overview[0].chapter == ""
    assert "Chapter 1- Introduction" in overview[0].text


def test_references_and_toc_excluded(parsed):
    all_text = " ".join(s.text for s in parsed.sections)
    assert "Smith, J." not in all_text and "Doe, A." not in all_text
    assert "Figure 2 Other" not in all_text
    assert {3, 7, 8} <= set(parsed.excluded_pages)


def test_section_page_ranges(parsed):
    s11 = next(s for s in parsed.sections if s.number == "1.1")
    assert (s11.page_start, s11.page_end) == (4, 5)
    s12 = next(s for s in parsed.sections if s.number == "1.2")
    assert (s12.page_start, s12.page_end) == (5, 5)


def test_inline_subheading_stays_in_section(parsed):
    s = next(s for s in parsed.sections if s.number == "2.1.1")
    assert "#### Architecture" in s.text


def test_heading_titles_recorded(parsed):
    assert parsed.heading_titles["1.1"] == "Drought Stress"


def test_is_real_table():
    assert is_real_table([["Genotype", "Drought"], ["K2", "Medium"]])
    assert not is_real_table([["just a text box"], ["more text"]])
    toc = [["1.1", "Taxonomy", "16"], ["1.2", "Intro", "17"], ["1.3", "Feed", "18"]]
    assert not is_real_table(toc)
