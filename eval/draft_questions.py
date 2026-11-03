"""Draft evaluation questions from the indexed thesis with the local LLM.

    python eval/draft_questions.py            # writes 10 drafts into eval/questions.jsonl

Each draft is written from one passage (8 text sections spread across the
chapters, plus 2 tables), so its expected pages are known. Drafts are marked
``"verified": false``: an 8B model's questions and reference answers must be
checked by someone who knows the thesis before the scores are trusted.
Hand-written entries in questions.jsonl (``"source": "manual"``) are kept.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# run_eval puts src/ on sys.path, so it is imported before thesis_rag.
from run_eval import QUESTIONS_PATH, CachedChat, load_questions, write_questions

from thesis_rag.chunking import TABLE, TEXT, count_tokens
from thesis_rag.config import get_settings
from thesis_rag.generate import get_llm
from thesis_rag.index import Index

DRAFT_SYSTEM = (
    "You write evaluation questions for a question-answering system over a PhD thesis "
    "on mulberry genomics. Given one passage, write ONE factual question that the "
    "passage answers, and the answer. The question must make sense on its own: do not "
    "say 'in this passage' or 'according to the text'. Prefer questions about specific "
    "methods, tools, numbers, genotypes or findings. The answer must be 1-2 sentences "
    "using names and numbers exactly as written in the passage. Reply with JSON only: "
    '{"question": "...", "answer": "..."}'
)


def pick_passages(index: Index, n_text: int = 8, n_table: int = 2):
    """Deterministic spread: evenly spaced eligible sections across chapters 1-5."""
    parents = sorted(index.parents.values(), key=lambda p: (int(p.chapter or 0), p.page_start))
    texts = [
        p
        for p in parents
        if p.chunk_type == TEXT and p.chapter and 250 <= count_tokens(p.text) <= 1200
    ]
    tables = [p for p in parents if p.chunk_type == TABLE]
    step = max(len(texts) / n_text, 1)
    chosen = [texts[int(i * step)] for i in range(min(n_text, len(texts)))]
    return chosen + tables[:n_table]


def parse_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    return json.loads(match.group(0)) if match else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-text", type=int, default=8)
    parser.add_argument("--n-table", type=int, default=2)
    args = parser.parse_args()

    settings = get_settings()
    index = Index(settings)
    llm = CachedChat(get_llm(settings, temperature=0.2, num_predict=200, format="json"), "draft")
    drafts = []
    for i, p in enumerate(pick_passages(index, args.n_text, args.n_table), start=1):
        print(f"[{i}] {p.header}", flush=True)
        data = parse_json(llm.complete(DRAFT_SYSTEM, f"Passage ({p.header}):\n\n{p.text}"))
        if not data.get("question"):
            print("    skipped: no question in model output")
            continue
        drafts.append(
            {
                "id": f"d{i:02d}",
                "question": data["question"].strip(),
                "expected_answer": str(data.get("answer", "")).strip(),
                "expected_pages": list(range(p.page_start, p.page_end + 1)),
                "source": "generated",
                "source_header": p.header,
                "verified": False,
            }
        )
        print(f"    Q: {drafts[-1]['question']}")

    keep = [q for q in load_questions(include_placeholders=True) if q["source"] != "generated"]
    write_questions(keep + drafts, Path(QUESTIONS_PATH))
    print(f"Wrote {len(drafts)} drafts and kept {len(keep)} manual entries in {QUESTIONS_PATH}")


if __name__ == "__main__":
    main()
