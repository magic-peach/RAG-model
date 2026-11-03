"""Evaluate retrieval and answers across pipeline configurations.

    python eval/run_eval.py                       # all configs, retrieval + LLM judge
    python eval/run_eval.py --retrieval-only      # hit@5 and MRR only (minutes, no LLM)
    python eval/run_eval.py --configs baseline,reranker --limit 5

Configurations are cumulative, so each row shows what one change adds:

    baseline     v1 notebook: 4000-char page chunks, dense search only
    structured   + section-aware chunks, headers, parents, summary routing
    hybrid       + BM25 fused with dense by RRF
    reranker     + bge-reranker-base cross-encoder (the default config)
    multi_query  + 3 LLM rephrasings per question

Every configuration uses the same answer prompt, so differences come from
retrieval, not prompting.

Metrics
-------
hit@5   share of questions where at least one of the 5 passages overlaps the
        expected pages. Summary passages never count: the whole-thesis summary
        spans pp.16-85 and would "hit" every question.
MRR@5   mean of 1/rank of the first passage that hits (0 if none).
Faithfulness, relevance
        1-5 scores from an LLM judge (prompts in eval/prompts/). The judge is
        the same local llama3.1:8b, which is a known weakness (a model grading
        itself is lenient); treat these as relative, not absolute, numbers.
Cited  share of answers (excluding "Not found") with at least one citation tag
        that matches a retrieved passage.
"Not found"
        share of answers that were exactly "Not found in the thesis.".

Every LLM call (answers, rephrasings, judgments) is cached in
data/cache/eval_llm.json keyed by its full input, so re-running after a
change only pays for the calls whose input changed.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from thesis_rag.chunking import SUMMARY  # noqa: E402
from thesis_rag.config import Settings, get_settings  # noqa: E402
from thesis_rag.generate import ThesisQA, format_context, get_llm  # noqa: E402
from thesis_rag.index import (  # noqa: E402
    Embedder,
    Index,
    SummaryCache,
    build_index,
    with_retry,
)
from thesis_rag.retrieve import Retriever, get_reranker  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = EVAL_DIR / "questions.jsonl"
PROMPTS_DIR = EVAL_DIR / "prompts"
RESULTS_PATH = EVAL_DIR / "results.md"
K = 5

CONFIGS: dict[str, dict] = {
    "baseline": {
        "structured_chunking": False,
        "summarize_sections": False,
        "use_bm25": False,
        "use_reranker": False,
        "use_multi_query": False,
        "use_parent_context": False,
    },
    "structured": {"use_bm25": False, "use_reranker": False, "use_multi_query": False},
    "hybrid": {"use_bm25": True, "use_reranker": False, "use_multi_query": False},
    "reranker": {"use_bm25": True, "use_reranker": True, "use_multi_query": False},
    "multi_query": {"use_bm25": True, "use_reranker": True, "use_multi_query": True},
}
DESCRIPTIONS = {
    "baseline": "v1: page chunks, dense only",
    "structured": "+ structured chunking",
    "hybrid": "+ BM25 + RRF",
    "reranker": "+ cross-encoder rerank",
    "multi_query": "+ multi-query",
}


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------


def load_questions(path: Path = QUESTIONS_PATH, include_placeholders: bool = False) -> list[dict]:
    """Read questions.jsonl; placeholders (empty question) are skipped unless asked."""
    rows = []
    for line in path.read_text().splitlines() if path.is_file() else []:
        if not line.strip():
            continue
        row = json.loads(line)
        row.setdefault("source", "manual")
        row.setdefault("verified", False)
        if include_placeholders or (row.get("question") and row.get("expected_pages")):
            rows.append(row)
    return rows


def write_questions(rows: list[dict], path: Path = QUESTIONS_PATH) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def passage_hits(passage, expected_pages: set[int]) -> bool:
    if passage.chunk_type == SUMMARY:
        return False
    return any(p in expected_pages for p in range(passage.page_start, passage.page_end + 1))


def first_hit_rank(passages, expected_pages, k: int = K) -> int | None:
    """1-based rank of the first passage overlapping the expected pages, or None."""
    expected = set(expected_pages)
    for rank, p in enumerate(passages[:k], start=1):
        if passage_hits(p, expected):
            return rank
    return None


def hit_at_k(rank: int | None) -> float:
    return 1.0 if rank is not None else 0.0


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


# ---------------------------------------------------------------------------
# Cached LLM calls
# ---------------------------------------------------------------------------


class CachedChat:
    """Wraps a chat model so identical prompts are answered from a JSON cache.

    Implements just ``invoke`` (all ThesisQA.answer and query expansion use),
    plus ``complete`` for plain system/user prompts.
    """

    def __init__(self, llm, tag: str, cache: SummaryCache | None = None):
        self.llm = llm
        self.tag = tag
        self.cache = cache or SummaryCache(get_settings().data_dir / "cache" / "eval_llm.json")

    def invoke(self, messages, **_):
        model = str(getattr(self.llm, "model", ""))
        key = self.cache.key(self.tag, model, *(str(m.content) for m in messages))
        text = self.cache.get_or_make(
            key, lambda: str(with_retry(lambda: self.llm.invoke(messages)).content)
        )
        return AIMessage(content=text)

    def complete(self, system: str, user: str) -> str:
        msgs = [SystemMessage(content=system), HumanMessage(content=user)]
        return str(self.invoke(msgs).content).strip()


def parse_score(raw: str) -> int | None:
    """Pull a 1-5 score out of the judge's reply (JSON, or a bare number as fallback)."""
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        score = int(json.loads(match.group(0))["score"]) if match else None
    except (ValueError, KeyError, TypeError):
        score = None
    if score is None:
        m = re.search(r"\b([1-5])\b", raw)
        score = int(m.group(1)) if m else None
    return score if score in range(1, 6) else None


def judge(llm: CachedChat, prompt_name: str, **fields) -> int | None:
    template = (PROMPTS_DIR / f"{prompt_name}.txt").read_text()
    prompt = template.format(**fields)
    return parse_score(llm.complete("You are a strict, careful grader.", prompt))


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def run_config(name, base: Settings, questions, pdf: Path, generate: bool, shared: dict):
    settings = Settings(**{**base.model_dump(), **CONFIGS[name]})
    build_index(pdf, settings)  # no-op when the index for this chunking already exists
    index = Index(settings, embedder=shared["embedder"])
    retriever = Retriever(
        settings, index, reranker=shared["reranker"], llm=CachedChat(shared["expand"], "expand")
    )
    qa = ThesisQA(settings, index, retriever, llm=CachedChat(shared["answer"], "answer"))
    judge_llm = CachedChat(shared["judge"], "judge")

    rows = []
    for i, q in enumerate(questions, start=1):
        result = retriever.retrieve(q["question"])
        rank = first_hit_rank(result.passages, q["expected_pages"])
        row = {
            "id": q["id"],
            "rank": rank,
            "hit": hit_at_k(rank),
            "rr": reciprocal_rank(rank),
            "retrieval_s": result.timings.get("total", 0) - result.timings.get("expand", 0),
            "pages": [[p.page_start, p.page_end] for p in result.passages],
        }
        if generate:
            ans = qa.answer(q["question"], result)
            row["answer"] = ans.answer
            row["not_found"] = ans.not_found
            row["cited"] = ans.cited
            row["faithfulness"] = (
                None  # "Not found" makes no claims, so there is nothing to check
                if ans.not_found
                else judge(
                    judge_llm,
                    "faithfulness",
                    context=format_context(result.passages),
                    question=q["question"],
                    answer=ans.answer,
                )
            )
            row["relevance"] = judge(
                judge_llm,
                "relevance",
                expected_answer=q.get("expected_answer") or "(none provided)",
                question=q["question"],
                answer=ans.answer,
            )
        rows.append(row)
        status = f"hit@{rank}" if rank else "miss"
        if generate:
            status += f" F={row['faithfulness']} R={row['relevance']}"
        print(f"  [{name}] {i}/{len(questions)} {q['id']}: {status}", flush=True)
    return rows


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def summarize(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "hit@5": _mean(r["hit"] for r in rows),
        "mrr@5": _mean(r["rr"] for r in rows),
        "faithfulness": _mean(r.get("faithfulness") for r in rows),
        "relevance": _mean(r.get("relevance") for r in rows),
        "not_found": _mean(float(r["not_found"]) for r in rows if "not_found" in r),
        # Of the answers that claim something, how many carry a valid citation tag.
        "cited": _mean(float(bool(r["cited"])) for r in rows if r.get("not_found") is False),
        "retrieval_s": statistics.median(r["retrieval_s"] for r in rows) if rows else None,
    }


def _fmt(value, kind: str) -> str:
    if value is None:
        return "–"
    if kind == "pct":
        return f"{value:.0%}"
    if kind == "score":
        return f"{value:.2f} / 5"
    if kind == "s":
        return f"{value:.1f}"
    return f"{value:.3f}"


def render_markdown(results: dict, questions: list[dict], meta: dict) -> str:
    verified = sum(1 for q in questions if q.get("verified"))
    lines = [
        "# Evaluation results",
        "",
        f"Generated by `python eval/run_eval.py` on {meta['date']}"
        f"{' (retrieval only)' if meta['retrieval_only'] else ''}.",
        "",
        f"- Questions: {len(questions)} ({verified} verified, "
        f"{len(questions) - verified} unverified LLM drafts)",
        f"- Generator and judge: `{meta['llm']}` via Ollama; embeddings `{meta['embedding']}`",
        f"- Retrieval: top {K} passages; hit = a non-summary passage overlaps the expected pages",
        "",
        "| Config | Change | hit@5 | MRR@5 | Faithfulness | Relevance | Cited | Not found "
        "| Retrieval s (median) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, res in results.items():
        s = res["summary"]
        lines.append(
            f"| `{name}` | {DESCRIPTIONS[name]} | {_fmt(s['hit@5'], 'pct')} | "
            f"{_fmt(s['mrr@5'], 'f')} | {_fmt(s['faithfulness'], 'score')} | "
            f"{_fmt(s['relevance'], 'score')} | {_fmt(s['cited'], 'pct')} | "
            f"{_fmt(s['not_found'], 'pct')} | "
            f"{_fmt(s['retrieval_s'], 's')} |"
        )
    lines += ["", "## Per-question rank of the first hit", ""]
    names = list(results)
    lines.append("| Question | " + " | ".join(f"`{n}`" for n in names) + " |")
    lines.append("|---|" + "---|" * len(names))
    by_id = {n: {r["id"]: r for r in results[n]["rows"]} for n in names}
    for q in questions:
        cells = []
        for n in names:
            rank = by_id[n].get(q["id"], {}).get("rank")
            cells.append(str(rank) if rank else "✗")
        tag = "" if q.get("verified") else " *(draft)*"
        lines.append(f"| {q['id']}: {q['question']}{tag} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Reading these numbers",
        "",
        "- Draft questions were written by the same model from a single passage, which "
        "favours retrieval of that passage; verified hand-written questions are the real test.",
        "- The judge is the generator itself (no paid APIs), so faithfulness and relevance "
        "are lenient; compare rows with each other rather than reading them as absolutes.",
        "- With this few questions, one question is "
        f"{1 / max(len(questions), 1):.0%} of hit@5; small differences are noise.",
        "- Retrieval time was measured on a heavily loaded laptop (CPU reranker); multi-query "
        "time excludes its extra LLM call.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--pdf", type=Path, default=ROOT / "data" / "raw" / "thesis.pdf")
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--limit", type=int, default=0, help="only the first N questions")
    parser.add_argument("--verified-only", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true", help="skip answers and judge")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    questions = load_questions()
    if args.verified_only:
        questions = [q for q in questions if q.get("verified")]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        sys.exit("No usable questions in eval/questions.jsonl (fill a placeholder or draft some).")

    base = get_settings()
    shared = {
        "embedder": Embedder(base),
        "reranker": get_reranker(base.reranker_model),
        "expand": get_llm(base, temperature=0.3, num_predict=200),
        "answer": get_llm(base, num_predict=700),
        "judge": get_llm(base, temperature=0.0, num_predict=150, format="json"),
    }
    generate = not args.retrieval_only
    results = {}
    for name in [c.strip() for c in args.configs.split(",") if c.strip()]:
        if name not in CONFIGS:
            sys.exit(f"Unknown config {name!r}; choose from {', '.join(CONFIGS)}")
        print(f"Config {name}: {DESCRIPTIONS[name]}", flush=True)
        rows = run_config(name, base, questions, args.pdf, generate, shared)
        results[name] = {"summary": summarize(rows), "rows": rows}

    meta = {
        "date": time.strftime("%Y-%m-%d"),
        "retrieval_only": args.retrieval_only,
        "llm": base.llm_model,
        "embedding": base.embedding_model,
    }
    args.output.write_text(render_markdown(results, questions, meta))
    # Full per-question output (answers included) stays out of git, next to the caches.
    raw = base.data_dir / "cache" / f"eval-{time.strftime('%Y%m%d-%H%M%S')}.json"
    raw.write_text(json.dumps({"meta": meta, "results": results}, indent=1, ensure_ascii=False))
    print(f"Wrote {args.output} and {raw}")


if __name__ == "__main__":
    main()
