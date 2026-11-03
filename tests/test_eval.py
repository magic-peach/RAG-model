"""Metric and parsing helpers of eval/run_eval.py (no LLM, no index)."""

import json
from dataclasses import dataclass

import pytest
from run_eval import (
    CachedChat,
    first_hit_rank,
    hit_at_k,
    load_questions,
    parse_score,
    reciprocal_rank,
    summarize,
)

from thesis_rag.index import SummaryCache

from .fakes import fake_llm


@dataclass
class P:
    page_start: int
    page_end: int
    chunk_type: str = "text"


def test_first_hit_rank_and_metrics():
    passages = [P(10, 11), P(20, 22), P(30, 30)]
    assert first_hit_rank(passages, [21]) == 2
    assert first_hit_rank(passages, [99]) is None
    assert hit_at_k(2) == 1.0 and hit_at_k(None) == 0.0
    assert reciprocal_rank(2) == 0.5 and reciprocal_rank(None) == 0.0


def test_summaries_never_count_as_hits():
    passages = [P(16, 85, "summary"), P(40, 40)]
    assert first_hit_rank(passages, [40]) == 2


def test_rank_is_limited_to_k():
    passages = [P(i, i) for i in range(1, 8)]
    assert first_hit_rank(passages, [7], k=5) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"score": 4, "reason": "ok"}', 4),
        ('Sure! {"score": "5", "reason": "x"}', 5),
        ("Score: 3", 3),
        ('{"score": 9}', None),
        ("no idea", None),
    ],
)
def test_parse_score(raw, expected):
    assert parse_score(raw) == expected


def test_summarize_skips_missing_scores():
    rows = [
        {"hit": 1.0, "rr": 1.0, "retrieval_s": 1.0, "faithfulness": None, "not_found": True},
        {"hit": 0.0, "rr": 0.0, "retrieval_s": 3.0, "faithfulness": 4, "not_found": False},
    ]
    s = summarize(rows)
    assert s["hit@5"] == 0.5 and s["faithfulness"] == 4 and s["not_found"] == 0.5
    assert s["retrieval_s"] == 2.0 and s["relevance"] is None


def test_load_questions_skips_placeholders(tmp_path):
    path = tmp_path / "q.jsonl"
    rows = [
        {"id": "q01", "question": "", "expected_pages": []},
        {"id": "d01", "question": "Which tool?", "expected_pages": [59], "verified": False},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert [q["id"] for q in load_questions(path)] == ["d01"]
    assert len(load_questions(path, include_placeholders=True)) == 2


def test_cached_chat_calls_model_once(tmp_path):
    llm = fake_llm(["first", "second"])
    chat = CachedChat(llm, "t", cache=SummaryCache(tmp_path / "c.json"))
    assert chat.complete("sys", "user") == "first"
    assert chat.complete("sys", "user") == "first"  # cached, the model is not asked again
    assert chat.complete("sys", "other") == "second"
