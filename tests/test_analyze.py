import json

import pandas as pd
import pytest

from thesis_rag.analyze import (
    DECISION_SUPPORT_LABEL,
    _parse_items,
    analyze_csv,
    build_knowledge_card,
    check_thresholds,
    compare_groups,
    correlations,
    describe,
    holm,
    pick_group_column,
    read_csv,
    stats_report,
)
from thesis_rag.config import Settings
from thesis_rag.generate import ThesisQA

from .fakes import fake_llm
from .test_generate import StubRetriever, _passage

CSV = (
    b"genotype,QD,yield,height\n"
    b"K2,1.5,10,100\nK2,3.0,11,102\nK2,2.5,12,104\nK2,4.0,10.5,101\n"
    b"V1,8.0,20,140\nV1,9.5,21,141\nV1,7.0,22,145\nV1,1.0,19.5,139\n"
)


@pytest.fixture
def df():
    return read_csv(CSV)


def test_describe(df):
    rows = {r["column"]: r for r in describe(df)}
    assert set(rows) == {"QD", "yield", "height"}
    assert rows["yield"]["n"] == 8 and rows["yield"]["mean"] == 15.75
    assert rows["QD"]["min"] == 1.0 and rows["QD"]["missing"] == 0


def test_pick_group_column(df):
    assert pick_group_column(df) == "genotype"
    assert pick_group_column(df[["QD", "yield"]]) is None


def test_compare_two_groups_uses_welch(df):
    rows = {r["column"]: r for r in compare_groups(df, "genotype")}
    y = rows["yield"]
    assert y["test"] == "Welch t-test" and y["nonparametric_test"] == "Mann-Whitney U"
    assert y["significant"] and y["p_adjusted"] >= y["p_value"]
    assert y["group_means"] == {"K2": 10.88, "V1": 20.62}
    assert y["cohens_d"] < 0  # K2 lower than V1


def test_compare_three_groups_uses_anova():
    df = pd.DataFrame({"g": list("aaabbbccc"), "v": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
    (row,) = compare_groups(df, "g")
    assert row["test"] == "one-way ANOVA" and 0 < row["eta_sq"] <= 1


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([None, 0.2]) == [None, 0.2]


def test_correlations(df):
    pairs = correlations(df)
    assert pairs[0]["x"] == "yield" and pairs[0]["y"] == "height" and pairs[0]["r"] > 0.9


def test_thresholds_from_card(df):
    card = {
        "statistical_thresholds": [
            {
                "parameter": "QD",
                "operator": "<",
                "value": 2.0,
                "action": "exclude",
                "citation": "[Ch.3 §3.8, pp.59–60]",
            },
            {"parameter": "MQ", "operator": "<", "value": 40, "action": "exclude"},
        ]
    }
    (t,) = check_thresholds(df, card)  # MQ is not a column, so it is skipped
    assert t["rule"] == "exclude if QD < 2.0"
    assert t["matching_rule"] == 2 and t["passing"] == 6 and t["passing_share"] == 0.75


def test_read_csv_errors():
    with pytest.raises(ValueError, match="no rows"):
        read_csv(b"a,b\n")
    with pytest.raises(ValueError, match="larger than"):
        read_csv(b"x" * 6_000_000)


def test_semicolon_csv_is_sniffed():
    assert list(read_csv(b"a;b\n1;2\n3;4\n").columns) == ["a", "b"]


def test_parse_items_variants():
    assert _parse_items('{"items": [{"item": "a"}]}') == [{"item": "a"}]
    assert _parse_items('[{"item": "b"}]') == [{"item": "b"}]
    assert _parse_items('{"objectives": [{"item": "c"}, "junk"]}') == [{"item": "c"}]
    assert _parse_items("no json") == []


def test_analyze_csv_stats_only(tmp_path):
    result = analyze_csv(CSV, Settings(data_dir=tmp_path), qa=None)
    assert result["label"] == DECISION_SUPPORT_LABEL
    assert result["stats"]["group_column"] == "genotype"
    assert "explanation" not in result
    assert "Welch t-test" in stats_report(result["stats"])


def test_analyze_csv_unknown_group(tmp_path):
    with pytest.raises(ValueError, match="not in the CSV"):
        analyze_csv(CSV, Settings(data_dir=tmp_path), qa=None, group_col="nope")


def test_analyze_csv_explains_with_citations(tmp_path):
    passages = [_passage(1, "3.8", 59), _passage(2, "4.11", 76)]
    llm = fake_llm(["V1 yields more (mean 20.62 vs 10.88) [Passage 2]. Limitations: n=8 is small."])
    qa = ThesisQA(settings=None, index=object(), retriever=StubRetriever(passages), llm=llm)
    result = analyze_csv(CSV, Settings(data_dir=tmp_path), qa=qa)
    assert result["explanation"].startswith("V1 yields more")
    assert "[Ch.4 §4.11, p.76]" in result["explanation"] and result["cited"] == [2]
    assert len(result["sources"]) == 2


def test_build_knowledge_card(tmp_path):
    passages = [_passage(1, "3.8", 59)]
    responses = [
        json.dumps({"items": [{"item": f"{field} fact", "citation": "[Passage 1]"}]})
        for field in range(4)
    ]
    responses.insert(
        2,
        json.dumps(
            {
                "items": [
                    {
                        "parameter": "QD",
                        "operator": "<",
                        "value": 2.0,
                        "action": "exclude",
                        "citation": "[Ch.9 §9.9, p.1]",
                    }
                ]
            }
        ),
    )
    stub = StubRetriever(passages)
    qa = ThesisQA(settings=None, index=object(), retriever=stub, llm=fake_llm([]))
    qa.index = type("I", (), {"manifest": {"title": "T", "source": "t.pdf"}})()
    settings = Settings(data_dir=tmp_path)
    card = build_knowledge_card(settings, qa=qa, llm=fake_llm(responses))
    assert card["reviewed"] is False and card["document"] == "T"
    assert card["objectives"][0]["citation"] == "[Ch.4 §3.8, p.59]"
    assert card["objectives"][0]["citation_valid"] is True
    assert card["statistical_thresholds"][0]["citation_valid"] is False
    assert json.loads((tmp_path / "knowledge_card.json").read_text()) == card
    # Second call reads the saved card instead of calling the LLM again.
    assert build_knowledge_card(settings, qa=qa, llm=fake_llm([])) == card
