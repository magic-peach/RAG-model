"""Decision support: compare new tabular data against the thesis's methods and findings.

Two features live here.

1. Knowledge card (one-time, reviewed by a person)
   ``python -m thesis_rag.analyze card`` asks the LLM to extract the thesis's
   objectives, methods, statistical thresholds, traits and key findings, each
   with a page citation, into ``data/knowledge_card.json``. Each field is
   extracted from its own retrieved passages (five small prompts rather than
   one huge one), and every citation is checked against those passages. The
   card is marked ``"reviewed": false`` until a person has checked it.

2. CSV analysis
   All arithmetic is done by pandas and scipy, never by the LLM: an 8B model
   cannot be trusted to compute a mean, let alone a p-value. The pipeline is

       CSV -> descriptive statistics, group comparisons (Welch t / ANOVA with
              non-parametric checks and Holm correction), correlations, and
              threshold checks against the knowledge card
           -> retrieve thesis passages about the same variables
           -> the LLM explains the computed numbers with citations and must
              end with a limitations note

   The output is labelled "decision support": it helps a researcher see how
   new data relates to the thesis, it does not make the decision.
"""

from __future__ import annotations

import io
import json
import logging
import math
import operator
import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import pandas as pd

from .config import Settings
from .generate import (
    cited_passages,
    format_context,
    get_llm,
    normalize_citations,
    passage_citation,
    source_dict,
)

if TYPE_CHECKING:
    from .generate import ThesisQA

log = logging.getLogger(__name__)

DECISION_SUPPORT_LABEL = (
    "Decision support: statistics computed from your data, explained against the thesis. "
    "Review with a domain expert before acting on it."
)
MAX_CSV_BYTES = 5_000_000
MAX_COLUMNS = 200
ALPHA = 0.05

# ---------------------------------------------------------------------------
# Knowledge card
# ---------------------------------------------------------------------------

CARD_PROMPT_VERSION = "v1"
CARD_SYSTEM = (
    "You extract structured facts from passages of a PhD thesis. Use only the passages. "
    "Copy names, tools and numbers exactly. For every item, give the citation tag of the "
    "passage it came from, copied exactly as shown after 'cite as'. If the passages "
    "contain nothing relevant, return an empty list. Reply with JSON only."
)

# field -> (retrieval query, what to extract, JSON shape of one item)
CARD_FIELDS: dict[str, tuple[str, str, str]] = {
    "objectives": (
        "objectives of the study aims of the research",
        "the research objectives of the thesis",
        '{"item": "...", "citation": "[...]"}',
    ),
    "methods": (
        "research methodology sequencing alignment variant calling tools pipeline",
        "the methods, tools and pipeline steps used, in order",
        '{"item": "step: tool or method and what it does", "citation": "[...]"}',
    ),
    "statistical_thresholds": (
        "filtering criteria threshold values QD FS SOR MQ MQRankSum ReadPosRankSum quality",
        "every numeric threshold or cut-off used to filter or classify data",
        '{"parameter": "QD", "operator": "<", "value": 2.0, "action": "exclude", '
        '"description": "...", "citation": "[...]"}',
    ),
    "traits": (
        "phenotypic traits drought tolerance nitrogen use efficiency root rot genotypes",
        "the plant traits studied and how genotypes differ in them",
        '{"item": "trait: description", "citation": "[...]"}',
    ),
    "key_findings": (
        "results key findings conclusions of the study",
        "the main results and conclusions",
        '{"item": "...", "citation": "[...]"}',
    ),
}


def card_path(settings: Settings):
    return settings.data_dir / "knowledge_card.json"


def load_knowledge_card(settings: Settings) -> dict | None:
    path = card_path(settings)
    return json.loads(path.read_text()) if path.is_file() else None


def _parse_items(raw: str) -> list[dict]:
    """Accept {"items": [...]}, a bare list, or {"<anything>": [...]}."""
    match = re.search(r"[\[{].*[\]}]", raw, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = data.get("items") or next((v for v in data.values() if isinstance(v, list)), [])
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []


def extract_field(qa: ThesisQA, llm, field: str) -> list[dict]:
    query, what, shape = CARD_FIELDS[field]
    passages = qa.retriever.retrieve(query).passages
    if not passages:
        return []
    user = (
        f"Passages:\n\n{format_context(passages)}\n\n---\n\n"
        f'Extract {what}. Reply as {{"items": [{shape}, ...]}}.'
    )
    from langchain_core.messages import HumanMessage, SystemMessage

    raw = str(llm.invoke([SystemMessage(content=CARD_SYSTEM), HumanMessage(content=user)]).content)
    valid_tags = {passage_citation(p) for p in passages}
    items = []
    for item in _parse_items(raw):
        citation = normalize_citations(str(item.get("citation", "")), passages)
        item["citation"] = citation
        # Flag rather than drop: the reviewer decides, but sees which ones to check first.
        item["citation_valid"] = citation in valid_tags
        items.append(item)
    return items


def build_knowledge_card(
    settings: Settings,
    qa: ThesisQA | None = None,
    llm=None,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict:
    path = card_path(settings)
    if path.is_file() and not force:
        return json.loads(path.read_text())
    if qa is None:
        from .generate import ThesisQA

        qa = ThesisQA(settings)
    llm = llm or get_llm(settings, temperature=0.0, num_predict=900, format="json")

    card: dict = {
        "document": qa.index.manifest.get("title", ""),
        "source": qa.index.manifest.get("source", ""),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": settings.llm_model,
        "prompt_version": CARD_PROMPT_VERSION,
        "reviewed": False,
    }
    for field in CARD_FIELDS:
        if progress:
            progress(f"Extracting {field}")
        card[field] = extract_field(qa, llm, field)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card, indent=2, ensure_ascii=False))
    return card


# ---------------------------------------------------------------------------
# CSV statistics (no LLM below this line until `explain`)
# ---------------------------------------------------------------------------


def _num(x) -> float | None:
    """JSON-safe float rounded to 4 significant figures (NaN/inf -> None)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return float(f"{x:.4g}")


def read_csv(data: bytes) -> pd.DataFrame:
    if len(data) > MAX_CSV_BYTES:
        raise ValueError(f"CSV is larger than {MAX_CSV_BYTES // 1_000_000} MB")
    try:
        df = pd.read_csv(io.BytesIO(data), sep=None, engine="python")  # sniff , ; or tab
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc
    if df.empty or df.shape[1] == 0:
        raise ValueError("CSV has no rows")
    if df.shape[1] > MAX_COLUMNS:
        raise ValueError(f"CSV has more than {MAX_COLUMNS} columns")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def describe(df: pd.DataFrame) -> list[dict]:
    rows = []
    for col in numeric_columns(df):
        s = df[col].dropna()
        rows.append(
            {
                "column": col,
                "n": int(s.size),
                "missing": int(df[col].isna().sum()),
                "mean": _num(s.mean()),
                "std": _num(s.std()),
                "min": _num(s.min()),
                "median": _num(s.median()),
                "max": _num(s.max()),
            }
        )
    return rows


_GROUP_HINT = re.compile(r"genotype|variety|cultivar|accession|group|treatment|line", re.I)


def pick_group_column(df: pd.DataFrame) -> str | None:
    """A categorical column with 2-20 levels, preferring names like 'genotype'."""
    candidates = []
    for col in df.columns:
        if col in numeric_columns(df):
            continue
        levels = df[col].dropna().nunique()
        if 2 <= levels <= min(20, max(len(df) // 2, 2)):
            candidates.append(col)
    hinted = [c for c in candidates if _GROUP_HINT.search(c)]
    return (hinted or candidates or [None])[0]


def holm(pvalues: list[float | None]) -> list[float | None]:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    present = sorted((p, i) for i, p in enumerate(pvalues) if p is not None)
    adjusted: list[float | None] = [None] * len(pvalues)
    m, running = len(present), 0.0
    for rank, (p, i) in enumerate(present):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return adjusted


def compare_groups(df: pd.DataFrame, group_col: str) -> list[dict]:
    """Per numeric column: do the groups differ?

    Two groups: Welch's t-test (no equal-variance assumption) with Cohen's d,
    plus Mann-Whitney U as a check that does not assume normality.
    Three or more: one-way ANOVA with eta squared, plus Kruskal-Wallis.
    p-values are Holm-adjusted across columns, because testing ten columns
    at alpha = 0.05 would otherwise produce a false positive fairly often.
    """
    from scipy import stats

    results = []
    for col in numeric_columns(df):
        groups = {
            str(name): g[col].dropna().to_numpy()
            for name, g in df.groupby(group_col, dropna=True)
            if g[col].notna().sum() >= 2
        }
        if len(groups) < 2:
            continue
        values = list(groups.values())
        means = {name: _num(v.mean()) for name, v in groups.items()}
        if len(groups) == 2:
            a, b = values
            test = stats.ttest_ind(a, b, equal_var=False)
            nonparam = stats.mannwhitneyu(a, b)
            pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) or float("nan")
            effect, effect_name = (a.mean() - b.mean()) / pooled, "cohens_d"
            test_name, nonparam_name = "Welch t-test", "Mann-Whitney U"
        else:
            test = stats.f_oneway(*values)
            nonparam = stats.kruskal(*values)
            allv = pd.Series([x for v in values for x in v])
            ss_between = sum(len(v) * (v.mean() - allv.mean()) ** 2 for v in values)
            ss_total = ((allv - allv.mean()) ** 2).sum()
            effect, effect_name = ss_between / ss_total if ss_total else float("nan"), "eta_sq"
            test_name, nonparam_name = "one-way ANOVA", "Kruskal-Wallis"
        results.append(
            {
                "column": col,
                "test": test_name,
                "statistic": _num(test.statistic),
                "p_value": _num(test.pvalue),
                "nonparametric_test": nonparam_name,
                "nonparametric_p": _num(nonparam.pvalue),
                effect_name: _num(effect),
                "group_means": means,
                "group_sizes": {name: int(v.size) for name, v in groups.items()},
            }
        )
    for row, adj in zip(results, holm([r["p_value"] for r in results]), strict=True):
        row["p_adjusted"] = _num(adj)
        row["significant"] = adj is not None and adj < ALPHA
    return results


def correlations(df: pd.DataFrame, max_pairs: int = 10) -> list[dict]:
    """Strongest Pearson correlations between numeric columns (|r| >= 0.3)."""
    from scipy import stats

    cols = numeric_columns(df)
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            sub = df[[a, b]].dropna()
            if len(sub) < 3 or sub[a].nunique() < 2 or sub[b].nunique() < 2:
                continue
            r, p = stats.pearsonr(sub[a], sub[b])
            if abs(r) >= 0.3:
                pairs.append({"x": a, "y": b, "r": _num(r), "p_value": _num(p), "n": len(sub)})
    pairs.sort(key=lambda d: -abs(d["r"] or 0))
    return pairs[:max_pairs]


_OPS: dict[str, Callable] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def check_thresholds(df: pd.DataFrame, card: dict | None) -> list[dict]:
    """Apply the thesis's numeric cut-offs to matching CSV columns.

    A column matches a threshold when its name equals the parameter name
    (case-insensitive), e.g. a "QD" column against "exclude if QD < 2.0".
    """
    if not card:
        return []
    by_name = {c.lower(): c for c in numeric_columns(df)}
    results = []
    for t in card.get("statistical_thresholds", []):
        col = by_name.get(str(t.get("parameter", "")).strip().lower())
        op = _OPS.get(str(t.get("operator", "")).strip())
        value = _num(t.get("value"))
        if not col or not op or value is None:
            continue
        s = df[col].dropna()
        flagged = int(op(s, value).sum())
        exclude = str(t.get("action", "exclude")).lower().startswith("excl")
        results.append(
            {
                "column": col,
                "rule": f"{'exclude' if exclude else 'keep'} if {col} {t['operator']} {value}",
                "n": int(s.size),
                "matching_rule": flagged,
                "passing": int(s.size - flagged) if exclude else flagged,
                "passing_share": _num((s.size - flagged if exclude else flagged) / s.size)
                if s.size
                else None,
                "citation": t.get("citation", ""),
            }
        )
    return results


def compute_stats(df: pd.DataFrame, group_col: str | None, card: dict | None) -> dict:
    if group_col and group_col not in df.columns:
        raise ValueError(f"Column {group_col!r} is not in the CSV")
    group_col = group_col or pick_group_column(df)
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "numeric_columns": numeric_columns(df),
        "group_column": group_col,
        "descriptive": describe(df),
        "comparisons": compare_groups(df, group_col) if group_col else [],
        "correlations": correlations(df),
        "thresholds": check_thresholds(df, card),
    }


def stats_report(stats: dict) -> str:
    """Plain-text rendering of the computed statistics for the LLM prompt."""
    lines = [f"Rows: {stats['rows']}. Columns: {', '.join(stats['columns'])}."]
    lines.append("Descriptive statistics:")
    for d in stats["descriptive"]:
        lines.append(
            f"- {d['column']}: n={d['n']}, mean={d['mean']}, sd={d['std']}, "
            f"median={d['median']}, range={d['min']} to {d['max']}"
        )
    if stats["comparisons"]:
        lines.append(f"Group comparisons by {stats['group_column']} (Holm-adjusted p):")
        for c in stats["comparisons"]:
            sig = "significant" if c["significant"] else "not significant"
            lines.append(
                f"- {c['column']}: {c['test']} p_adj={c['p_adjusted']} ({sig} at 0.05); "
                f"group means {c['group_means']}"
            )
    if stats["correlations"]:
        lines.append("Correlations (Pearson):")
        lines += [
            f"- {c['x']} vs {c['y']}: r={c['r']}, p={c['p_value']}" for c in stats["correlations"]
        ]
    if stats["thresholds"]:
        lines.append("Thesis thresholds applied to this data:")
        lines += [
            f"- {t['rule']}: {t['passing']} of {t['n']} rows pass {t['citation']}"
            for t in stats["thresholds"]
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Explanation (LLM)
# ---------------------------------------------------------------------------

EXPLAIN_SYSTEM = """You help a researcher interpret new data in light of a PhD thesis on \
mulberry genomics. You receive statistics that were already computed by software, and \
passages from the thesis. Rules:
1. Never calculate, estimate or change a number. Only quote numbers from the statistics \
block, exactly as written.
2. Relate the results to the thesis only through the passages, and end each sentence that \
uses the thesis with that passage's tag, copied exactly as shown after "cite as".
3. If the thesis passages say nothing relevant about these variables, say so plainly.
4. Write 1-3 short paragraphs, then a final paragraph that starts with "Limitations:" \
covering sample size, missing values, multiple testing and that correlation is not \
causation, as they apply here."""


def explain(stats: dict, qa: ThesisQA) -> dict:
    variables = ", ".join(stats["numeric_columns"][:8])
    query = f"{variables} {stats['group_column'] or ''} thresholds traits genotypes results"
    passages = qa.retriever.retrieve(query.strip()).passages
    from langchain_core.messages import HumanMessage, SystemMessage

    user = (
        f"Statistics computed from the uploaded data:\n{stats_report(stats)}\n\n---\n\n"
        f"Thesis passages:\n\n{format_context(passages)}\n\n---\n\n"
        "Explain what these results mean for the researcher, in light of the thesis."
    )
    msg = qa.llm.invoke([SystemMessage(content=EXPLAIN_SYSTEM), HumanMessage(content=user)])
    text = normalize_citations(str(msg.content).strip(), passages)
    return {
        "explanation": text,
        "sources": [source_dict(i, p) for i, p in enumerate(passages, start=1)],
        "cited": cited_passages(text, passages),
    }


def analyze_csv(
    data: bytes, settings: Settings, qa: ThesisQA | None, group_col: str | None = None
) -> dict:
    """Full CSV pipeline. ``qa=None`` returns statistics only (no index or LLM needed)."""
    df = read_csv(data)
    stats = compute_stats(df, group_col or None, load_knowledge_card(settings))
    result = {"label": DECISION_SUPPORT_LABEL, "stats": stats}
    if qa is not None:
        result.update(explain(stats, qa))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """``python -m thesis_rag.analyze card`` or ``... csv data.csv [--group genotype]``."""
    import argparse

    from .config import get_settings

    parser = argparse.ArgumentParser(description="Knowledge card and CSV decision support.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    card_p = sub.add_parser("card", help="extract data/knowledge_card.json from the thesis")
    card_p.add_argument("--force", action="store_true", help="re-extract even if it exists")
    csv_p = sub.add_parser("csv", help="analyse a CSV file against the thesis")
    csv_p.add_argument("path")
    csv_p.add_argument("--group", default=None, help="categorical column to compare by")
    csv_p.add_argument("--stats-only", action="store_true", help="skip the LLM explanation")
    args = parser.parse_args()
    settings = get_settings()

    if args.cmd == "card":
        card = build_knowledge_card(settings, force=args.force, progress=print)
        counts = {k: len(v) for k, v in card.items() if isinstance(v, list)}
        print(f'Wrote {card_path(settings)}: {counts}. Review it, then set "reviewed": true.')
        return

    from .generate import ThesisQA

    with open(args.path, "rb") as f:
        data = f.read()
    qa = None if args.stats_only else ThesisQA(settings)
    print(json.dumps(analyze_csv(data, settings, qa, args.group), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
