# Evaluation

How the pipeline is measured, what the current numbers say, and what they do
not say. The latest generated table is in [`eval/results.md`](../eval/results.md).

## What is measured

Five configurations, each adding one change to the previous one, so every row
shows the effect of a single decision:

| Config | What it adds |
|---|---|
| `baseline` | The v1 notebook: `PyPDFLoader` text, 4,000-character page chunks, dense search only, no parents or summaries |
| `structured` | Section-aware chunks with headers, parent passages, tables as chunks, summary routing |
| `hybrid` | BM25 fused with dense search by Reciprocal Rank Fusion |
| `reranker` | `bge-reranker-base` cross-encoder over the 20 fused candidates (**the default**) |
| `multi_query` | Three LLM rephrasings per question, also fused by RRF |

Every configuration uses the same answer prompt and model, so differences come
from retrieval and chunking only.

| Metric | Definition |
|---|---|
| **hit@5** | Share of questions where at least one of the 5 passages overlaps the expected pages |
| **MRR@5** | Mean of 1 / rank of the first such passage (0 if none in the top 5) |
| **Faithfulness** | LLM judge, 1–5: is every claim in the answer supported by the passages? (`eval/prompts/faithfulness.txt`) |
| **Relevance** | LLM judge, 1–5: does the answer address the question? (`eval/prompts/relevance.txt`) |
| **Cited** | Share of answers (excluding "Not found") with at least one citation tag matching a retrieved passage |
| **Not found** | Share of answers that were exactly "Not found in the thesis." |

Two choices worth knowing:

- **Summary passages never count as hits.** The whole-thesis summary spans
  pp. 16–85 and would "hit" every question, which would reward routing rather
  than retrieval.
- **"Not found" answers get no faithfulness score.** They make no claims, so
  there is nothing to check; they are counted in their own column instead.

## Results (10 unverified questions)

| Config | hit@5 | MRR@5 | Faithfulness | Relevance | Cited | Not found |
|---|---|---|---|---|---|---|
| baseline | 70% | 0.575 | 4.83 | 3.30 | 83% | 40% |
| + structured | 80% | 0.650 | 4.78 | 4.50 | 78% | 10% |
| + hybrid | 90% | 0.717 | 4.89 | 4.50 | 78% | 10% |
| **+ reranker** | **100%** | **0.925** | 4.90 | 4.60 | 50% | 0% |
| + multi-query | 100% | 0.925 | 4.78 | 4.40 | 56% | 10% |

### What the numbers support

- **Retrieval improves at every step.** hit@5 goes 70 → 80 → 90 → 100% and MRR
  0.58 → 0.65 → 0.72 → 0.93. The reranker is the largest single gain: it moves
  the right passage to rank 1 for 9 of 10 questions.
- **The baseline's real failure is "Not found".** With page chunks the right
  passage was often missing or buried, so the model correctly refused in 4 of
  10 cases. Relevance (3.3 → 4.5+) reflects that, not better wording.
- **Multi-query adds nothing here** and costs an extra LLM call per question,
  so it stays off by default. It may help hand-written questions whose wording
  differs from the thesis more than these drafts do.
- **Exact identifiers need BM25.** d09 ("drought tolerance of the K2 genotype")
  is answered by a table that dense search alone ranks 4th; adding BM25 moves it
  up and the reranker puts it first.

### What they do not support

- **10 questions is a small sample.** One question moves hit@5 by 10 points;
  differences under ~20 points are not meaningful.
- **The questions are unverified drafts.** Llama wrote each one from a single
  passage, which favours finding that passage. Some reference answers are weak
  (d01, d03, d05). The 25 empty entries in `eval/questions.jsonl` are for
  hand-written questions, which are the real test.
- **The judge is the same model as the generator.** No paid APIs, so llama3.1:8b
  grades itself and is lenient: faithfulness sits at 4.8–4.9 everywhere and
  varied by up to 0.4 between identical runs. Compare rows with each other; do
  not read the scores as absolute quality.
- **The Cited rate is the weak spot.** Llama 8B sometimes answers correctly
  without any tag, more often with the reranker's longer parent passages. The
  answers are not wrong, but they are not traceable either. This is a prompt and
  model limitation, recorded honestly rather than patched by adding citations
  after generation.
- **Latency was measured on a heavily loaded laptop** with the reranker on CPU;
  only the relative cost of reranking (~10–15 s here) is informative.

## Choosing a faster reranker

After the comparison above, latency became the priority: on the development
laptop, reranking with bge-reranker-base took ~14 s per question. The same
retrieval eval with `ms-marco-MiniLM-L-6-v2` (run as ONNX through fastembed):

| Reranker | hit@5 | hit@3 | MRR | Rerank time |
|---|---|---|---|---|
| bge-reranker-base (PyTorch) | 100% | 90% | 0.925 | ~14 s |
| MiniLM-L-6 (ONNX) | 100% | 100% | 0.883 | ~0.1–1 s |

Every expected page is still found, and within the top 3, so the default now
uses MiniLM and sends 3 passages instead of 5. The answer-quality columns above
were measured with the previous answer prompt and are not re-run for the new
one (which adds reasoning and labelled general-knowledge answers); re-running
`python eval/run_eval.py` would refresh them.

## Lessons from building the eval

- Reading the answers, not just the scores, found a real bug: the first answer
  prompt showed the format as `<claim> [Ch.X §Y, p.Z]` and the model copied the
  placeholder literally. The fix (passages labelled `cite as [tag]`, plus
  `normalize_citations`) is covered by unit tests.
- Every LLM call in the harness is cached by its full input, so after a code
  change only the calls whose input changed are paid for again. The full run
  (50 answers + ~90 judgments) took about an hour on the loaded machine.

## Running it

```bash
python eval/draft_questions.py          # optional: regenerate the 10 drafts
python eval/run_eval.py --retrieval-only   # hit@5 and MRR, minutes, no LLM
python eval/run_eval.py                 # everything, writes eval/results.md
python eval/run_eval.py --verified-only --configs baseline,reranker
```

To add a question, fill one of the empty entries in `eval/questions.jsonl`:

```json
{"id": "q01", "question": "Which aligner was used for the exome reads?",
 "expected_answer": "BWA", "expected_pages": [71], "source": "manual", "verified": true}
```
