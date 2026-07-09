# Dataset labeling — how the good/bad rows were produced

The benchmark needs both correct outputs and realistic wrong ones to measure
whether a tool catches bad outputs (precision / recall / F1). This file documents
exactly how every labeled row was created, so the labels are auditable by hand.

## Files

| File | Role | Tracked in git |
|---|---|---|
| `*_seed.jsonl` | Immutable good-only source rows (the original v4 curated data) | yes |
| `*.jsonl` | Generated labeled datasets the benchmark runs on | yes |
| `build_labeled.py` | Deterministic, idempotent generator: seed → labeled | yes |

Regenerate with:

```bash
python -m benchmarks.datasets.build_labeled
```

It reads only the `*_seed.jsonl` files and overwrites the `*.jsonl` files. It
never mutates the seeds, so the output is byte-identical on every run.

## Row schema

Each labeled row adds two fields to the seed schema:

- `label` — `"good"` or `"bad"`
- `failure_kind` — `null` for good rows; one of the taxonomy values below for bad
  rows (see `docs/plans/benchmark_v2_accuracy.md` for definitions)

## How each dataset's negatives were built

### qa_factual.jsonl — 100 rows (50 good / 50 bad)

- **good**: the seed's known-correct answer, verbatim.
- **bad**: a hand-authored, confidently-worded *wrong* answer for the same
  question — a plausible distractor, not gibberish. Most are `wrong_entity`
  (right shape, wrong specific fact, e.g. "The capital of France is Berlin.");
  a few are `contradiction` (asserts the opposite, e.g. "Plants absorb oxygen
  during photosynthesis.").
- The 20 seed questions each get one authored wrong answer (`QA_BAD`), and 30
  additional authored question/answer pairs (`QA_EXTRA`) each contribute a
  good+bad pair, bringing the set to 100 balanced rows.

### summarization.jsonl — 16 rows (8 good / 8 bad)

- **good**: the seed's reference summary, verbatim.
- **bad**, alternating by index:
  - `omission` — the summary is clipped to its first clause, dropping required
    facts present in the reference.
  - `hallucination` — an invented, source-absent claim is appended ("…and the
    project received a Nobel Prize for its impact").

### rag_retrieval.jsonl — 16 rows (8 good / 8 bad)

- **good**: the seed's context-faithful answer, verbatim.
- **bad**, alternating by index:
  - `unfaithful_to_context` — a claim attributed to a "2050 government study not
    present in the passage" is appended, so the answer cites something absent from
    the retrieved context.
  - `contradiction` — the reference assertion is negated / prefixed with
    "Contrary to the passage, …".

## Balance and honesty notes

- qa_factual is class-balanced 50/50 at 100 rows — large enough for stable
  precision/recall. summarization and rag_retrieval are 50/50 at 16 rows; they are
  kept to genuinely distinct rows rather than padded with near-duplicate
  paraphrases, so their per-metric numbers carry more variance and should be read
  as directional.
- All negatives are synthetic and derived by disclosed rules. None were chosen to
  favor any particular tool. A reviewer can read `build_labeled.py` end to end and
  reproduce every label.
