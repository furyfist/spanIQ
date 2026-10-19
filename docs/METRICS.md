# Metrics Reference

## ResponseDriftMetric

**What it measures:** Has the vocabulary and word distribution of responses shifted from baseline?

**How it works:** Tokenizes actual_output and all baseline_outputs into word-frequency distributions, then computes Population Stability Index (PSI) between them. PSI < 0.1 means no significant drift. PSI 0.1–0.25 is moderate. PSI > 0.25 is significant.

**When to use:** Detecting prompt drift, model swaps, or style changes that alter vocabulary without necessarily changing meaning.

**When NOT to use:** When you only have 1–2 baseline outputs — PSI needs a reasonable reference distribution. Also not useful when word choice naturally varies a lot across valid responses.

**Default threshold:** 0.10 (PSI < 0.10 passes)

**Requires:** `baseline_outputs`

---

## SemanticSimilarityMetric

**What it measures:** Is the actual output semantically close to the expected or baseline outputs?

**How it works:** Embeds text using `all-MiniLM-L6-v2` (80MB local model, no API; set `SPANIQ_EMBEDDING_MODEL` or call `spaniq.statistical.embeddings.set_model()` to use another model or a local path). Computes cosine similarity between actual_output and expected_output (if provided) or the mean of baseline embeddings. Baseline embeddings are cached, so monitoring does not re-encode the baseline for every trace. Similarity >= threshold passes.

**When to use:** Replacing deepeval's AnswerRelevancyMetric for cases where you have reference outputs. Free, deterministic, runs locally.

**When NOT to use:** When semantic similarity is not a good proxy for quality — e.g., creative writing, brainstorming, or outputs where many valid phrasings exist. It also cannot catch an output that copies the reference and adds an invented claim: the copy is more similar to the reference than a correct paraphrase is (see `BENCHMARK_ACCURACY.md`).

**Default threshold:** 0.70 (similarity >= 0.70 passes)

**Requires:** `expected_output` or `baseline_outputs`

---

## OutputStabilityMetric

**What it measures:** Is the LLM producing structurally consistent outputs?

**How it works:** Extracts four structural features from each output — character count, word count, sentence count, average word length. Each feature scores 0 while the actual output stays inside the range the baseline outputs span, and rises toward 1 the further outside it lands (scaled by the range width, with a floor so a constant baseline feature does not make every change a hard 1.0). The score is the mean over the four features, and the reason names the feature that moved most.

**When to use:** Detecting when a model starts producing wildly different length responses, switches from structured to unstructured output, or changes verbosity significantly.

**When NOT to use:** When structural variation is intentional — e.g., a model that sometimes gives short and sometimes long answers depending on context.

**Default threshold:** 0.15 (deviation < 0.15 passes)

**Requires:** `baseline_outputs`

---

## ConsistencyMetric

**What it measures:** Given the same input, are repeated outputs consistent with the historical distribution of outputs?

**How it works:** Embeds actual_output and all baseline_outputs. Computes pairwise cosine distances between actual_output and each baseline (actual-to-baseline distances), and between all baseline pairs (baseline-to-baseline distances). Runs a one-sided KS test asking whether the actual-to-baseline distances are larger than the baseline-to-baseline ones, so an output that is closer than usual never fails. The score is the KS statistic.

**When to use:** Detecting when a model becomes erratic — outputs to the same prompt vary more than they historically did. The KS test catches distributional shape changes that mean-based metrics miss.

**When NOT to use:** When you have fewer than 5 baseline outputs — the baseline-to-baseline distance distribution is too small for a meaningful KS test.

**Default threshold:** the one-sided KS critical value at `alpha=0.05` for the baseline's sample sizes, recomputed per measurement and reported as the metric's threshold. A fixed cut-off ignores sample size; with 5 to 30 baseline outputs, chance alone often exceeds a small fixed value, which is why the old default of 0.05 failed nearly every output. Pass `threshold=<float>` for a fixed cut-off.

**Requires:** `baseline_outputs` (minimum 5)
