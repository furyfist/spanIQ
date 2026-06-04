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

**How it works:** Embeds text using `all-MiniLM-L6-v2` (80MB local model, no API). Computes cosine similarity between actual_output and expected_output (if provided) or the mean of baseline embeddings. Similarity >= threshold passes.

**When to use:** Replacing deepeval's AnswerRelevancyMetric for cases where you have reference outputs. Free, deterministic, runs locally.

**When NOT to use:** When semantic similarity is not a good proxy for quality — e.g., creative writing, brainstorming, or outputs where many valid phrasings exist.

**Default threshold:** 0.70 (similarity >= 0.70 passes)

**Requires:** `expected_output` or `baseline_outputs`

---

## OutputStabilityMetric

**What it measures:** Is the LLM producing structurally consistent outputs?

**How it works:** Extracts four structural features from each output — character count, word count, sentence count, average word length. Computes JS divergence between the actual output's features and the distribution of baseline features.

**When to use:** Detecting when a model starts producing wildly different length responses, switches from structured to unstructured output, or changes verbosity significantly.

**When NOT to use:** When structural variation is intentional — e.g., a model that sometimes gives short and sometimes long answers depending on context.

**Default threshold:** 0.15 (JS < 0.15 passes)

**Requires:** `baseline_outputs`

---

## ConsistencyMetric

**What it measures:** Given the same input, are repeated outputs consistent with the historical distribution of outputs?

**How it works:** Embeds actual_output and all baseline_outputs. Computes pairwise cosine distances between actual_output and each baseline (actual-to-baseline distances), and between all baseline pairs (baseline-to-baseline distances). Runs a KS test between these two distance distributions. A non-significant result (low KS statistic) means the actual output is within normal variation.

**When to use:** Detecting when a model becomes erratic — outputs to the same prompt vary more than they historically did. The KS test catches distributional shape changes that mean-based metrics miss.

**When NOT to use:** When you have fewer than 5 baseline outputs — the baseline-to-baseline distance distribution is too small for a meaningful KS test.

**Default threshold:** 0.05 (KS statistic < 0.05 passes)

**Requires:** `baseline_outputs` (minimum 5)
