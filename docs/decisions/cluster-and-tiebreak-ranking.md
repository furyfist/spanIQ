# Decision: Chain clustering + metric-breadth tie-break for attribution ranking

Attribution now groups PELT break indices into clusters (chain: each member within `cluster_window` traces of the previous) before ranking. Within the event cluster, tie-breaking is: (1) earliest PELT index, (2) most broken metrics, (3) earliest CUSUM alarm. Cascade members compute their `lead_gap` relative to the preceding member in the chain, not to a hardcoded zero.

**The bugs:**

1. **No clustering.** The original code sorted all component break indices globally and declared the minimum the root cause. A single spurious early break in any component — even from noise right after the warmup window stabilized — would unconditionally win root-cause status over a real break at trace 101, because 1 < 101 with no further qualification.

2. **Broken cascade confidence.** `lead_gap` was only computed for `i == 0` (the component that sorted first). Every other component got `lead_gap = 0`, zeroing out 40% of its confidence score unconditionally. Cascade confidence values were not comparable to each other or to the root-cause confidence.

3. **Naïve tie-break.** When two components tied on PELT index (PELT cannot separate breaks closer together than its localization resolution at the given series length), the winner was determined by Python's default list ordering — effectively arbitrary.

**What we chose:**

- **Chain clustering:** sort all breaks ascending, then walk the list and start a new cluster whenever the gap to the previous member exceeds `cluster_window`. Take the largest cluster as the main event. Isolated breaks outside the cluster are treated as separate (possibly noise) events and classified as healthy relative to the main event.
- **Tie-break ordering:** within a cluster with tied PELT indices, rank by (a) most broken metrics — broader metric impact is stronger evidence of root cause, not cascade; (b) earlier CUSUM alarm — independent online detection provides a second, non-PELT signal of which component shifted first.
- **Per-member lead_gap:** `lead_gap` for index `i` is now `break_trace_index[i] - break_trace_index[i-1]`, giving every cascade member a real gap score. Confidence values across the chain are now comparable.

**What we rejected:** using CUSUM alarm index as the primary sort key. CUSUM fires on rolling-window scores which are contaminated by the cascade: generation's vocabulary shift (switching from "Paris" to "I cannot determine the answer") produces a large, fast PSI signal that CUSUM catches earlier than retrieval's empty-context drift, even though retrieval broke first. CUSUM alarm order is unreliable as a primary tiebreaker when the cascade contamination is strong; metric breadth is a more stable signal.

**Honest limitation:** when two components' real breaks are closer together than PELT's localization resolution at the given series length (roughly `min_size` traces), PELT will return identical or near-identical indices. The tie-break resolves ranking using independent evidence rather than claiming false index precision. This is a property of the method documented in `ATTRIBUTION.md`, not a bug to be papered over.

**Commits:** `a0b9b5b` — fix(attributor): add warmup offset to PELT break indices so they map to real trace numbers  
`d6c84c0` — fix(attributor): chain-cluster PELT breaks and tie-break by metric count before CUSUM
