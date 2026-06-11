# Decision: Temporal ordering, not causal inference

V3 ranks components by when their score series broke, not by causal graph analysis.

Rejected alternative: Shapley values, counterfactuals, causal inference (arXiv 2509.08682 route). These require interventional data or strong assumptions that cannot be validated from observational trace data alone.

V3's verdict language says "broke first" — never "caused". Temporal precedence within a proximity cluster is defensible and computable at $0. The confidence formula weights lead time and metric coverage. Causal graphs are a V4 candidate.
