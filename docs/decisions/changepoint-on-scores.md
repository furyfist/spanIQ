# Decision: Run changepoint detection on metric score series, not raw embeddings

V3 uses the float score series that V2 already produces as input to CUSUM and PELT.

Rejected alternative: run changepoint detection on raw embeddings (multivariate time series). This would require inventing a multivariate pipeline, is harder to explain, and ignores work V2 already did. Scores are 1-D, comparable, and stored. Layering on top of V2 is cleaner than reinventing it.
