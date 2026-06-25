# Benchmark Methodology

This document describes exactly how the spanIQ determinism benchmark is run, what it measures, what it deliberately does not measure, and how anyone can reproduce the numbers themselves.

## Why this file exists

Without a methodology document, the benchmark table is just numbers. With it, the numbers become evidence you can check. This document serves three readers:

1. The developer who sees the claim and asks "how did you configure the competitors?"
2. The reproducer who wants to clone the repo and verify the numbers.
3. The reviewer who wants to judge whether the claims are honest.

Everything here is derived from the code in `benchmarks/`. Where a number comes from a live run, the run is described. Where a number requires an API key we did not have at publish time, it is marked `[PENDING]` rather than guessed.
