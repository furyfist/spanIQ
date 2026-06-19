# Decision: PELT penalty defaults to BIC (3 * log(n)), not a fixed constant

`detect_changepoints()` now defaults `penalty=None`, which at call time computes `3 * log(n)` from the actual series length. Fixed constants can no longer be passed in implicitly — they must be an explicit override.

**The bug:** `penalty` was hardcoded to `3.0`. For a ~180-point scored series (200 traces minus 20-trace warmup), BIC gives `3 * log(180) ≈ 15.3` — about 5× larger. With penalty=3.0, PELT needed very little evidence before declaring a changepoint, and the first few post-warmup scores (where variance is still settling) were sufficient to trigger a spurious break at trace ~10 on a system that was healthy until trace 101. This produced a false root-cause determination before the real injected drift arrived.

**What we chose:** `penalty=None` as the default, with `bic_penalty(n)` computed inside `detect_changepoints()` from `len(series)`. This scales automatically with series length — longer series require proportionally more evidence. The `attributor.py` `pelt_penalty` parameter and the CLI `--penalty` flag both default to `None` and pass it through unchanged, so the BIC formula is used end-to-end unless a caller explicitly overrides it.

**What we rejected:** a larger hardcoded default (e.g. `penalty=15.0`). A fixed constant would be correct for n=180 but wrong for n=50 (too conservative) or n=5000 (too permissive). BIC's log(n) dependence is the standard data-driven solution.

**What would change this:** empirical validation showing BIC is systematically too conservative for short series (<50 scored points) or too permissive for very long ones (>10,000 points). In those regimes, a length-dependent floor/ceiling on the computed penalty may be warranted. The `bic_penalty(n)` helper is intentionally exposed so callers can compose it: `penalty=max(10.0, bic_penalty(n))`.

**Commit:** `e6534ae` — fix(pelt): default penalty to BIC (3*log(n)) instead of hardcoded 3.0
