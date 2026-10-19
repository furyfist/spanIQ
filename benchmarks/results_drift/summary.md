# Drift benchmark

Source: synthetic streams. Detection rate and delay are over runs where the scenario was switched on; delay counts traces after the change. False alarms are per 1,000 traces seen before any change (all traces for control).

| detector | scenario | runs | detected | median delay | false alarms / 1k | root correct |
|---|---|---|---|---|---|---|
| legacy | context_truncated | 10 | 100% | 2 | 0.0 | 100% |
| window | context_truncated | 10 | 100% | 29 | 0.0 | 100% |
| legacy | control | 10 | n/a | n/a | 0.0 | n/a |
| window | control | 10 | n/a | n/a | 0.0 | n/a |
| legacy | fact_swap | 10 | 0% | n/a | 0.0 | n/a |
| window | fact_swap | 10 | 90% | 119 | 0.0 | 100% |
| legacy | model_swap | 10 | 100% | 2 | 0.0 | 100% |
| window | model_swap | 10 | 100% | 29 | 0.0 | 100% |
| legacy | partial_retriever | 10 | 0% | n/a | 0.0 | n/a |
| window | partial_retriever | 10 | 100% | 94 | 0.0 | 0% |
| legacy | prompt_format_change | 10 | 100% | 2 | 0.0 | 100% |
| window | prompt_format_change | 10 | 100% | 29 | 0.0 | 100% |
| legacy | retriever_degraded | 10 | 100% | 2 | 0.0 | 0% |
| window | retriever_degraded | 10 | 100% | 29 | 0.0 | 90% |
| legacy | verbose_drift | 10 | 100% | 2 | 0.0 | 100% |
| window | verbose_drift | 10 | 100% | 29 | 0.0 | 100% |

## Sampled LLM judge, modeled

Expected traces after a change before a judge sampling a fraction of traffic has flagged 3 outputs, for different chances that the judge flags a changed output. Cost assumes $0.000165 per call. This table is arithmetic, not a run.

| sampling rate | cost / 1k traces | delay if p_flag=0.3 | delay if p_flag=0.6 | delay if p_flag=0.9 |
|---|---|---|---|---|
| 1% | $0.0016 | 1000 | 500 | 333 |
| 5% | $0.0082 | 200 | 100 | 67 |
| 20% | $0.0329 | 50 | 25 | 17 |
