# Validation Results

Calibrated CUSUM h = 1.00 (empirical, target ARL0 >= 500)

## Detection Delay (CUSUM) + Localization Error (PELT, pen=3.0)

| shift | delay (median traces) | localization error (median) | false alarms /1k |
|---|---|---|---|
| 0.5s | 79 | ±20 | 0.06 |
| 1.0s | 45 | ±5 | 0.06 |
| 2.0s | 17 | ±0 | 0.06 |
| 4.0s | 9 | ±0 | 0.06 |

## Attribution Accuracy (root component ranked first)

| lead gap | accuracy |
|---|---|
| 2 traces | 84% |
| 3 traces | 83% |
| 4 traces | 90% |
| 5 traces | 90% |
| 6 traces | 93% |
| 7 traces | 93% |
| 8 traces | 95% |
| 9 traces | 96% |
| 10 traces | 97% |
| 11 traces | 97% |
| 12 traces | 97% |
| 13 traces | 97% |
| 14 traces | 97% |
| 15 traces | 97% |

Attribution accuracy (lead gap >= 5): 95%
Attribution accuracy (lead gap < 5): 86% (documented limitation)
