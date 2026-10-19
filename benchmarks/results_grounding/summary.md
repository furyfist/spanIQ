# Grounding benchmark

Positive class: hallucinated. AUC 0.5 is chance. Best F1 uses the threshold that maximises F1 on the same data, so it is optimistic.

| dataset | scorer | n | hallucinated | AUC | best F1 |
|---|---|---|---|---|---|
| local | unsupported_claims | 32 | 16 | 0.734 | 0.667 |
| local | novel_words | 32 | 16 | 0.131 | 0.667 |
