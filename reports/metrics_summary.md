# Model Metrics Summary

## Baseline vs Main Model — Macro F1

| Model | Accuracy | Macro F1 |
|---|---|---|
| BoW + Naive Bayes (baseline) | 0.9563 | 0.9494 |
| TF-IDF + LogReg (main) | 0.975 | 0.9684 |

## Per-Class F1 (TF-IDF + LogReg)

| Department | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Roads | 0.9706 | 1.0 | 0.9851 | 33 |
| Water | 0.9667 | 1.0 | 0.9831 | 29 |
| Electricity | 1.0 | 1.0 | 1.0 | 29 |
| Sanitation | 1.0 | 1.0 | 1.0 | 28 |
| Traffic | 1.0 | 0.8846 | 0.9388 | 26 |
| Parks | 0.875 | 0.9333 | 0.9032 | 15 |

## Key Observations

- `class_weight='balanced'` corrects for Parks being the smallest class (9% of corpus).
- Macro F1 is the right metric here — accuracy inflates when predicting the majority class.
- Roads vs Traffic is the hardest boundary due to shared vocabulary ('signal', 'road', 'junction').
- TF-IDF bigrams capture 'pipe burst', 'no water', 'power cut' as single features.