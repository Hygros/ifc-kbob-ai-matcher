## Evaluation Report

Generated: 2026-04-16 11:14:26

### Inputs
- Summary CSV: `summary_eval-bge-m3-ifc-kbob-finetuned-missing-queries_typos_model-1d06a0d7_queries_missing_typo-ba9d4d3a_no-reranker-7521044b.csv`
- Details CSV: `details_eval-bge-m3-ifc-kbob-finetuned-missing-queries_typos_model-1d06a0d7_queries_missing_typo-ba9d4d3a_no-reranker-7521044b.csv`

### Overview
![Model overview](overview_eval-bge-m3-ifc-kbob-finetuned-missing-queries_typos_model-1d06a0d7_queries_missing_typo-ba9d4d3a_no-reranker-7521044b.svg)

### Leaderboard

#### Baseline (Bi-Encoder)

| Rank | Model | Hit@1 | Hit@10 | Hit@20 | Hit@30 | Hit@50 | MRR@10 | MAP@10 | nDCG@10 | Recall@10 | Avg expected score | Hit@1 95% CI | Hit@10 95% CI | MRR@10 95% CI | nDCG@10 95% CI | Top1 errors |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|
| 1 | models\\Hygroskopisch\\bge-m3-ifc-kbob-finetuned | 68.12% | 88.17% | 94.34% | 96.92% | 98.46% | 0.739 | 0.682 | 0.731 | 0.805 | 0.837 | [0.634, 0.725] | [0.850, 0.911] | [0.695, 0.778] | [0.690, 0.767] | 124 |

#### Reranked (Bi-Encoder + Cross-Encoder)

| Rank | Model | Cross-Encoder | Hit@1 | Hit@10 | Hit@20 | Hit@30 | Hit@50 | MRR@10 | MAP@10 | nDCG@10 | Recall@10 | Avg expected score | Hit@1 95% CI | Hit@10 95% CI | MRR@10 95% CI | nDCG@10 95% CI | Top1 errors |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|

Anzahl Queries: 389

### Hardest Queries (Baseline)
Queries mit den meisten Top1-Fehlern in der Baseline:

- (2 Fehler) IfcTendonConduit COPULER
- (2 Fehler) IfcBuildingElementPart Sathl
- (2 Fehler) IfcBuildingElementProxy RIAL
- (1 Fehler) IfcBearing CYLINDRIVAL
- (1 Fehler) IfcBearing YCLINDRICAL
