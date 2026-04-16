## Evaluation Report

Generated: 2026-04-16 11:04:03

### Inputs
- Summary CSV: `summary_eval-bge-m3-ifc-kbob-finetuned-missing-queries_model-1d06a0d7_queries_missing-a91a834f_no-reranker-7521044b.csv`
- Details CSV: `details_eval-bge-m3-ifc-kbob-finetuned-missing-queries_model-1d06a0d7_queries_missing-a91a834f_no-reranker-7521044b.csv`

### Overview
![Model overview](overview_eval-bge-m3-ifc-kbob-finetuned-missing-queries_model-1d06a0d7_queries_missing-a91a834f_no-reranker-7521044b.svg)

### Leaderboard

#### Baseline (Bi-Encoder)

| Rank | Model | Hit@1 | Hit@10 | Hit@20 | Hit@30 | Hit@50 | MRR@10 | MAP@10 | nDCG@10 | Recall@10 | Avg expected score | Hit@1 95% CI | Hit@10 95% CI | MRR@10 95% CI | nDCG@10 95% CI | Top1 errors |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|
| 1 | models\\Hygroskopisch\\bge-m3-ifc-kbob-finetuned | 75.32% | 92.80% | 96.40% | 98.20% | 98.71% | 0.803 | 0.750 | 0.794 | 0.860 | 0.867 | [0.707, 0.792] | [0.900, 0.950] | [0.766, 0.835] | [0.759, 0.824] | 96 |

#### Reranked (Bi-Encoder + Cross-Encoder)

| Rank | Model | Cross-Encoder | Hit@1 | Hit@10 | Hit@20 | Hit@30 | Hit@50 | MRR@10 | MAP@10 | nDCG@10 | Recall@10 | Avg expected score | Hit@1 95% CI | Hit@10 95% CI | MRR@10 95% CI | nDCG@10 95% CI | Top1 errors |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|

Anzahl Queries: 389

### Hardest Queries (Baseline)
Queries mit den meisten Top1-Fehlern in der Baseline:

- (6 Fehler) IfcBuildingElementPart TrackElement
- (6 Fehler) IfcRailing HANDRAIL
- (5 Fehler) IfcRailing GUARDRAIL
- (4 Fehler) IfcBuildingElementProxy Stahl
- (2 Fehler) IfcBearing CYLINDRICAL
