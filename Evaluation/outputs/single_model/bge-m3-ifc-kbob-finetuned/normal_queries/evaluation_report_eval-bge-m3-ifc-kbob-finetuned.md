## Evaluation Report

Generated: 2026-04-16 10:03:27

### Inputs
- Summary CSV: `summary_eval-bge-m3-ifc-kbob-finetuned_model-1d06a0d7_queries-b9bc9eb9_no-reranker-7521044b.csv`
- Details CSV: `details_eval-bge-m3-ifc-kbob-finetuned_model-1d06a0d7_queries-b9bc9eb9_no-reranker-7521044b.csv`

### Overview
![Model overview](overview_eval-bge-m3-ifc-kbob-finetuned_model-1d06a0d7_queries-b9bc9eb9_no-reranker-7521044b.svg)

### Leaderboard

#### Baseline (Bi-Encoder)

| Rank | Model | Hit@1 | Hit@10 | Hit@20 | Hit@30 | Hit@50 | MRR@10 | MAP@10 | nDCG@10 | Recall@10 | Avg expected score | Hit@1 95% CI | Hit@10 95% CI | MRR@10 95% CI | nDCG@10 95% CI | Top1 errors |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|
| 1 | models\\Hygroskopisch\\bge-m3-ifc-kbob-finetuned | 97.43% | 99.49% | 99.74% | 99.74% | 100.00% | 0.984 | 0.932 | 0.954 | 0.960 | 0.911 | [0.954, 0.990] | [0.987, 1.000] | [0.971, 0.994] | [0.939, 0.968] | 10 |

#### Reranked (Bi-Encoder + Cross-Encoder)

| Rank | Model | Cross-Encoder | Hit@1 | Hit@10 | Hit@20 | Hit@30 | Hit@50 | MRR@10 | MAP@10 | nDCG@10 | Recall@10 | Avg expected score | Hit@1 95% CI | Hit@10 95% CI | MRR@10 95% CI | nDCG@10 95% CI | Top1 errors |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|

Anzahl Queries: 389

### Hardest Queries (Baseline)
Queries mit den meisten Top1-Fehlern in der Baseline:

- (1 Fehler) IfcCovering MEMBRANE Abdichtung
- (1 Fehler) IfcPavement FLEXIBLE Polymermodifiziertes Bitumen
- (1 Fehler) IfcPile BORED Beton C20/25 insitu
- (1 Fehler) IfcPile BORED Stahlbeton C20/25 insitu
- (1 Fehler) IfcPile COHESION Beton C20/25 insitu
