## Evaluation Report

Generated: 2026-04-16 11:08:54

### Inputs
- Summary CSV: `summary_eval-bge-m3-ifc-kbob-finetuned-typos_model-1d06a0d7_queries_typos-ec28a584_no-reranker-7521044b.csv`
- Details CSV: `details_eval-bge-m3-ifc-kbob-finetuned-typos_model-1d06a0d7_queries_typos-ec28a584_no-reranker-7521044b.csv`

### Overview
![Model overview](overview_eval-bge-m3-ifc-kbob-finetuned-typos_model-1d06a0d7_queries_typos-ec28a584_no-reranker-7521044b.svg)

### Leaderboard

#### Baseline (Bi-Encoder)

| Rank | Model | Hit@1 | Hit@10 | Hit@20 | Hit@30 | Hit@50 | MRR@10 | MAP@10 | nDCG@10 | Recall@10 | Avg expected score | Hit@1 95% CI | Hit@10 95% CI | MRR@10 95% CI | nDCG@10 95% CI | Top1 errors |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|
| 1 | models\\Hygroskopisch\\bge-m3-ifc-kbob-finetuned | 88.43% | 94.86% | 98.20% | 98.97% | 99.49% | 0.909 | 0.844 | 0.876 | 0.890 | 0.880 | [0.848, 0.918] | [0.928, 0.967] | [0.881, 0.935] | [0.847, 0.902] | 45 |

#### Reranked (Bi-Encoder + Cross-Encoder)

| Rank | Model | Cross-Encoder | Hit@1 | Hit@10 | Hit@20 | Hit@30 | Hit@50 | MRR@10 | MAP@10 | nDCG@10 | Recall@10 | Avg expected score | Hit@1 95% CI | Hit@10 95% CI | MRR@10 95% CI | nDCG@10 95% CI | Top1 errors |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|

Anzahl Queries: 389

### Hardest Queries (Baseline)
Queries mit den meisten Top1-Fehlern in der Baseline:

- (1 Fehler) IfcCourse ARMOUR5 Gestedin
- (1 Fehler) IfcCourse PAVEENT Asühalt
- (1 Fehler) IfcCovering MESMBRANE Sbdichtung
- (1 Fehler) IfcCovering WRYPPING Kunssttoff
- (1 Fehler) IfcMember CHORD Hokz
