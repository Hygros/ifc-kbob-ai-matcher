## Evaluation Report

Generated: 2026-03-01 13:00:18

### Inputs
- Summary CSV: `summary_single_finetuned_20260301_130017_BAAI_bge-m3_ifcentity_material_strength_no-reranker.csv`
- Details CSV: `details_single_finetuned_20260301_130017_BAAI_bge-m3_ifcentity_material_strength_no-reranker.csv`

### Overview
![Model overview](overview_single_finetuned_20260301_130017_BAAI_bge-m3_ifcentity_material_strength_no-reranker.svg)

### Leaderboard

#### Baseline (Bi-Encoder)

| Rank | Model | Hit@1 | Hit@10 | Hit@20 | Hit@30 | Hit@50 | MRR@10 | MAP@10 | nDCG@10 | Recall@10 | Avg expected score | Hit@1 95% CI | Hit@10 95% CI | MRR@10 95% CI | nDCG@10 95% CI | Top1 errors |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|
| 1 | BAAI/bge-m3 | 69.18% | 96.06% | 100.00% | 100.00% | 100.00% | 0.790 | 0.760 | 0.813 | 0.931 | 0.645 | [0.638, 0.742] | [0.932, 0.982] | [0.746, 0.826] | [0.774, 0.849] | 86 |

#### Reranked (Bi-Encoder + Cross-Encoder)

| Rank | Model | Cross-Encoder | Hit@1 | Hit@10 | Hit@20 | Hit@30 | Hit@50 | MRR@10 | MAP@10 | nDCG@10 | Recall@10 | Avg expected score | Hit@1 95% CI | Hit@10 95% CI | MRR@10 95% CI | nDCG@10 95% CI | Top1 errors |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|

Anzahl Queries: 279

### Hardest Queries (Baseline)
Queries mit den meisten Top1-Fehlern in der Baseline:

- (11 Fehler) IfcBeam Beton C30/37
- (7 Fehler) IfcWall Beton C30/37
- (6 Fehler) IfcPile Beton C20/25
- (6 Fehler) IfcRamp Beton C30/37
- (6 Fehler) IfcWall Stahlbeton C30/37
