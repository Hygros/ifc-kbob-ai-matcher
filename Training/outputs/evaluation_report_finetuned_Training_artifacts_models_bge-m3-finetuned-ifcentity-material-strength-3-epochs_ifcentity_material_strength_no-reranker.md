## Evaluation Report

Generated: 2026-03-01 18:11:36

### Inputs
- Summary CSV: `summary_single_finetuned_20260301_181136_Training_artifacts_models_bge-m3-finetuned-ifcentity-material-strength-3-epochs_ifcentity_material_strength_no-reranker.csv`
- Details CSV: `details_single_finetuned_20260301_181136_Training_artifacts_models_bge-m3-finetuned-ifcentity-material-strength-3-epochs_ifcentity_material_strength_no-reranker.csv`

### Overview
![Model overview](overview_single_finetuned_20260301_181136_Training_artifacts_models_bge-m3-finetuned-ifcentity-material-strength-3-epochs_ifcentity_material_strength_no-reranker.svg)

### Leaderboard

#### Baseline (Bi-Encoder)

| Rank | Model | Hit@1 | Hit@10 | Hit@20 | Hit@30 | Hit@50 | MRR@10 | MAP@10 | nDCG@10 | Recall@10 | Avg expected score | Hit@1 95% CI | Hit@10 95% CI | MRR@10 95% CI | nDCG@10 95% CI | Top1 errors |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|
| 1 | Training/artifacts/models/bge-m3-finetuned-ifcentity-material-strength-3-epochs | 76.70% | 99.28% | 99.64% | 100.00% | 100.00% | 0.834 | 0.806 | 0.858 | 0.962 | 0.649 | [0.717, 0.817] | [0.980, 1.000] | [0.796, 0.871] | [0.829, 0.888] | 65 |

#### Reranked (Bi-Encoder + Cross-Encoder)

| Rank | Model | Cross-Encoder | Hit@1 | Hit@10 | Hit@20 | Hit@30 | Hit@50 | MRR@10 | MAP@10 | nDCG@10 | Recall@10 | Avg expected score | Hit@1 95% CI | Hit@10 95% CI | MRR@10 95% CI | nDCG@10 95% CI | Top1 errors |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|

Anzahl Queries: 279

### Hardest Queries (Baseline)
Queries mit den meisten Top1-Fehlern in der Baseline:

- (10 Fehler) IfcBeam Beton C30/37
- (8 Fehler) IfcWall Beton C30/37
- (7 Fehler) IfcWall Stahlbeton C30/37
- (4 Fehler) IfcMember Stahl
- (4 Fehler) IfcMember Litze
