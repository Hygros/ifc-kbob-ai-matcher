## Evaluation Report

Generated: 2026-03-07 09:05:57

### Inputs
- Summary CSV: `summary_finetuned_epoch-04-6685778a_ifcentity_material_s-aa2be901_bge-reranker-v2-m3-a92d7717.csv`
- Details CSV: `details_finetuned_epoch-04-6685778a_ifcentity_material_s-aa2be901_bge-reranker-v2-m3-a92d7717.csv`

### Overview
![Model overview](overview_finetuned_epoch-04-6685778a_ifcentity_material_s-aa2be901_bge-reranker-v2-m3-a92d7717.svg)

### Leaderboard

#### Baseline (Bi-Encoder)

| Rank | Model | Hit@1 | Hit@10 | Hit@20 | Hit@30 | Hit@50 | MRR@10 | MAP@10 | nDCG@10 | Recall@10 | Avg expected score | Hit@1 95% CI | Hit@10 95% CI | MRR@10 95% CI | nDCG@10 95% CI | Top1 errors |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|
| 1 | Training/artifacts/models/bge-m3-finetuned-generated_queries_without_exposure/epochs/epoch-04 | 66.31% | 92.47% | 94.98% | 95.70% | 96.42% | 0.745 | 0.635 | 0.713 | 0.831 | 0.624 | [0.609, 0.724] | [0.894, 0.952] | [0.700, 0.790] | [0.677, 0.754] | 94 |

#### Reranked (Bi-Encoder + Cross-Encoder)

| Rank | Model | Cross-Encoder | Hit@1 | Hit@10 | Hit@20 | Hit@30 | Hit@50 | MRR@10 | MAP@10 | nDCG@10 | Recall@10 | Avg expected score | Hit@1 95% CI | Hit@10 95% CI | MRR@10 95% CI | nDCG@10 95% CI | Top1 errors |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|
| 1 | Training/artifacts/models/bge-m3-finetuned-generated_queries_without_exposure/epochs/epoch-04 | BAAI/bge-reranker-v2-m3 | 24.73% | 82.08% | 90.68% | 95.70% | 96.42% | 0.430 | 0.327 | 0.435 | 0.631 | 0.544 | [0.201, 0.294] | [0.781, 0.875] | [0.386, 0.474] | [0.397, 0.474] | 210 |

Anzahl Queries: 279

### Hardest Queries (Baseline)
Queries mit den meisten Top1-Fehlern in der Baseline:

- (5 Fehler) IfcBearing S235JR
- (5 Fehler) IfcBearing Stahl
- (5 Fehler) IfcColumn S235JR
- (5 Fehler) IfcPile Beton C20/25
- (4 Fehler) IfcMember Stahl

### Hardest Queries (Reranked)
Queries mit den meisten Top1-Fehlern nach Re-Ranking:

- (12 Fehler) IfcMember Stahl
- (11 Fehler) IfcBeam Beton C30/37
- (9 Fehler) IfcMember Holz
- (7 Fehler) IfcPlate Stahl
- (7 Fehler) IfcTrackElement Stahl
