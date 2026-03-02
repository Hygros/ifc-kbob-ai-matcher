## Evaluation Report

Generated: 2026-02-28 19:40:54

### Inputs
- Summary CSV: `summary_ifcentity_predefined_material_bge-reranker-v2-m3.csv`
- Details CSV: `details.csv`

### Overview
![Model overview](overview_ifcentity_predefined_material_bge-reranker-v2-m3_bge-reranker-v2-m3.svg)

### Leaderboard

#### Baseline (Bi-Encoder)

| Rank | Model | Hit@1 | Hit@10 | Hit@20 | Hit@30 | Hit@50 | MRR@10 | MAP@10 | nDCG@10 | Recall@10 | Avg expected score | Hit@1 95% CI | Hit@10 95% CI | MRR@10 95% CI | nDCG@10 95% CI | Top1 errors |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|
| 1 | BAAI/bge-m3 | 36.56% | 74.91% | 83.51% | 88.53% | 92.83% | 0.496 | 0.433 | 0.512 | 0.665 | 0.543 | [0.310, 0.425] | [0.695, 0.799] | [0.450, 0.544] | [0.464, 0.555] | 181 |
| 2 | intfloat/multilingual-e5-large | 31.18% | 63.08% | 77.42% | 83.87% | 88.17% | 0.403 | 0.352 | 0.417 | 0.549 | 0.851 | [0.262, 0.366] | [0.570, 0.688] | [0.356, 0.461] | [0.375, 0.472] | 179 |
| 3 | sentence-transformers/distiluse-base-multilingual-cased-v2 | 30.47% | 53.41% | 62.37% | 72.40% | 84.23% | 0.370 | 0.287 | 0.359 | 0.469 | 0.572 | [0.249, 0.366] | [0.475, 0.591] | [0.318, 0.424] | [0.314, 0.406] | 231 |
| 4 | sentence-transformers/LaBSE | 29.75% | 54.84% | 64.16% | 72.04% | 79.93% | 0.387 | 0.342 | 0.394 | 0.474 | 0.461 | [0.244, 0.353] | [0.486, 0.609] | [0.338, 0.437] | [0.347, 0.444] | 247 |
| 5 | google/embeddinggemma-300m | 26.52% | 83.87% | 88.17% | 90.32% | 97.85% | 0.438 | 0.373 | 0.496 | 0.770 | 0.586 | [0.215, 0.312] | [0.796, 0.885] | [0.399, 0.483] | [0.460, 0.536] | 191 |
| 6 | intfloat/multilingual-e5-base | 24.73% | 55.91% | 65.95% | 77.78% | 87.10% | 0.342 | 0.279 | 0.337 | 0.433 | 0.857 | [0.197, 0.301] | [0.505, 0.616] | [0.294, 0.395] | [0.296, 0.387] | 212 |
| 7 | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 15.05% | 42.29% | 53.41% | 62.01% | 79.57% | 0.221 | 0.136 | 0.191 | 0.261 | 0.525 | [0.108, 0.188] | [0.371, 0.487] | [0.184, 0.266] | [0.160, 0.225] | 268 |
| 8 | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 10.75% | 36.92% | 48.39% | 59.86% | 73.84% | 0.170 | 0.098 | 0.143 | 0.195 | 0.567 | [0.072, 0.151] | [0.317, 0.430] | [0.134, 0.217] | [0.115, 0.173] | 213 |
| 9 | kforth/IfcMaterial2MP | 7.89% | 48.75% | 64.87% | 72.76% | 86.02% | 0.181 | 0.118 | 0.186 | 0.316 | 0.582 | [0.047, 0.113] | [0.430, 0.552] | [0.146, 0.212] | [0.152, 0.213] | 187 |
| 10 | google-bert/bert-base-multilingual-cased | 6.09% | 24.73% | 41.22% | 55.20% | 81.36% | 0.106 | 0.068 | 0.102 | 0.163 | 0.621 | [0.032, 0.086] | [0.201, 0.301] | [0.076, 0.136] | [0.075, 0.129] | 244 |
| 11 | google-bert/bert-base-multilingual-uncased | 4.30% | 30.82% | 48.03% | 60.22% | 76.70% | 0.110 | 0.059 | 0.102 | 0.175 | 0.681 | [0.022, 0.068] | [0.251, 0.358] | [0.083, 0.139] | [0.080, 0.121] | 219 |
| 12 | kforth/IfcElement2ConstructionSets | 0.72% | 7.53% | 11.11% | 14.70% | 29.75% | 0.024 | 0.008 | 0.018 | 0.030 | 0.981 | [0.000, 0.018] | [0.047, 0.108] | [0.013, 0.038] | [0.010, 0.027] | 279 |
| 13 | google-bert/bert-base-german-cased | 0.36% | 6.81% | 8.96% | 13.62% | 23.30% | 0.020 | 0.013 | 0.024 | 0.045 | 0.829 | [0.000, 0.011] | [0.039, 0.093] | [0.010, 0.032] | [0.013, 0.036] | 249 |

#### Reranked (Bi-Encoder + Cross-Encoder)

| Rank | Model | Cross-Encoder | Hit@1 | Hit@10 | Hit@20 | Hit@30 | Hit@50 | MRR@10 | MAP@10 | nDCG@10 | Recall@10 | Avg expected score | Hit@1 95% CI | Hit@10 95% CI | MRR@10 95% CI | nDCG@10 95% CI | Top1 errors |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|
| 1 | google-bert/bert-base-multilingual-cased | BAAI/bge-reranker-v2-m3 | 26.52% | 52.69% | 54.84% | 55.20% | 81.36% | 0.347 | 0.243 | 0.307 | 0.386 | 0.519 | [0.220, 0.323] | [0.477, 0.583] | [0.306, 0.398] | [0.273, 0.347] | 243 |
| 2 | intfloat/multilingual-e5-base | BAAI/bge-reranker-v2-m3 | 26.16% | 68.82% | 74.55% | 77.78% | 87.10% | 0.396 | 0.280 | 0.378 | 0.543 | 0.531 | [0.213, 0.315] | [0.642, 0.742] | [0.355, 0.448] | [0.344, 0.420] | 213 |
| 3 | google/embeddinggemma-300m | BAAI/bge-reranker-v2-m3 | 23.66% | 78.85% | 87.10% | 90.32% | 97.85% | 0.413 | 0.307 | 0.417 | 0.623 | 0.540 | [0.194, 0.283] | [0.746, 0.835] | [0.376, 0.454] | [0.384, 0.452] | 250 |
| 4 | intfloat/multilingual-e5-large | BAAI/bge-reranker-v2-m3 | 22.94% | 75.27% | 81.36% | 83.87% | 88.17% | 0.406 | 0.303 | 0.410 | 0.607 | 0.539 | [0.179, 0.276] | [0.703, 0.799] | [0.366, 0.451] | [0.377, 0.447] | 245 |
| 5 | BAAI/bge-m3 | BAAI/bge-reranker-v2-m3 | 21.51% | 75.63% | 83.51% | 88.53% | 92.83% | 0.389 | 0.280 | 0.389 | 0.592 | 0.540 | [0.170, 0.260] | [0.710, 0.805] | [0.349, 0.433] | [0.356, 0.424] | 252 |
| 6 | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | BAAI/bge-reranker-v2-m3 | 21.51% | 53.76% | 59.50% | 62.01% | 79.57% | 0.309 | 0.208 | 0.280 | 0.386 | 0.523 | [0.168, 0.262] | [0.484, 0.595] | [0.267, 0.351] | [0.244, 0.319] | 208 |
| 7 | sentence-transformers/distiluse-base-multilingual-cased-v2 | BAAI/bge-reranker-v2-m3 | 20.43% | 64.87% | 69.89% | 72.40% | 84.23% | 0.367 | 0.268 | 0.360 | 0.508 | 0.534 | [0.158, 0.254] | [0.599, 0.703] | [0.329, 0.412] | [0.324, 0.401] | 257 |
| 8 | google-bert/bert-base-multilingual-uncased | BAAI/bge-reranker-v2-m3 | 19.00% | 47.67% | 57.35% | 60.22% | 76.70% | 0.275 | 0.175 | 0.235 | 0.314 | 0.531 | [0.136, 0.237] | [0.412, 0.538] | [0.227, 0.320] | [0.196, 0.273] | 212 |
| 9 | kforth/IfcMaterial2MP | BAAI/bge-reranker-v2-m3 | 18.28% | 60.93% | 68.46% | 72.76% | 86.02% | 0.322 | 0.211 | 0.298 | 0.443 | 0.533 | [0.136, 0.222] | [0.556, 0.670] | [0.282, 0.363] | [0.264, 0.332] | 250 |
| 10 | sentence-transformers/LaBSE | BAAI/bge-reranker-v2-m3 | 17.20% | 56.27% | 65.23% | 72.04% | 79.93% | 0.314 | 0.215 | 0.295 | 0.408 | 0.536 | [0.127, 0.213] | [0.505, 0.620] | [0.266, 0.357] | [0.255, 0.333] | 254 |
| 11 | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | BAAI/bge-reranker-v2-m3 | 16.49% | 50.18% | 58.42% | 59.86% | 73.84% | 0.253 | 0.194 | 0.257 | 0.373 | 0.521 | [0.122, 0.213] | [0.452, 0.563] | [0.216, 0.301] | [0.225, 0.302] | 216 |
| 12 | google-bert/bert-base-german-cased | BAAI/bge-reranker-v2-m3 | 8.24% | 12.54% | 13.62% | 13.62% | 23.30% | 0.096 | 0.043 | 0.059 | 0.061 | 0.507 | [0.054, 0.117] | [0.086, 0.165] | [0.066, 0.132] | [0.038, 0.083] | 249 |
| 13 | kforth/IfcElement2ConstructionSets | BAAI/bge-reranker-v2-m3 | 3.58% | 13.62% | 13.98% | 14.70% | 29.75% | 0.069 | 0.040 | 0.056 | 0.069 | 0.510 | [0.014, 0.059] | [0.100, 0.179] | [0.045, 0.096] | [0.036, 0.078] | 224 |

Anzahl Queries: 279

### Hardest Queries (Baseline)
Queries mit den meisten Top1-Fehlern in der Baseline:

- (657 Fehler) Material: Beton
- (432 Fehler) Material: Stahl
- (275 Fehler) Material: Stahlbeton
- (169 Fehler) Material: B500B
- (154 Fehler) Material: Holz

### Hardest Queries (Reranked)
Queries mit den meisten Top1-Fehlern nach Re-Ranking:

- (689 Fehler) Material: Beton
- (483 Fehler) Material: Stahl
- (267 Fehler) Material: Stahlbeton
- (182 Fehler) Material: Holz
- (169 Fehler) Material: B500B
