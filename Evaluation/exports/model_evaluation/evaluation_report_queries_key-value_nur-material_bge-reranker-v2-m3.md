## Evaluation Report

Generated: 2026-02-28 18:41:43

### Inputs
- Summary CSV: `summary_queries_key-value_nur-material_bge-reranker-v2-m3.csv`
- Details CSV: `details_queries_key-value_nur-material_bge-reranker-v2-m3.csv`

### Overview
![Model overview](overview_queries_key-value_nur-material_bge-reranker-v2-m3_bge-reranker-v2-m3.svg)

### Leaderboard

#### Baseline (Bi-Encoder)

| Rank | Model | Hit@1 | Hit@10 | Hit@20 | Hit@30 | Hit@50 | MRR@10 | MAP@10 | nDCG@10 | Recall@10 | Avg expected score | Hit@1 95% CI | Hit@10 95% CI | MRR@10 95% CI | nDCG@10 95% CI | Top1 errors |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|
| 1 | intfloat/multilingual-e5-large | 35.84% | 82.44% | 84.95% | 90.68% | 92.47% | 0.549 | 0.479 | 0.571 | 0.745 | 0.859 | [0.308, 0.410] | [0.778, 0.867] | [0.509, 0.594] | [0.533, 0.609] | 179 |
| 2 | BAAI/bge-m3 | 35.13% | 75.99% | 84.59% | 91.04% | 96.77% | 0.472 | 0.391 | 0.484 | 0.664 | 0.586 | [0.303, 0.405] | [0.713, 0.816] | [0.426, 0.522] | [0.444, 0.527] | 181 |
| 3 | kforth/IfcMaterial2MP | 32.97% | 73.12% | 83.15% | 88.53% | 94.62% | 0.469 | 0.337 | 0.427 | 0.567 | 0.632 | [0.271, 0.389] | [0.681, 0.782] | [0.423, 0.519] | [0.388, 0.467] | 187 |
| 4 | google/embeddinggemma-300m | 31.54% | 82.80% | 84.23% | 94.27% | 97.49% | 0.450 | 0.386 | 0.506 | 0.783 | 0.672 | [0.269, 0.373] | [0.787, 0.871] | [0.411, 0.495] | [0.476, 0.545] | 191 |
| 5 | intfloat/multilingual-e5-base | 24.01% | 75.27% | 83.15% | 87.46% | 89.96% | 0.374 | 0.337 | 0.423 | 0.632 | 0.866 | [0.194, 0.299] | [0.704, 0.799] | [0.329, 0.425] | [0.382, 0.473] | 212 |
| 6 | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 23.66% | 69.89% | 78.14% | 79.21% | 83.15% | 0.343 | 0.237 | 0.346 | 0.565 | 0.784 | [0.195, 0.290] | [0.647, 0.756] | [0.302, 0.392] | [0.315, 0.379] | 213 |
| 7 | google-bert/bert-base-multilingual-uncased | 21.51% | 68.10% | 78.85% | 86.02% | 93.19% | 0.356 | 0.279 | 0.370 | 0.551 | 0.731 | [0.167, 0.265] | [0.627, 0.737] | [0.312, 0.403] | [0.332, 0.412] | 219 |
| 8 | sentence-transformers/distiluse-base-multilingual-cased-v2 | 17.20% | 68.46% | 88.17% | 89.25% | 95.70% | 0.372 | 0.314 | 0.406 | 0.593 | 0.513 | [0.125, 0.219] | [0.627, 0.731] | [0.333, 0.413] | [0.370, 0.449] | 231 |
| 9 | google-bert/bert-base-multilingual-cased | 12.54% | 69.89% | 82.44% | 88.89% | 92.11% | 0.247 | 0.199 | 0.306 | 0.578 | 0.602 | [0.090, 0.165] | [0.645, 0.753] | [0.211, 0.284] | [0.275, 0.343] | 244 |
| 10 | sentence-transformers/LaBSE | 11.47% | 51.97% | 57.71% | 61.65% | 72.76% | 0.263 | 0.241 | 0.311 | 0.464 | 0.490 | [0.079, 0.156] | [0.462, 0.573] | [0.226, 0.301] | [0.274, 0.349] | 247 |
| 11 | google-bert/bert-base-german-cased | 10.75% | 41.22% | 61.29% | 64.52% | 86.38% | 0.222 | 0.163 | 0.219 | 0.328 | 0.862 | [0.072, 0.147] | [0.355, 0.468] | [0.183, 0.259] | [0.181, 0.253] | 249 |
| 12 | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 3.94% | 36.56% | 69.53% | 75.63% | 80.65% | 0.110 | 0.062 | 0.103 | 0.170 | 0.681 | [0.020, 0.065] | [0.314, 0.430] | [0.084, 0.136] | [0.080, 0.127] | 268 |
| 13 | kforth/IfcElement2ConstructionSets | 0.00% | 43.73% | 51.25% | 62.72% | 79.21% | 0.083 | 0.060 | 0.130 | 0.304 | 0.982 | [0.000, 0.000] | [0.385, 0.497] | [0.072, 0.095] | [0.114, 0.149] | 279 |

#### Reranked (Bi-Encoder + Cross-Encoder)

| Rank | Model | Cross-Encoder | Hit@1 | Hit@10 | Hit@20 | Hit@30 | Hit@50 | MRR@10 | MAP@10 | nDCG@10 | Recall@10 | Avg expected score | Hit@1 95% CI | Hit@10 95% CI | MRR@10 95% CI | nDCG@10 95% CI | Top1 errors |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|
| 1 | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | BAAI/bge-reranker-v2-m3 | 25.45% | 73.12% | 74.91% | 75.63% | 80.65% | 0.403 | 0.249 | 0.356 | 0.527 | 0.577 | [0.211, 0.308] | [0.686, 0.785] | [0.365, 0.450] | [0.328, 0.389] | 208 |
| 2 | google-bert/bert-base-multilingual-uncased | BAAI/bge-reranker-v2-m3 | 24.01% | 72.40% | 84.23% | 86.02% | 93.19% | 0.397 | 0.253 | 0.354 | 0.519 | 0.577 | [0.194, 0.297] | [0.679, 0.781] | [0.355, 0.449] | [0.319, 0.389] | 212 |
| 3 | intfloat/multilingual-e5-base | BAAI/bge-reranker-v2-m3 | 23.66% | 74.55% | 82.80% | 87.46% | 89.96% | 0.383 | 0.258 | 0.369 | 0.564 | 0.580 | [0.192, 0.294] | [0.701, 0.796] | [0.345, 0.433] | [0.344, 0.400] | 213 |
| 4 | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | BAAI/bge-reranker-v2-m3 | 22.58% | 73.12% | 78.14% | 79.21% | 83.15% | 0.384 | 0.237 | 0.345 | 0.523 | 0.579 | [0.179, 0.281] | [0.685, 0.785] | [0.343, 0.433] | [0.317, 0.378] | 216 |
| 5 | kforth/IfcElement2ConstructionSets | BAAI/bge-reranker-v2-m3 | 19.71% | 53.05% | 61.29% | 62.72% | 79.21% | 0.301 | 0.202 | 0.274 | 0.393 | 0.571 | [0.158, 0.246] | [0.477, 0.588] | [0.259, 0.348] | [0.243, 0.311] | 224 |
| 6 | google-bert/bert-base-multilingual-cased | BAAI/bge-reranker-v2-m3 | 12.90% | 80.29% | 87.10% | 88.89% | 92.11% | 0.386 | 0.314 | 0.444 | 0.721 | 0.589 | [0.091, 0.165] | [0.754, 0.849] | [0.350, 0.418] | [0.410, 0.472] | 243 |
| 7 | intfloat/multilingual-e5-large | BAAI/bge-reranker-v2-m3 | 12.19% | 78.49% | 88.53% | 90.68% | 92.47% | 0.338 | 0.255 | 0.372 | 0.606 | 0.593 | [0.086, 0.167] | [0.746, 0.835] | [0.306, 0.381] | [0.343, 0.404] | 245 |
| 8 | google-bert/bert-base-german-cased | BAAI/bge-reranker-v2-m3 | 10.75% | 47.31% | 63.08% | 64.52% | 86.38% | 0.210 | 0.164 | 0.233 | 0.392 | 0.589 | [0.072, 0.143] | [0.416, 0.530] | [0.178, 0.245] | [0.198, 0.263] | 249 |
| 9 | kforth/IfcMaterial2MP | BAAI/bge-reranker-v2-m3 | 10.39% | 74.91% | 86.02% | 88.53% | 94.62% | 0.338 | 0.253 | 0.364 | 0.583 | 0.592 | [0.072, 0.138] | [0.703, 0.808] | [0.300, 0.374] | [0.332, 0.395] | 250 |
| 10 | google/embeddinggemma-300m | BAAI/bge-reranker-v2-m3 | 10.39% | 78.14% | 83.87% | 94.27% | 97.49% | 0.331 | 0.245 | 0.363 | 0.599 | 0.593 | [0.072, 0.140] | [0.738, 0.833] | [0.296, 0.368] | [0.336, 0.392] | 250 |
| 11 | BAAI/bge-m3 | BAAI/bge-reranker-v2-m3 | 9.68% | 77.42% | 89.96% | 91.04% | 96.77% | 0.310 | 0.226 | 0.346 | 0.590 | 0.594 | [0.065, 0.133] | [0.728, 0.824] | [0.281, 0.347] | [0.318, 0.376] | 252 |
| 12 | sentence-transformers/LaBSE | BAAI/bge-reranker-v2-m3 | 8.96% | 53.41% | 59.50% | 61.65% | 72.76% | 0.259 | 0.170 | 0.246 | 0.351 | 0.593 | [0.061, 0.127] | [0.477, 0.599] | [0.222, 0.296] | [0.215, 0.279] | 254 |
| 13 | sentence-transformers/distiluse-base-multilingual-cased-v2 | BAAI/bge-reranker-v2-m3 | 7.89% | 74.19% | 86.02% | 89.25% | 95.70% | 0.297 | 0.210 | 0.321 | 0.530 | 0.586 | [0.054, 0.113] | [0.692, 0.792] | [0.268, 0.331] | [0.297, 0.350] | 257 |

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
