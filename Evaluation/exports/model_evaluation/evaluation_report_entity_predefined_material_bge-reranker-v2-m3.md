## Evaluation Report

Generated: 2026-02-28 19:40:50

### Inputs
- Summary CSV: `summary_entity_predefined_material_bge-reranker-v2-m3.csv`
- Details CSV: `details.csv`

### Overview
![Model overview](overview_entity_predefined_material_bge-reranker-v2-m3_bge-reranker-v2-m3.svg)

### Leaderboard

#### Baseline (Bi-Encoder)

| Rank | Model | Hit@1 | Hit@10 | Hit@20 | Hit@30 | Hit@50 | MRR@10 | MAP@10 | nDCG@10 | Recall@10 | Avg expected score | Hit@1 95% CI | Hit@10 95% CI | MRR@10 95% CI | nDCG@10 95% CI | Top1 errors |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|
| 1 | BAAI/bge-m3 | 32.26% | 73.12% | 82.08% | 88.17% | 93.19% | 0.450 | 0.387 | 0.465 | 0.623 | 0.552 | [0.272, 0.380] | [0.676, 0.785] | [0.407, 0.496] | [0.422, 0.514] | 181 |
| 2 | google/embeddinggemma-300m | 31.54% | 85.30% | 88.17% | 90.32% | 97.85% | 0.469 | 0.400 | 0.518 | 0.780 | 0.595 | [0.262, 0.375] | [0.810, 0.896] | [0.428, 0.517] | [0.479, 0.562] | 191 |
| 3 | intfloat/multilingual-e5-large | 28.67% | 70.61% | 81.36% | 85.66% | 90.68% | 0.411 | 0.361 | 0.443 | 0.619 | 0.852 | [0.237, 0.344] | [0.654, 0.763] | [0.365, 0.468] | [0.402, 0.496] | 179 |
| 4 | sentence-transformers/LaBSE | 28.32% | 60.22% | 71.33% | 74.55% | 79.57% | 0.396 | 0.344 | 0.406 | 0.512 | 0.504 | [0.233, 0.337] | [0.538, 0.667] | [0.346, 0.453] | [0.359, 0.457] | 247 |
| 5 | sentence-transformers/distiluse-base-multilingual-cased-v2 | 26.16% | 58.42% | 72.40% | 82.08% | 92.47% | 0.366 | 0.284 | 0.364 | 0.503 | 0.582 | [0.219, 0.315] | [0.525, 0.638] | [0.319, 0.416] | [0.320, 0.409] | 231 |
| 6 | intfloat/multilingual-e5-base | 26.16% | 60.57% | 75.63% | 82.80% | 88.53% | 0.358 | 0.307 | 0.370 | 0.499 | 0.860 | [0.217, 0.319] | [0.550, 0.670] | [0.314, 0.411] | [0.325, 0.419] | 212 |
| 7 | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 14.34% | 50.90% | 69.53% | 76.34% | 82.44% | 0.235 | 0.170 | 0.243 | 0.380 | 0.590 | [0.108, 0.183] | [0.450, 0.565] | [0.197, 0.277] | [0.210, 0.280] | 268 |
| 8 | google-bert/bert-base-multilingual-cased | 12.19% | 38.35% | 51.97% | 68.10% | 87.10% | 0.190 | 0.137 | 0.185 | 0.277 | 0.610 | [0.086, 0.158] | [0.335, 0.448] | [0.153, 0.228] | [0.151, 0.219] | 244 |
| 9 | kforth/IfcMaterial2MP | 11.11% | 61.29% | 72.04% | 78.49% | 89.96% | 0.249 | 0.174 | 0.256 | 0.416 | 0.597 | [0.075, 0.154] | [0.559, 0.672] | [0.211, 0.290] | [0.222, 0.290] | 187 |
| 10 | google-bert/bert-base-multilingual-uncased | 10.39% | 48.39% | 67.74% | 74.19% | 82.44% | 0.192 | 0.130 | 0.197 | 0.329 | 0.703 | [0.070, 0.142] | [0.427, 0.541] | [0.157, 0.229] | [0.164, 0.228] | 219 |
| 11 | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 4.66% | 40.50% | 60.22% | 72.76% | 83.87% | 0.125 | 0.096 | 0.155 | 0.290 | 0.645 | [0.022, 0.072] | [0.348, 0.464] | [0.101, 0.152] | [0.130, 0.185] | 213 |
| 12 | google-bert/bert-base-german-cased | 0.36% | 7.89% | 11.47% | 17.56% | 34.41% | 0.022 | 0.016 | 0.028 | 0.053 | 0.842 | [0.000, 0.011] | [0.050, 0.111] | [0.013, 0.034] | [0.017, 0.041] | 249 |
| 13 | kforth/IfcElement2ConstructionSets | 0.00% | 6.81% | 15.41% | 21.86% | 40.14% | 0.016 | 0.008 | 0.017 | 0.031 | 0.980 | [0.000, 0.000] | [0.043, 0.102] | [0.008, 0.025] | [0.009, 0.026] | 279 |

#### Reranked (Bi-Encoder + Cross-Encoder)

| Rank | Model | Cross-Encoder | Hit@1 | Hit@10 | Hit@20 | Hit@30 | Hit@50 | MRR@10 | MAP@10 | nDCG@10 | Recall@10 | Avg expected score | Hit@1 95% CI | Hit@10 95% CI | MRR@10 95% CI | nDCG@10 95% CI | Top1 errors |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|
| 1 | intfloat/multilingual-e5-base | BAAI/bge-reranker-v2-m3 | 25.09% | 70.97% | 78.49% | 82.80% | 88.53% | 0.392 | 0.267 | 0.369 | 0.542 | 0.547 | [0.201, 0.301] | [0.661, 0.765] | [0.349, 0.439] | [0.337, 0.408] | 213 |
| 2 | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | BAAI/bge-reranker-v2-m3 | 21.51% | 69.53% | 74.19% | 76.34% | 82.44% | 0.356 | 0.255 | 0.356 | 0.549 | 0.543 | [0.168, 0.262] | [0.649, 0.746] | [0.316, 0.400] | [0.324, 0.393] | 208 |
| 3 | sentence-transformers/distiluse-base-multilingual-cased-v2 | BAAI/bge-reranker-v2-m3 | 19.35% | 72.04% | 79.21% | 82.08% | 92.47% | 0.380 | 0.276 | 0.379 | 0.554 | 0.543 | [0.147, 0.238] | [0.670, 0.771] | [0.338, 0.421] | [0.348, 0.415] | 257 |
| 4 | google/embeddinggemma-300m | BAAI/bge-reranker-v2-m3 | 19.35% | 77.42% | 86.02% | 90.32% | 97.85% | 0.371 | 0.267 | 0.380 | 0.595 | 0.554 | [0.147, 0.237] | [0.729, 0.823] | [0.332, 0.409] | [0.348, 0.411] | 250 |
| 5 | kforth/IfcMaterial2MP | BAAI/bge-reranker-v2-m3 | 19.00% | 63.80% | 74.55% | 78.49% | 89.96% | 0.336 | 0.222 | 0.310 | 0.459 | 0.544 | [0.143, 0.238] | [0.590, 0.701] | [0.296, 0.380] | [0.274, 0.347] | 250 |
| 6 | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | BAAI/bge-reranker-v2-m3 | 18.64% | 64.52% | 71.68% | 72.76% | 83.87% | 0.316 | 0.228 | 0.322 | 0.503 | 0.542 | [0.138, 0.242] | [0.588, 0.701] | [0.271, 0.364] | [0.286, 0.358] | 216 |
| 7 | intfloat/multilingual-e5-large | BAAI/bge-reranker-v2-m3 | 18.28% | 73.84% | 82.44% | 85.66% | 90.68% | 0.367 | 0.268 | 0.375 | 0.573 | 0.553 | [0.133, 0.222] | [0.692, 0.789] | [0.327, 0.405] | [0.343, 0.407] | 245 |
| 8 | BAAI/bge-m3 | BAAI/bge-reranker-v2-m3 | 17.92% | 74.55% | 83.87% | 88.17% | 93.19% | 0.358 | 0.257 | 0.367 | 0.579 | 0.553 | [0.136, 0.222] | [0.695, 0.794] | [0.321, 0.396] | [0.336, 0.402] | 252 |
| 9 | google-bert/bert-base-multilingual-uncased | BAAI/bge-reranker-v2-m3 | 17.20% | 59.86% | 69.18% | 74.19% | 82.44% | 0.314 | 0.210 | 0.293 | 0.428 | 0.541 | [0.129, 0.217] | [0.539, 0.659] | [0.273, 0.354] | [0.258, 0.327] | 212 |
| 10 | sentence-transformers/LaBSE | BAAI/bge-reranker-v2-m3 | 16.85% | 56.27% | 67.03% | 74.55% | 79.57% | 0.313 | 0.210 | 0.293 | 0.407 | 0.552 | [0.124, 0.213] | [0.509, 0.627] | [0.265, 0.355] | [0.254, 0.331] | 254 |
| 11 | google-bert/bert-base-multilingual-cased | BAAI/bge-reranker-v2-m3 | 16.49% | 56.63% | 65.23% | 68.10% | 87.10% | 0.291 | 0.204 | 0.280 | 0.412 | 0.535 | [0.125, 0.213] | [0.513, 0.620] | [0.254, 0.335] | [0.247, 0.317] | 243 |
| 12 | google-bert/bert-base-german-cased | BAAI/bge-reranker-v2-m3 | 11.47% | 16.13% | 17.20% | 17.56% | 34.41% | 0.130 | 0.069 | 0.089 | 0.091 | 0.514 | [0.079, 0.158] | [0.118, 0.206] | [0.096, 0.169] | [0.065, 0.116] | 249 |
| 13 | kforth/IfcElement2ConstructionSets | BAAI/bge-reranker-v2-m3 | 8.24% | 20.43% | 21.51% | 21.86% | 40.14% | 0.122 | 0.067 | 0.093 | 0.117 | 0.518 | [0.050, 0.118] | [0.161, 0.253] | [0.089, 0.159] | [0.069, 0.118] | 224 |

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
