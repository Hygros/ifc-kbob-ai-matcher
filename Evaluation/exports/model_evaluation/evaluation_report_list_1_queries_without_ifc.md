## Evaluation Report

Generated: 2026-02-23 11:15:02

### Inputs
- Summary CSV: `summary_list_1_queries_without_ifc.csv`
- Details CSV: `details_list_1_queries_without_ifc.csv`

### Overview
![Model overview](overview_list_1_queries_without_ifc.svg)

### Leaderboard

| Rank | Model | Top1 | Top5 | Top10 | MRR | MAP@10 | nDCG@10 | Cov@95% (margin) | Avg expected score | Top1 errors |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | intfloat/multilingual-e5-base | 25.0% | 25.0% | 25.0% | 0.269 | 0.250 | 0.250 | 17.9% | 0.824 | 21 |
| 2 | google/embeddinggemma-300m | 25.0% | 32.1% | 42.9% | 0.300 | 0.273 | 0.308 | 7.1% | 0.496 | 21 |
| 3 | sentence-transformers/LaBSE | 21.4% | 25.0% | 32.1% | 0.271 | 0.241 | 0.259 | 10.7% | 0.353 | 22 |
| 4 | intfloat/multilingual-e5-large | 21.4% | 28.6% | 28.6% | 0.269 | 0.238 | 0.254 | 10.7% | 0.833 | 22 |
| 5 | sentence-transformers/distiluse-base-multilingual-cased-v2 | 21.4% | 35.7% | 35.7% | 0.274 | 0.260 | 0.285 | 7.1% | 0.291 | 22 |
| 6 | BAAI/bge-m3 | 17.9% | 32.1% | 39.3% | 0.277 | 0.249 | 0.285 | 10.7% | 0.494 | 23 |
| 7 | google-bert/bert-base-multilingual-uncased | 7.1% | 25.0% | 42.9% | 0.148 | 0.121 | 0.196 | 0.0% | 0.615 | 26 |
| 8 | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 7.1% | 17.9% | 21.4% | 0.134 | 0.118 | 0.137 | 0.0% | 0.474 | 26 |
| 9 | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 3.6% | 32.1% | 39.3% | 0.176 | 0.167 | 0.223 | 0.0% | 0.390 | 27 |
| 10 | google-bert/bert-base-multilingual-cased | 3.6% | 25.0% | 25.0% | 0.148 | 0.125 | 0.157 | 0.0% | 0.581 | 27 |
| 11 | google-bert/bert-base-german-cased | 0.0% | 21.4% | 25.0% | 0.089 | 0.081 | 0.122 | 0.0% | 0.773 | 28 |
| 12 | kforth/IfcMaterial2MP | 0.0% | 10.7% | 14.3% | 0.077 | 0.041 | 0.066 | 0.0% | 0.488 | 28 |
| 13 | kforth/IfcElement2ConstructionSets | 0.0% | 7.1% | 10.7% | 0.048 | 0.017 | 0.034 | 0.0% | 0.728 | 28 |

Anzahl Queries: 28

### Hardest Queries
Queries mit den meisten Top1-Fehlern über alle Modelle:

- (13 Fehler) ReinforcingBar Bügel B500B
- (13 Fehler) ReinforcingBar Längsstab B500B
- (13 Fehler) ReinforcingBar SHEAR Bügel B500B
- (13 Fehler) ReinforcingBar MAIN 1. Lage B500B
- (13 Fehler) ReinforcingBar unterer Stab B500B
