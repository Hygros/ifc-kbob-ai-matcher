## Evaluation Report

Generated: 2026-02-22 16:40:20

### Inputs
- Summary CSV: `summary_list_1_queries_without_ifc.csv`
- Details CSV: `details_list_1_queries_without_ifc.csv`

### Metric Meaning
- Top1 Accuracy: Anteil Queries, bei denen das richtige Material auf Rang 1 steht.

  Formel: $\mathrm{Top1} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}(\mathrm{Rang}_i = 1)$
Beispiel: 0.8 bedeutet 8 von 10 direkt korrekt.

- Top5 Accuracy: Anteil Queries, bei denen das richtige Material irgendwo in den Top 5 steht.

  Formel: $\mathrm{Top5} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}(\mathrm{Rang}_i \leq 5)$

- Top10 Accuracy: Anteil Queries, bei denen das richtige Material irgendwo in den Top 10 steht.

  Formel: $\mathrm{Top10} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}(\mathrm{Rang}_i \leq 10)$

- MRR (Mean Reciprocal Rank): bewertet den Rang des richtigen Treffers (höher = besser).
  Formel: $\mathrm{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\mathrm{Rang}_i}$
Dabei gilt: Rang 1 zählt voll, Rang 2 nur 0.5, Rang 3 nur 0.33, Rang 4: 0.25, Rang 5: 0.2 etc.
- Avg expected score: mittlerer Similarity-Score des korrekten Materials (nur als internes Vertrauenssignal pro Modell, nicht perfekt modellübergreifend vergleichbar).
  Formel: $\mathrm{AvgExpectedScore} = \frac{1}{N} \sum_{i=1}^{N} \max_{j \in E_i} s_{ij}$
Dabei ist $E_i$ die Menge der passenden Expected-Kandidaten für Query $i$, $s_{ij}$ der Similarity-Score zwischen Query $i$ und Kandidat $j$, und $N$ die Anzahl der Queries.
Beispiel: Bei 3 Queries mit besten Expected-Scores 0.82, 0.67 und 0.91 gilt: $(0.82 + 0.67 + 0.91) / 3 = 0.80$.
- Expected score (pro Query): höchster Similarity-Score unter den zum erwarteten Material gehörenden Kandidaten.
  Formel: $\mathrm{ExpectedScore}_i = \max_{j \in E_i} s_{ij}$
Dabei ist $i$ die Query, $E_i$ die Menge der passenden Expected-Kandidaten, und $s_{ij}$ der Similarity-Score zwischen Query $i$ und Kandidat $j$.
Die Score-Höhe allein ist nicht das wichtigste Kriterium; Ranking-Metriken (Top1/Top5/Top10/MRR) sind für Zuordnung robuster.

### Overview
![Model overview](overview_list_1_queries_without_ifc.svg)

### Leaderboard

| Rank | Model | Cases | Top1 | Top5 | Top10 | MRR | Avg expected score | Top1 errors |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | google/embeddinggemma-300m | 28 | 21.4% | 32.1% | 42.9% | 0.282 | 0.496 | 22 |
| 2 | intfloat/multilingual-e5-base | 28 | 21.4% | 25.0% | 25.0% | 0.245 | 0.823 | 22 |
| 3 | sentence-transformers/LaBSE | 28 | 17.9% | 25.0% | 32.1% | 0.251 | 0.352 | 23 |
| 4 | BAAI/bge-m3 | 28 | 14.3% | 28.6% | 39.3% | 0.247 | 0.492 | 24 |
| 5 | sentence-transformers/distiluse-base-multilingual-cased-v2 | 28 | 14.3% | 35.7% | 35.7% | 0.233 | 0.289 | 24 |
| 6 | intfloat/multilingual-e5-large | 28 | 14.3% | 28.6% | 28.6% | 0.228 | 0.833 | 24 |
| 7 | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 28 | 3.6% | 32.1% | 39.3% | 0.165 | 0.387 | 27 |
| 8 | google-bert/bert-base-multilingual-uncased | 28 | 3.6% | 21.4% | 42.9% | 0.117 | 0.614 | 27 |
| 9 | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 28 | 3.6% | 17.9% | 17.9% | 0.110 | 0.472 | 27 |
| 10 | google-bert/bert-base-multilingual-cased | 28 | 0.0% | 25.0% | 25.0% | 0.124 | 0.581 | 28 |
| 11 | google-bert/bert-base-german-cased | 28 | 0.0% | 17.9% | 21.4% | 0.084 | 0.772 | 28 |
| 12 | kforth/IfcMaterial2MP | 28 | 0.0% | 7.1% | 14.3% | 0.069 | 0.485 | 28 |
| 13 | kforth/IfcElement2ConstructionSets | 28 | 0.0% | 0.0% | 7.1% | 0.035 | 0.728 | 28 |

### Hardest Queries
Queries mit den meisten Top1-Fehlern über alle Modelle:

- (13 Fehler) ReinforcingBar Bügel B500B
- (13 Fehler) ReinforcingBar Längsstab B500B
- (13 Fehler) ReinforcingBar SHEAR Bügel B500B
- (13 Fehler) ReinforcingBar MAIN 1. Lage B500B
- (13 Fehler) ReinforcingBar unterer Stab B500B
