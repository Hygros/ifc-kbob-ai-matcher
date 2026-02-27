# Metric Explanations

Diese Datei beschreibt die Metriken der aktuellen Evaluations-Pipeline.

## Kontext und Notation

- Für jede Query gibt es eine Rangliste von Kandidaten (höherer Similarity-Score = besserer Rang).
- Relevanz ist aktuell **binär**: relevant oder nicht relevant.
- $R_i$: Menge der relevanten Kandidaten für Query $i$.
- $N$: Anzahl Queries.
- $rank_i$: Rang des ersten relevanten Treffers in Query $i$.

Beispiel-Setup (für alle Rechnungen unten):

- Query 1 relevante Ränge: $[1,3,8]$
- Query 2 relevante Ränge: $[2,12]$
- Query 3 relevante Ränge: $[7,11,13,18]$
- Query 4 relevante Ränge: $[6]$

Damit gelten für den ersten relevanten Rang: $rank = [1,2,7,6]$.

## Ranking-Metriken

- **Cases**
  - Definition: Anzahl ausgewerteter Queries.
  - Beispiel: $N=4$.

- **Hit@1**
  - Definition: Anteil Queries mit mindestens einem relevanten Treffer in Top 1.
  - Formel: $\mathrm{Hit@1}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(rank_i\le1)$
  - Beispiel: $\frac{1+0+0+0}{4}=0.25$.

- **Hit@5**
  - Definition: Anteil Queries mit mindestens einem relevanten Treffer in Top 5.
  - Formel: $\mathrm{Hit@5}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(rank_i\le5)$
  - Beispiel: $\frac{1+1+0+0}{4}=0.50$.

- **Hit@10**
  - Definition: Anteil Queries mit mindestens einem relevanten Treffer in Top 10.
  - Formel: $\mathrm{Hit@10}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(rank_i\le10)$
  - Beispiel: $\frac{1+1+1+1}{4}=1.00$.

- **MRR@10**
  - Definition: Mean Reciprocal Rank, abgeschnitten auf Top 10.
  - Formel: $\mathrm{MRR@10}=\frac{1}{N}\sum_{i=1}^{N} rr_i$, mit
    $rr_i=\frac{1}{rank_i}$ falls $rank_i\le10$, sonst $0$.
  - Beispiel: $\frac{1 + 0.5 + 1/7 + 1/6}{4}=0.4524$.

- **MAP@10**
  - Definition: Mean Average Precision bis Rang 10.
  - Für Query $i$:
    $\mathrm{AP@10}_i=\frac{1}{|R_i|}\sum_{k=1}^{10} P_i(k)\cdot rel_i(k)$,
    mit $rel_i(k)\in\{0,1\}$.
  - Gesamt: $\mathrm{MAP@10}=\frac{1}{N}\sum_i \mathrm{AP@10}_i$.
  - Beispiel:
    - $\mathrm{AP@10}_{Q1}=\frac{1 + 2/3 + 3/8}{3}=0.6806$
    - $\mathrm{AP@10}_{Q2}=\frac{1/2}{2}=0.2500$
    - $\mathrm{AP@10}_{Q3}=\frac{1/7}{4}=0.0357$
    - $\mathrm{AP@10}_{Q4}=\frac{1/6}{1}=0.1667$
    - $\mathrm{MAP@10}=\frac{0.6806+0.2500+0.0357+0.1667}{4}=0.2833$.

- **nDCG@10 (binär)**
  - Definition: normalisierte DCG bis Rang 10 mit binärer Relevanz.
  - Formel:
    $\mathrm{DCG@10}_i=\sum_{k=1}^{10}\frac{rel_i(k)}{\log_2(k+1)}$,
    $\mathrm{nDCG@10}_i=\frac{\mathrm{DCG@10}_i}{\mathrm{IDCG@10}_i}$,
    $\mathrm{nDCG@10}=\frac{1}{N}\sum_i\mathrm{nDCG@10}_i$.
  - Beispiel:
    - $Q1: \mathrm{nDCG@10}=0.8520$
    - $Q2: \mathrm{nDCG@10}=0.3869$
    - $Q3: \mathrm{nDCG@10}=0.1301$
    - $Q4: \mathrm{nDCG@10}=0.3562$
    - Mittel: $\frac{0.8520+0.3869+0.1301+0.3562}{4}=0.4313$.

- **Recall@10**
  - Definition: Anteil der relevanten Kandidaten, die in Top 10 gefunden wurden.
  - Formel pro Query: $\mathrm{Recall@10}_i=\frac{|R_i\cap Top10_i|}{|R_i|}$,
    Gesamt: $\frac{1}{N}\sum_i \mathrm{Recall@10}_i$.
  - Beispiel:
    - $Q1: 3/3=1.0$
    - $Q2: 1/2=0.5$
    - $Q3: 1/4=0.25$
    - $Q4: 1/1=1.0$
    - Recall@10: $\frac{1+0.5+0.25+1}{4}=0.6875$.

## Unsicherheitsangaben (95%-Bootstrap-CIs)

Für einige Kennzahlen werden 95%-Konfidenzintervalle per Bootstrap auf dem Testset berechnet:

- **Hit@1 95% CI** (`hit@1_ci_low`, `hit@1_ci_high`)
- **Hit@10 95% CI** (`hit@10_ci_low`, `hit@10_ci_high`)
- **MRR@10 95% CI** (`mrr@10_ci_low`, `mrr@10_ci_high`)
- **nDCG@10 95% CI** (`ndcg@10_ci_low`, `ndcg@10_ci_high`)

Beispielinterpretation:

- Hit@1 = $0.62$ mit CI $[0.56, 0.68]$ bedeutet:
  Unter Bootstrap-Resampling liegt der plausible Bereich der Hit@1-Schätzung
  bei etwa 56% bis 68%.

## Weitere Report-Spalten

- **Avg expected score**
  - Aktuell: Mittelwert des Top1-Scores auf dem ausgewerteten Testteil.
  - Beispiel: Scores $[0.81, 0.74, 0.69, 0.88] \Rightarrow 0.78$.

- **Top1 errors**
  - Im Report als Fehlersumme aus `details.csv` berechnet.
  - Formel: $Top1Errors = N\cdot(1-Hit@1)$.
  - Beispiel: $N=100$, $Hit@1=0.62 \Rightarrow 38$ Fehler.

## Hinweis zu Modi

- Es gibt nur einen Modus: Ranking-Evaluation auf allen Queries.
