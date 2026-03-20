# Bi-Encoder Training Pipeline (BAAI/bge-m3)

Diese Pipeline ist separat von `Evaluation/` und verändert die bestehende Evaluation nicht.
Sie deckt den kompletten Ablauf ab:

- Rohdaten validieren
- Trainingspaare erzeugen
- optional Hard-Negatives aus Fehlerfällen minen
- Fine-Tuning starten
- Modell reproduzierbar evaluieren

## Ziel

- robuste Fine-Tuning-Läufe für `BAAI/bge-m3`
- reproduzierbare Artefakte pro Run
- bessere Top-1-Trefferquote durch Hard-Negatives (False Positives / Near-Misses)

## Komponenten und Datenfluss

1. `Training/validate_training_data.py`
   - prüft Query-/Expected-Dateien oder erzeugte JSONL-Pairs
   - erkennt auch Inkonsistenzen in `hard_negatives`

2. `Training/prepare_training_data.py`
   - erzeugt `(query, positive)`-JSONL aus Query-/Expected-Dateien
   - optional mit `--hard-negatives-file` pro Query ergänzbar

3. `Training/mine_hard_negatives.py`
   - liest `details_*.csv` aus der Evaluation
   - extrahiert für Top1-fehlerhafte Queries harte Negatives
   - schreibt ein JSONL-Artefakt für die Datenaufbereitung

4. `Training/train_bge_m3.py`
   - trainiert mit `MultipleNegativesRankingLoss`
   - unterstützt 2-Text (`query, positive`) und 3-Text (`query, positive, hard_negative`) Beispiele
   - Modi: `off`, `fallback`, `strict`

5. `Training/run_training_pipeline.py`
   - orchestriert `validate -> prepare -> validate -> train`
   - reicht alle Hard-Negative-Optionen durch

6. `Training/run_single_model_evaluation.py`
   - evaluiert ein einzelnes Modell reproduzierbar
   - erzeugt deterministische `summary/details/overview/report`-Artefakte

## Voraussetzungen

- Python-Umgebung mit Paketen aus `requirements.txt`
- Für GPU-Training: kompatibles CUDA + PyTorch

### Wichtiger Hinweis für Windows/PowerShell

Script-Aufrufe wie `Training/run_single_model_evaluation.py ...` können den falschen Python-Interpreter verwenden.
Empfehlung: immer explizit mit dem venv-Interpreter starten.

Beispiel:

```powershell
c:/Users/wpx619/.AAA_Python_MTH/ifc-kbob-ai-matcher/.venv/Scripts/python Training/run_training_pipeline.py ...
```

## Schnellstart (ohne Hard-Negatives)

```powershell
python Training/run_training_pipeline.py `
  --query-file Evaluation/exports/queries/list_1_queries_with_ifc.txt `
  --expected-file Evaluation/expected_material/list_1_expected_mit-ohne_ifc.txt `
  --base-model BAAI/bge-m3 `
  --output-dir Training/artifacts/models/bge-m3-finetuned `
  --epochs 2 `
  --batch-size 8 `
  --lr 2e-5 `
  --max-length 512 `
  --device cuda `
  --fp16 `
  --deduplicate
```

Das beste Modell (gemäss Dev-Evaluator) wird im `--output-dir` gespeichert.
Zusätzlich werden standardmässig Epochen-Checkpoints unter `<output-dir>/epochs/epoch-01...` abgelegt.

## Hard-Negative Workflow (empfohlen)

Dieser Ablauf ist für deutliche Hit@1-Verbesserungen gedacht.

### Schritt 1: Details-CSV aus bestehender Evaluation erzeugen

Falls bereits vorhanden, direkt mit Schritt 2 starten.

```powershell
python Training/run_single_model_evaluation.py `
  --model models/BAAI/bge-m3 `
  --query-file Evaluation\exports\queries\generated_queries_without_exposure.txt `
  --expected-file Evaluation\exports\queries\mapping_generated_queries_without_exposure.txt `
  --cross-encoder-model "" `
  --device auto `
  --run-label baseline `
  --output-dir Training/outputs
```


### Schritt 2: Hard-Negatives minen

```powershell
python Training/mine_hard_negatives.py `
  --details-file Training/outputs/details_baseline_bge-m3-dfabd00b_generated_queries_wi-944514b4_no-reranker-7521044b.csv `
  --out Training/artifacts/hard_negatives_from_latest_eval.jsonl `
  --max-expected-rank 10 `
  --max-negatives-per-query 3
```

### Schritt 3: Pipeline mit Hard-Negatives trainieren

```powershell
python Training/run_training_pipeline.py `
  --query-file Evaluation\exports\queries\generated_queries_without_exposure.txt `
  --expected-file Evaluation\exports\queries\mapping_generated_queries_without_exposure.txt `
  --base-model BAAI/bge-m3 `
  --pairs-out Training/artifacts/training_pairs_with_hard_negatives.jsonl `
  --hard-negatives-file Training/artifacts/hard_negatives_from_latest_eval.jsonl `
  --hard-negative-mode strict `
  --hard-negative-selection random `
  --output-dir Training/artifacts/models/Hygroskopisch/bge-m3-finetuned-mnrl-hn `
  --deduplicate `
  --epochs 5 `
  --batch-size 6 `
  --lr 1.5e-5 `
  --fp16 `
  --max-length 128 `
  --max-per-positive 30
```

### Schritt 4: Modell erneut evaluieren (bi-encoder only)

```powershell
python Training/run_single_model_evaluation.py `
  --model Training/artifacts/models/bge-m3-hn-phase1 `
  --query-file Evaluation/exports/queries/ifcentity_predefined_material_strength.txt `
  --expected-file Evaluation/expected_material/expected.txt `
  --cross-encoder-model "" `
  --device auto `
  --run-label hn_phase1 `
  --output-dir Training/outputs
```

## Hard-Negative Modi im Trainer

- `--hard-negative-mode off`
  - ignoriert `hard_negatives`
  - trainiert klassisch mit `(query, positive)`

- `--hard-negative-mode fallback` (Default)
  - nutzt vorhandene `hard_negatives`
  - füllt fehlende Negatives pro Sample robust auf
  - guter Startmodus für gemischte Datensätze

- `--hard-negative-mode strict`
  - trainiert nur auf Datensätzen mit echten `hard_negatives`
  - Samples ohne Hard-Negatives werden verworfen
  - sinnvoll für fokussierte Feinschärfung

Auswahl bei mehreren Negatives pro Query:

- `--hard-negative-selection first`: nimmt das erste Negative (deterministisch)
- `--hard-negative-selection random`: zufällige Auswahl (Seed-gesteuert)

## Wichtige Pipeline-Parameter

- `--deduplicate`
  - entfernt identische `(query, positive)`-Paare

- `--max-per-positive N`
  - begrenzt Überrepräsentation häufiger Positives

- `--dev-ratio`
  - Anteil der Queries für Dev-Evaluator während des Trainings

- `--max-length`
  - Token-Limit pro Text
  - bei OOM häufig auf 256 reduzieren

- `--save-each-epoch`
  - speichert Epochenmodelle unter `<output-dir>/epochs/`

## Manuelle Einzel-Schritte (optional)

```powershell
python Training/validate_training_data.py `
  --query-file Evaluation/exports/queries/list_1_queries_with_ifc.txt `
  --expected-file Evaluation/expected_material/list_1_expected_mit-ohne_ifc.txt

python Training/prepare_training_data.py `
  --query-file Evaluation/exports/queries/list_1_queries_with_ifc.txt `
  --expected-file Evaluation/expected_material/list_1_expected_mit-ohne_ifc.txt `
  --out Training/artifacts/training_pairs.jsonl `
  --deduplicate

python Training/validate_training_data.py --pairs-file Training/artifacts/training_pairs.jsonl

python Training/train_bge_m3.py `
  --train-file Training/artifacts/training_pairs.jsonl `
  --base-model BAAI/bge-m3 `
  --output-dir Training/artifacts/models/bge-m3-finetuned `
  --epochs 2 `
  --batch-size 8 `
  --lr 2e-5 `
  --max-length 512 `
  --dev-ratio 0.1 `
  --device cuda `
  --fp16
```

## Artefakte pro Trainingslauf

Im Modell-Output entstehen u. a.:

- trainiertes Modell
- `run_metadata.json` mit Run-ID, Hyperparametern und Hard-Negative-Statistiken
- optional Epochen-Checkpoints in `epochs/`

Im Evaluations-Output entstehen u. a.:

- `summary_<run-label>_<model>_<query>_<ce>.csv`
- `details_<run-label>_<model>_<query>_<ce>.csv`
- `overview_<run-label>_<model>_<query>_<ce>.svg`
- `evaluation_report_<run-label>_<model>_<query>_<ce>.md`
- `overview_single_latest.svg`
- `evaluation_report_single_latest.md`

## Übergabe an bestehende Evaluation

Die bestehende Evaluation nutzt aktuell den lokalen Modellpfad `models/BAAI/bge-m3`.

Wenn du dein Fine-Tune-Modell dort vergleichen willst:

```powershell
Rename-Item models/BAAI/bge-m3 models/BAAI/bge-m3.base -ErrorAction SilentlyContinue
Copy-Item Training/artifacts/models/bge-m3-finetuned models/BAAI/bge-m3 -Recurse -Force

python Evaluation/run_evaluation_pipeline.py `
  --query-source Evaluation/exports/queries/list_1_queries_with_ifc.txt `
  --expected-file Evaluation/expected_material/list_1_expected_mit-ohne_ifc.txt `
  --cross-encoder-model BAAI/bge-reranker-v2-m3 `
  --rerank-top-n 30
```

Optional zurücksetzen:

```powershell
Remove-Item models/BAAI/bge-m3 -Recurse -Force
Rename-Item models/BAAI/bge-m3.base bge-m3
```

## Training mit Queries und Zuordnungen aus dem Dashboard

```powershell
python Training/run_training_pipeline.py `
  --query-file Training/data/dashboard_training_queries.txt `
  --expected-file Training/data/dashboard_training_expected.txt `
  --base-model Training/artifacts/models/bge-m3-stage2-real-queries/epochs/epoch-03 `
  --output-dir Training/artifacts/models/bge-m3-finetuned-dashboard `
  --deduplicate --max-per-positive 30 `
  --epochs 3
```

## Qualitäts- und Reproduzierbarkeits-Checks

- Eval-Set vor Vergleichsläufen fixieren (gleiche Query-/Expected-Datei)
- Run-ID und Parameter pro Lauf dokumentieren (`run_metadata.json`)
- bei Hard-Negative-Experimenten immer Baseline vs. neuer Run im selben Setup vergleichen
- zusätzlich zu Hit@1 auch Hit@10/MRR/nDCG beobachten

## Troubleshooting

- `ModuleNotFoundError` unter Windows:
  - meist falscher Interpreter
  - venv-Python explizit verwenden

- CUDA OOM:
  - `--batch-size` reduzieren (z. B. 4)
  - `--max-length` senken (z. B. 256)
  - testweise `--device cpu`

- zu wenige Hard-Negatives:
  - `mine_hard_negatives.py` mit höherem `--max-expected-rank` ausführen
  - `--max-negatives-per-query` erhöhen
  - für robuste erste Runs `--hard-negative-mode fallback` nutzen
