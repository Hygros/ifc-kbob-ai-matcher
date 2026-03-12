# Bi-Encoder Training Pipeline (BAAI/bge-m3)

Diese Pipeline ist **separat** von `Evaluation/` und verändert die bestehende Evaluation nicht.

## Ziel

- Trainingspaare aus Query/Expected erzeugen
- `BAAI/bge-m3` per Fine-Tuning weitertrainieren
- Modellartefakt lokal speichern

## Voraussetzungen

- Python-Umgebung mit `requirements.txt`
- Für GPU-Training: kompatibles CUDA + PyTorch

## Schnellstart

1) Rohdaten validieren + Trainingspipeline starten:

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

1) Einzelne Schritte manuell (optional):

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

## Übergabe an bestehende Evaluation

Die Evaluation nutzt aktuell den lokalen Modellpfad `models/BAAI/bge-m3`.

Wenn du dein Fine-Tune-Modell mit der **bestehenden** Evaluation vergleichen willst:

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

## Schneller Vergleich nur für ein Modell

Wenn du nicht alle Modelle durchlaufen lassen willst, nutze den Single-Model-Runner:

```powershell
python Training/run_single_model_evaluation.py `
  --model BAAI/bge-m3 `
  --query-file Evaluation/exports/queries/ifcentity_material_strength.txt `
  --expected-file Evaluation/expected_material/expected.txt `
  --cross-encoder-model BAAI/bge-reranker-v2-m3 `
  --rerank-top-n 30 `
  --device auto `
  --run-label finetuned `
  --output-dir Training/outputs
```

Die Ausgabe wird deterministisch gespeichert als `summary_<run-label>_<model>_<query>_<ce>.csv`, `details_<run-label>_<model>_<query>_<ce>.csv`, `overview_<run-label>_<model>_<query>_<ce>.svg` und `evaluation_report_<run-label>_<model>_<query>_<ce>.md`.

Zusätzlich werden immer `overview_single_latest.svg` und `evaluation_report_single_latest.md` aktualisiert.

## Hinweise

- Bei CUDA-Fehlern oder OOM: `--batch-size` reduzieren (z. B. 4), `--max-length` auf 256 setzen oder `--device cpu` nutzen.
- Das Training nutzt `MultipleNegativesRankingLoss` auf `(query, positive)` Paaren.
- Expected-Zeilen unterstützen `;` oder `|` als Trenner und optional `::gewicht`.

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
