# Bi-Encoder Training Pipeline (BAAI/bge-m3)

Diese Pipeline ermöglicht reproduzierbare Fine-Tuning-Läufe für `BAAI/bge-m3`.

## Empfehlung auf einen Blick

1. Standard und Veröffentlichungs-Pfad: **Clean Baseline Runbook** (A-E).
2. Optional danach: **Hard-Negative Folge-Workflow** für einen zweiten Verbesserungs-Run.
3. Wenn genau ein sauberer Lauf benötigt wird: nach Clean Baseline Schritt E stoppen.

Kurzform: **Empfohlen ist zuerst und primär der Clean-Baseline-Pfad.**

## Inhalte der Pipeline

Die Orchestrierung in `Training/run_training_pipeline.py` führt aus:

- `validate -> prepare -> validate -> qa -> manifest -> train`

Wichtige Komponenten:

- `Training/validate_training_data.py`: Validierung von Rohdateien und JSONL-Pairs
- `Training/prepare_training_data.py`: Erstellung von `(query, positive)`-Pairs, optional mit Hard-Negatives
- `Training/run_data_qa_preflight.py`: QA-Gates vor dem Training (STOP/WARN)
- `Training/train_bge_m3.py`: Fine-Tuning mit `MultipleNegativesRankingLoss`
- `Evaluation/run_single_model_evaluation.py`: Reproduzierbare Einzelevaluation eines Modells (dense-only Default, optional Split-Matrix)
- `Training/mine_hard_negatives.py`: Mining harter Negatives aus `details_*.csv` (inkl. automatischer Intra-Family HN)
- `Training/mine_family_hard_negatives.py`: Intra-Family Hard-Negatives aus KBOB-Taxonomie (wird von `mine_hard_negatives.py` aufgerufen)

## Voraussetzungen

- Python-Umgebung mit allen Paketen aus `requirements.txt`
- Optional für GPU: kompatibles CUDA + PyTorch

## Interpreter-Hinweis (wichtig unter Windows)

Direkter Skriptaufruf kann auf Windows den falschen Interpreter verwenden.
Falls nötig, nutze den venv-Interpreter explizit:

```powershell
.\.venv\Scripts\python.exe Training/run_training_pipeline.py ...
```

Alle Beispiele unten verwenden `python ...`.
Wenn dein System mehrere Interpreter hat, ersetze `python` durch deinen venv-Pfad.

## Clean Baseline Runbook (empfohlener Standard)

Ziel: ein einzelner, sauber dokumentierter Baseline-Lauf ohne Sweeps/Ablationen.

### Zielkonfiguration

- `prefix_mode`: `no_prefix` (Default im dense-only BGE-M3 Pfad)
- `hard_negative_mode`: `strict`
- `hard_negative_selection`: `first` (deterministisch)
- `num_hard_negatives`: `1` (Default; K=2 für Multi-HN)
- `qa_preflight`: aktiv
- `split_manifest`: aktiv

### Vor Schritt A: benötigte Eingabedateien

Pflichtdateien für den empfohlenen Standardpfad (strict):

- Train Query TXT: `Training/query_generation/generated_queries/generated_queries.txt`
- Train Expected TXT: `Training/query_generation/generated_queries/mapping_generated_queries.txt`
- Hard-Negatives JSONL: `Training/artifacts/hard_negatives_from_latest_eval.jsonl`

Hinweis zur Herkunft:

- Die Hard-Negatives JSONL wird einmalig aus einer Evaluations-Details-CSV erzeugt (siehe Bootstrap).

### Schritt 0 (einmalig): Hard-Negatives-Datei erzeugen, falls sie fehlt

1. Basis-Modell auf dem Trainings-Set evaluieren, um eine `details_*.csv` zu erzeugen:

```powershell
python Evaluation/run_single_model_evaluation.py `
  --model BAAI/bge-m3 `
  --query-file Training/query_generation/generated_queries/generated_queries.txt `
  --expected-file Training/query_generation/generated_queries/mapping_generated_queries.txt `
  --device auto `
  --run-label bootstrap_hn_source `
  --output-dir Evaluation/outputs/single_model
```

1. Aus dieser `details_*.csv` die Hard-Negatives-JSONL minen:

```powershell
python Training/mine_hard_negatives.py `
  --details-file <PATH_TO_DETAILS_CSV> `
  --out Training/artifacts/hard_negatives_from_latest_eval.jsonl `
  --max-expected-rank 10 `
  --max-negatives-per-query 6 `
  --min-predicted-score 0.0 `
  --cross-query-positive-protection family `
  --include-narrow-wins `
  --cross-family-extras '{"Gussasphalt": ["Dichtungsbahn bituminös"]}' `
  --query-near-jaccard-threshold 0.60
```

Hinweis: Intra-Family Hard-Negatives (aus `mine_family_hard_negatives.py`) werden automatisch dazu­gemined und an die Ausgabe angehängt.
Dafür nutzt das Skript die Standard-Query/Mapping-Dateien und `material_ökobilanz.txt`.
Falls nicht gewünscht, mit `--no-family-hard-negatives` deaktivieren.

Hinweis zu `--cross-family-extras`:

- Erwartet einen JSON-String: `'{"<Positive-Material>": ["<Extra-HN-1>", ...]}'`
- Fügt Hard-Negatives aus **anderen** KBOB-Familien hinzu, die das Intra-Family-Mining nicht erfasst.
- Beispiel: `Gussasphalt` (Familie `#Asphalt`) und `Dichtungsbahn bituminös` (Familie `#Dichtung_Bituminös`) liegen in verschiedenen Familien — ohne diesen Parameter würde `Dichtungsbahn bituminös` nie als HN für Pavement-Bitumen-Queries gemined.
- Weglassen oder `''` übergeben, wenn keine cross-family Extras benötigt werden.

Empfehlung zur Cross-Query-Protection:

- Bei vielen Queries mit kleinem Positive-Vokabular `family` nutzen (robuster Standard).
- `global` nur verwenden, wenn Positives stark disjunkt sind; sonst werden zu viele Kandidaten entfernt.
- `off` nur für bewusst permissive Experimente (in Kombination mit angepassten QA-Grenzwerten).

Danach mit Schritt A starten.

Alternative (nicht der strikte Standardpfad):

- Ohne vorhandene Hard-Negatives-Datei kannst du `--hard-negative-mode fallback` nutzen und `--hard-negatives-file` weglassen.
- Für den in diesem Runbook empfohlenen strict-Baseline-Lauf ist die JSONL jedoch Pflicht.

### Schritt A: Preflight ohne Training

```powershell
python Training/run_training_pipeline.py `
  --query-file Training/query_generation/generated_queries/generated_queries.txt `
  --expected-file Training/query_generation/generated_queries/mapping_generated_queries.txt `
  --base-model BAAI/bge-m3 `
  --pairs-out Training/artifacts/training_pairs_baseline_clean_run.jsonl `
  --output-dir Training/artifacts/models/baseline_clean_run `
  --hard-negatives-file Training/artifacts/hard_negatives_from_latest_eval.jsonl `
  --hard-negative-mode strict `
  --hard-negative-selection first `
  --qa-eval-query-file Training/data/dashboard_training_queries.txt `
  --qa-eval-expected-file Training/data/dashboard_training_expected.txt `
  --seed 42 `
  --dev-ratio 0.1 `
  --run-id baseline_clean_run_precheck `
  --stop-before-train
```

Preflight muss ohne STOP-Failures enden.

### Schritt B: Baseline trainieren

```powershell
python Training/run_training_pipeline.py `
  --query-file Training/query_generation/generated_queries/generated_queries.txt `
  --expected-file Training/query_generation/generated_queries/mapping_generated_queries.txt `
  --base-model BAAI/bge-m3 `
  --pairs-out Training/artifacts/training_pairs_baseline_clean_run.jsonl `
  --output-dir Training/artifacts/models/baseline_clean_run `
  --hard-negatives-file Training/artifacts/hard_negatives_from_latest_eval.jsonl `
  --hard-negative-mode strict `
  --hard-negative-selection first `
  --qa-eval-query-file Training/data/dashboard_training_queries.txt `
  --qa-eval-expected-file Training/data/dashboard_training_expected.txt `
  --seed 42 `
  --dev-ratio 0.1 `
  --run-id baseline_clean_run
```

Betriebshinweis:

- Während dieses Laufs keine weiteren Experimente starten.
- Monitoring in einem separaten Terminal durchführen.

### Schritt C: Baseline evaluieren

```powershell
python Evaluation/run_single_model_evaluation.py `
  --model Training/artifacts/models/baseline_clean_run `
  --query-file Training/query_generation/generated_queries/generated_queries.txt `
  --expected-file Training/query_generation/generated_queries/mapping_generated_queries.txt `
  --device auto `
  --run-label baseline_clean_run `
  --output-dir Evaluation/outputs/single_model
```

### Schritt D: Pflicht-Artefakte prüfen

Diese Artefakte müssen vorhanden sein:

- `Training/artifacts/models/baseline_clean_run/run_metadata.json`
- `Training/outputs/manifests/split_manifest_baseline_clean_run.json`
- `Training/outputs/qa/qa_report_baseline_clean_run.json`
- `Training/outputs/qa/qa_gate_baseline_clean_run.csv`
- Evaluationsdateien mit Label `baseline_clean_run` in `Evaluation/outputs/single_model/`

### Schritt E: Kennzahlen dokumentieren

Mindestens dokumentieren:

- `Hit@1`
- `MRR@10`
- `Recall@10`
- Trainingsdauer und Device
- finale Train/Dev Pair- und Query-Anzahlen
- `records_with_hard_negatives`
- `fallback_negatives_used`
- `dropped_unusable`

## Optional: Hard-Negative Folge-Workflow (zweiter Lauf)

Nur ausführen, wenn die Baseline bereits abgeschlossen und dokumentiert ist.

### Schritt 1: Details-CSV für Mining erzeugen

```powershell
python Evaluation/run_single_model_evaluation.py `
  --model Training/artifacts/models/baseline_clean_run `
  --query-file Training/query_generation/generated_queries/generated_queries.txt `
  --expected-file Training/query_generation/generated_queries/mapping_generated_queries.txt `
  --device auto `
  --run-label baseline_for_hn_mining `
  --output-dir Evaluation/outputs/single_model
```

### Schritt 2: Hard-Negatives minen

```powershell
python Training/mine_hard_negatives.py `
  --details-file <PATH_TO_DETAILS_CSV> `
  --out Training/artifacts/hard_negatives_from_latest_eval.jsonl `
  --max-expected-rank 10 `
  --max-negatives-per-query 3 `
  --min-predicted-score 0.0 `
  --cross-query-positive-protection family `
  --cross-family-extras '{"Gussasphalt": ["Dichtungsbahn bituminös"]}' `
  --query-near-jaccard-threshold 0.60
```

Hinweis: `<PATH_TO_DETAILS_CSV>` ist die passende `details_*.csv` aus `Evaluation/outputs/single_model/`.
Intra-Family Hard-Negatives werden automatisch mit erzeugt (siehe Schritt 0).
`--cross-family-extras` wie in Schritt 0 anpassen, falls weitere cross-family Paare benötigt werden.

### Schritt 3: Folge-Run mit Hard-Negatives trainieren

```powershell
python Training/run_training_pipeline.py `
  --query-file Training/query_generation/generated_queries/generated_queries.txt `
  --expected-file Training/query_generation/generated_queries/mapping_generated_queries.txt `
  --base-model BAAI/bge-m3 `
  --pairs-out Training/artifacts/training_pairs_mnrl_hn_run.jsonl `
  --output-dir Training/artifacts/models/mnrl_hn_run `
  --hard-negatives-file Training/artifacts/hard_negatives_from_latest_eval.jsonl `
  --hard-negative-mode strict `
  --hard-negative-selection first `
  --qa-eval-query-file Training/data/dashboard_training_queries.txt `
  --qa-eval-expected-file Training/data/dashboard_training_expected.txt `
  --seed 42 `
  --dev-ratio 0.1 `
  --run-id mnrl_hn_run
```

### Schritt 4: Folge-Run evaluieren

```powershell
python Evaluation/run_single_model_evaluation.py `
  --model Training/artifacts/models/mnrl_hn_run `
  --query-file Training/query_generation/generated_queries/generated_queries.txt `
  --expected-file Training/query_generation/generated_queries/mapping_generated_queries.txt `
  --device auto `
  --run-label mnrl_hn_run `
  --output-dir Evaluation/outputs/single_model
```

## Wichtige Parameter von run_training_pipeline.py

- `--run-id`: stabile Artefaktnamen pro Lauf
- `--hard-negative-mode`: `off|fallback|strict`
- `--hard-negative-selection`: `first|random|random_preselected`
- `--num-hard-negatives`: Anzahl Hard-Negatives pro Record (K, Default `1`)
- `--qa-preflight` / `--no-qa-preflight`
- `--qa-fail-on-stop` / `--no-qa-fail-on-stop`
- `--qa-fn-cross-query-scope`: `off|family|global` (Default: `family`)
- `--qa-fn-cross-query-near-jaccard-threshold`: Schwelle fuer Scope `family` (Default: `0.60`)
- `--qa-eval-query-file`, `--qa-eval-expected-file`
- `--split-manifest-dir`: Ziel für `split_manifest_<run_id>.json`
- `--rule-policy-file`: optionale Policy-Datei für Rule-Hash/Traceability
- `--stop-before-train`: Preflight bis inkl. QA/Manifest, ohne Training

## Artefakte pro Lauf

### Modell-Artefakte

- trainiertes Modell in `--output-dir`
- `run_metadata.json`
- optional Checkpoints in `<output-dir>/epochs/`

### Pipeline-Artefakte

- `Training/outputs/qa/qa_report_<run_id>.json`
- `Training/outputs/qa/qa_gate_<run_id>.csv`
- `Training/outputs/manifests/split_manifest_<run_id>.json`

### Evaluations-Artefakte

- `summary_<run_label>_*.csv`
- `details_<run_label>_*.csv`
- `overview_<run_label>_*.svg`
- `evaluation_report_<run_label>_*.md`
- `split_eval_matrix_<run_label>_*.csv`
- `split_eval_matrix_<run_label>_*.json`
- `overview_single_latest.svg`
- `evaluation_report_single_latest.md`

Hinweis zur Einzelevaluation:

- `Evaluation/run_single_model_evaluation.py` läuft standardmäßig ohne Reranking.
- Reranking nur bei Bedarf explizit setzen, z. B. `--cross-encoder-model BAAI/bge-reranker-v2-m3 --rerank-top-n 30`.

## Manuelle Einzel-Schritte (optional)

```powershell
python Training/validate_training_data.py `
  --query-file Training/query_generation/generated_queries/generated_queries.txt `
  --expected-file Training/query_generation/generated_queries/mapping_generated_queries.txt

python Training/prepare_training_data.py `
  --query-file Training/query_generation/generated_queries/generated_queries.txt `
  --expected-file Training/query_generation/generated_queries/mapping_generated_queries.txt `
  --out Training/artifacts/training_pairs.jsonl `
  --seed 42 `
  --deduplicate

python Training/validate_training_data.py `
  --pairs-file Training/artifacts/training_pairs_manual.jsonl

python Training/train_bge_m3.py `
  --train-file Training/artifacts/training_pairs_manual.jsonl `
  --base-model BAAI/bge-m3 `
  --output-dir Training/artifacts/models/manual_run `
  --epochs 2 `
  --batch-size 8 `
  --lr 2e-5 `
  --max-length 512 `
  --dev-ratio 0.1 `
  --seed 42 `
  --device auto `
  --hard-negative-mode fallback `
  --hard-negative-selection first
```

## Übergabe an bestehende Evaluation (optional)

Die bestehende Evaluation verwendet standardmässig `models/BAAI/bge-m3`.
Wenn du ein Fine-Tune-Modell dort vergleichen willst:

```powershell
Rename-Item models/BAAI/bge-m3 models/BAAI/bge-m3.base -ErrorAction SilentlyContinue
Copy-Item Training/artifacts/models/baseline_clean_run models/BAAI/bge-m3 -Recurse -Force

python Evaluation/run_evaluation_pipeline.py `
  --query-source Evaluation/outputs/queries/list_1_queries_with_ifc.txt `
  --expected-file Evaluation/ground_truth/list_1_expected_mit-ohne_ifc.txt `
  --cross-encoder-model BAAI/bge-reranker-v2-m3 `
  --rerank-top-n 30
```

Optional rückgängig machen:

```powershell
Remove-Item models/BAAI/bge-m3 -Recurse -Force
Rename-Item models/BAAI/bge-m3.base bge-m3
```

## Troubleshooting

- `ModuleNotFoundError` unter Windows:
  - Meist falscher Interpreter.
  - Explizit mit `.\.venv\Scripts\python.exe` starten.

- Strict-HN bricht vor Training ab:
  - `hn_viability_<run_id>.json` prüfen.
  - `strict` nur verwenden, wenn `strict_hn_usable=true`.

- QA STOP-Failure blockiert Training:
  - `qa_gate_<run_id>.csv` lesen.
  - Ursache beheben und genau einen neuen Lauf starten.

- QA STOP-Failure `stop_false_negative_rate_cross_query` / `stop_false_negative_count_any_scope`:
  - Hard-Negatives neu minen mit `--cross-query-positive-protection family`.
  - Optional `--query-near-jaccard-threshold` feiner einstellen (z. B. 0.55 bis 0.70).
  - Bei Many-Query/Few-Label-Daten die QA-Stop-Grenzwerte (`--qa-fn-cross-query-stop-rate`, `--qa-fn-any-scope-stop-count`) datenrealistisch konfigurieren.

- CUDA OOM:
  - `--batch-size` reduzieren (z. B. 8 -> 4).
  - `--max-length` reduzieren (z. B. 512 -> 256).
  - Testweise `--device cpu`.


