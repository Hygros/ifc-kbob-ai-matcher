# IFC-basierte Ökobilanzierung & Material-Matching

Automatisierte Zuordnung von IFC-Bauelementen zu Ökobilanzdaten (KBOB) mit Sentence-Transformer-basiertem Matching und Berechnung von Umweltindikatoren (UBP21, GWP, Primärenergie).

## Du möchtest das Tool ausprobieren und der Code interessiert dich weniger?
Dann gehe auf diese Seite: [https://hygroskopisch-ifc-kbob-ai-matcher.hf.space/]


## Überblick

Das Projekt besteht aus drei Hauptbereichen und einer gemeinsamen Codebasis:

| Bereich | Zweck |
| --------- | ------- |
| **Dashboard** | Streamlit-App: IFC-Upload, AI-Materialzuordnung, 3D-Viewer, Umweltindikator-Visualisierung |
| **Evaluation** | Evaluation von Bi-Encoder- und Cross-Encoder-Modellen gegen erwartete Materialzuordnungen |
| **Training** | Fine-Tuning von Sentence-Transformer-Modellen (BAAI/bge-m3) mit eigenen Trainingsdaten |
| **core** | Gemeinsam genutzte Module: IFC-Extraktion, SBERT-Matching, UBP-Berechnung |

### Pipeline-Ablauf

```text
IFC-Datei
  → core/ifc_extraction    (Elemente, Materialien, PropertySets → JSONL)
  → core/sbert             (Bi-Encoder-Matching + Cross-Encoder-Reranking gegen KBOB-DB)
  → Dashboard              (Nutzer wählt Zuordnung, UBP-Berechnung, Charts)
```

### SBERT Query-Bildung

Die Bi-Encoder-Queries werden in `core/sbert/sentence_transformer.py` aus folgenden Basisfeldern aufgebaut:

```text
IfcEntity
PredefinedType
Material
Durchmesser
CastingMethod
StrengthClass
```

Das Feld `Name` wird bewusst **nicht** direkt als Rohtext in die Basis-Query übernommen, um Rauschen aus projektspezifischen Benennungen zu reduzieren.

Für Baugrubensicherungs-Fälle wird stattdessen eine gezielte, fuzzy und mehrsprachige Disambiguierung über `Name` angewendet (DE/FR/IT/EN), z. B.:

```text
Schlitzwand | Bohrpfahlwand | Spundwand | Nagelwand | Rühlwand
verankert | gespriesst | auskragend | unverankert
400 | 800
```

Diese erkannten Tokens werden zusätzlich an die Query angehängt.

## Projektstruktur

```text
ifc-kbob-ai-matcher/
│
├── core/                              # Gemeinsam genutzte Module
│   ├── ifc_extraction/                # IFC-Parsing und Element-/Materialextraktion
│   │   ├── ifc_extraction_core.py     #   Kern-Logik: PropertySets, Einheiten, Materialschichten
│   │   ├── ifc_extraction_main.py     #   CLI-Einstiegspunkt (python -m core.ifc_extraction.ifc_extraction_main)
│   │   ├── ifc_material_extract_util.py
│   │   ├── ifc_batch_export_to_csv.py #   Batch-Export ganzer IFC-Ordner → CSV + Analyse-Reports
│   │   └── ifc_reinforcement_relation.py
│   ├── sbert/                         # Sentence-Transformer Matching-Engine
│   │   ├── sentence_transformer.py    #   Bi-Encoder + Cross-Encoder Reranking gegen KBOB
│   │   ├── batch_benchmark.py         #   Batch-Size-Benchmark für optimale Encoding-Performance
│   │   └── cross_encoder.py           #   Standalone Cross-Encoder-Demo
│   ├── calculate_ubp21_per_element.py # UBP/GWP/Energie-Berechnung pro Element
│   └── ifc_units_reader.py            # IFC-Einheiten-Interpretation (SI, Prefixes)
│
├── Dashboard/                         # Streamlit-Webanwendung
│   ├── app_with_viewer.py             #   Haupteinstiegspunkt
│   ├── config.py                      #   Modell-Optionen, Indikator-Definitionen, Schwellwerte
│   ├── domain/
│   │   └── mapping.py                 #   Domain-Logik: Bewehrung, Betonzuordnung, Gruppierung
│   ├── services/
│   │   ├── bootstrap.py               #   App-Initialisierung: Modell-Vorladung, Viewer-Start
│   │   ├── ifc_pipeline.py            #   IFC → JSONL → SBERT Pipeline (subprocess + API)
│   │   ├── kbob_materials.py          #   KBOB-Datenbank-Zugriff
│   │   ├── training_export.py         #   Export manueller Zuordnungen als Trainingsdaten
│   │   ├── ubp.py                     #   UBP-Berechnung und Ergebnis-Merge
│   │   └── viewer.py                  #   3D-IFC-Viewer (ifc-lite) Integration
│   ├── ui/
│   │   ├── header.py                  #   KPI-Metriken
│   │   ├── tab_ai_mapping.py          #   AI-Mapping-Tab: Materialauswahl, Viewer-Sync
│   │   ├── tab_charts.py              #   Charts-Tab: Balken/Torte/Bubble nach KPI
│   │   └── tab_uploads.py             #   Upload-Tab: Modellwahl, IFC-Upload, Quick-Load
│   ├── data/                          #   Gespeicherte JSONL-Ergebnisse
│   ├── ifc-lite/                      #   TypeScript/Vite 3D-Viewer (npm/pnpm)
│   └── static/                        #   Hochgeladene IFC-Dateien für Viewer
│
├── Evaluation/                        # Modell-Evaluation
│   ├── run_evaluation_pipeline.py     #   Orchestrator: Query-Export → Evaluate → Report
│   ├── evaluate_material_models.py    #   Kern-Engine: 13 Bi-Encoder + Cross-Encoder Benchmarks
│   ├── build_evaluation_report.py     #   Markdown-Report + SVG-Übersichtsgrafik generieren
│   ├── build_split_evaluation_matrix.py
│   ├── export_sbert_queries_to_txt.py #   IFC/JSONL → Query-TXT für Evaluation
│   ├── run_single_model_evaluation.py #   Einzelevaluation eines einzelnen Bi-Encoder-Modells
│   ├── retrieval_metrics.py           #   Hit@K, MRR, MAP@10, nDCG@10, Recall@10
│   ├── metric_explanations.md         #   Erklärung der Metriken
│   ├── ground_truth/                  #   Ground-Truth-Dateien für Evaluation
│   ├── outputs/                       #   Generierte Queries + Evaluationsergebnisse
│   │   ├── queries/                   #     Exportierte Query-TXT-Dateien
│   │   └── single_model/              #     Outputs aus run_single_model_evaluation.py
│   ├── tests/                         #   Unit-Tests
│
├── Training/                          # Bi-Encoder Fine-Tuning
│   ├── run_training_pipeline.py       #   Orchestrator: validate → prepare → validate → qa → manifest → train
│   ├── prepare_training_data.py       #   Query/Expected TXT → JSONL-Trainingspaare
│   ├── train_bge_m3.py                #   Fine-Tuning mit MultipleNegativesRankingLoss
│   ├── validate_training_data.py      #   Validierung von Roh- und JSONL-Trainingsdaten
│   ├── run_data_qa_preflight.py       #   QA-Gates vor dem Training
│   ├── mine_hard_negatives.py         #   Mining harter Negatives aus Evaluation-Details
│   ├── mine_family_hard_negatives.py  #   Intra-Family Hard-Negatives
│   ├── query_generation/              #   Query-Generator-Skripte + Policy
│   │   ├── sources/                   #     Eingabedateien (possible_*.txt, Materialliste)
│   │   └── generated_queries/         #     Generierte Query-/Mapping-TXT aus den Query-Generatoren
│   ├── data/                          #   Rohdaten (Query-/Expected-TXT, Excel)
│   ├── artifacts/                     #   Trainierte Modelle + Trainingspaare
│   ├── outputs/                       #   QA-Reports, Manifeste, Auswertungen
│   └── tests/
│
├── test-scripts/                      # Ad-hoc-Beispiele und lokale Tests
├── models/                            # Lokaler Modell-Cache (SBERT, Cross-Encoder)
├── IFC-Modelle/                       # Test-IFC-Dateien und UBP-Berechnungsergebnisse
├── Ökobilanzdaten.sqlite3             # KBOB-Materialdatenbank (im Repo enthalten)
│
├── run_ifc_sbert_pipeline.py          # CLI-Einstiegspunkt: IFC → JSONL → SBERT-Matching
├── requirements.txt                   # Python-Abhängigkeiten
├── DATENFLUSS_EIGENSCHAFTEN.md        # Dokumentation: Datenfluss IFC → SBERT → Dashboard
├── CONTRIBUTING.md
├── LICENSE                            # MPL-2.0
└── THIRD_PARTY_NOTICES.md
```

## Voraussetzungen

- **Python 3.12**
- **KBOB-Datenbank:** `Ökobilanzdaten.sqlite3` (bereinigte und gefilterte Ökobilanzdaten der KBOB)
- **Optional:** Node.js / pnpm für die 3D-IFC-Viewer-Integration im Dashboard

## Quickstart

```bash
# Repository klonen
git clone https://github.com/Hygros/ifc-kbob-ai-matcher.git
cd ifc-kbob-ai-matcher

# Virtuelle Umgebung erstellen und aktivieren
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# Abhängigkeiten installieren
pip install -r requirements.txt

# Dashboard starten
streamlit run Dashboard/app_with_viewer.py
```

## Nutzung

### Dashboard

```bash
streamlit run Dashboard/app_with_viewer.py
```

1. **Uploads-Tab:** IFC-Datei hochladen, SBERT-Modell und optional Cross-Encoder wählen (Cross-Encoder sind nicht trainiert und verschlechtern daher das Resultat), "Mapping berechnen" klicken.
2. **AI-Mapping-Tab:** Vom AI vorgeschlagene KBOB-Materialien prüfen und bestätigen/korrigieren. 3D-Viewer zeigt das gewählte Element. Bewehrungsannahmen konfigurieren.
3. **Charts-Tab:** UBP, CO₂, Energie und weitere KPIs nach Element, Material oder IfcEntity visualisieren.

> **Credits:** Der integrierte 3D-Viewer basiert auf dem Open-Source-Projekt [ifc-lite](https://github.com/louistrue/ifc-lite) von [Louis True](https://github.com/louistrue) (Lizenz: MPL-2.0).

Manuell korrigierte Zuordnungen werden automatisch als Trainingsdaten nach `Training/data/` exportiert.

Der Dashboard-Trainingsexport verwendet dieselbe Query-Logik wie das Laufzeit-Matching (inklusive Name-basierter Disambiguierungs-Tokens), damit Evaluation/Training und Dashboard konsistent bleiben.

### CLI-Pipeline (ohne Dashboard)

```bash
# Komplette Pipeline: IFC → JSONL → SBERT-Matching
python run_ifc_sbert_pipeline.py <Pfad-zur-IFC-Datei>

# Nur IFC-Extraktion
python -m core.ifc_extraction.ifc_extraction_main <Pfad-zur-IFC-Datei>

# Batch-Export (Ordner mit IFC-Dateien → CSV + Analyse)
python -m core.ifc_extraction.ifc_batch_export_to_csv --ifc-folder <Ordner> --output-csv export.csv
```

## Evaluation

Die Evaluation vergleicht bis zu 13 Bi-Encoder-Modelle (+ optionalen Cross-Encoder (nicht trainiert und verschlechtern das Resultat)) gegen Ground-Truth-Zuordnungen.

```bash
# Komplette Pipeline mit interaktiver Modell-/Dateiauswahl
python Evaluation/run_evaluation_pipeline.py

# Mit expliziten Parametern
python Evaluation/run_evaluation_pipeline.py \
  --query-source Evaluation/ground_truth/queries.txt \
  --expected-file Evaluation/ground_truth/expected.txt \
  --cross-encoder-model BAAI/bge-reranker-v2-m3 \
  --rerank-top-n 30

# Einzelne Schritte
python Evaluation/evaluate_material_models.py   # Nur Evaluation
python Evaluation/build_evaluation_report.py     # Nur Report generieren
```

**Expected-Format** (eine Zeile pro Query, `|`-separiert für Alternativen, `::` für Relevanz-Gewichtung):

```text
Tiefgründung Ortbetonbohrpfahl 900
Material A | Material B | Material C
Material A::1.0 | Material B::0.7
```

Ergebnisse: `Evaluation/outputs/results/` (CSV, Markdown-Report, SVG-Grafik).

**Metriken:** Hit@K, MRR, MAP@10, nDCG@10, Recall@10 — Details in [Evaluation/metric_explanations.md](Evaluation/metric_explanations.md).

## Training

Empfohlener Standardpfad (Clean Baseline) ist in [Training/README.md](Training/README.md) dokumentiert.

Beispiel für einen reproduzierbaren Pipeline-Run (strict Hard-Negatives):

```bash
python Training/run_training_pipeline.py \
  --query-file Training/query_generation/generated_queries/generated_queries.txt \
  --expected-file Training/query_generation/generated_queries/mapping_generated_queries.txt \
  --base-model BAAI/bge-m3 \
  --pairs-out Training/artifacts/training_pairs_baseline_clean_run.jsonl \
  --output-dir Training/artifacts/models/baseline_clean_run \
  --hard-negatives-file Training/artifacts/hard_negatives_from_latest_eval.jsonl \
  --hard-negative-mode strict \
  --hard-negative-selection first \
  --qa-eval-query-file Training/data/dashboard_training_queries.txt \
  --qa-eval-expected-file Training/data/dashboard_training_expected.txt \
  --seed 42 \
  --dev-ratio 0.1 \
  --run-id baseline_clean_run
```

Modell danach evaluieren:

```bash
python Evaluation/run_single_model_evaluation.py \
  --model Training/artifacts/models/baseline_clean_run \
  --query-file Training/query_generation/generated_queries/generated_queries.txt \
  --expected-file Training/query_generation/generated_queries/mapping_generated_queries.txt \
  --device auto \
  --run-label baseline_clean_run \
  --output-dir Evaluation/outputs/single_model
```

Hinweis für Windows bei mehreren Python-Installationen: optional explizit mit `.\\.venv\\Scripts\\python.exe` starten.

## Umgebungsvariablen

Konfiguration über Umgebungsvariablen:

| Variable | Beschreibung | Default |
| ---------- | ------------- | --------- |
| `KBOB_DATABASE_PATH` | DB-Pfad für `core/*`-Pipelines | `./Ökobilanzdaten.sqlite3` |
| `KBOB_DB_PATH` | DB-Pfad für Dashboard/Evaluation/QA (Priorität vor Fallback-Pfaden) | leer |
| `ECOBILANZ_DB_PATH` | Alternativer DB-Pfad-Name (Fallback) | leer |
| `SBERT_DEVICE` | Device erzwingen: `cpu` oder `cuda` | Auto (GPU ab 500 Queries) |
| `SBERT_BATCH_SIZE` | Feste Batch-Size | `64` |
| `SBERT_AUTO_BENCH_BATCH` | Batch-Benchmark vor Matching | `0` |
| `SBERT_AUTO_HEURISTIC_BATCH` | Heuristische Batch-Size | `1` (aktiv) |
| `SBERT_CUDA_QUERY_THRESHOLD` | Mindest-Queries für Auto-GPU | `500` |
| `SBERT_CROSS_ENCODER_REVISION` | Pinned Cross-Encoder Revision | — |
| `SBERT_CROSS_ENCODER_ALLOW_UPDATES` | Erlaubt Remote-Code-Updates für Cross-Encoder (0/1) | `0` |

## Tests

```bash
python -m unittest discover -s Evaluation/tests -p "test_*.py" -v
python -m unittest discover -s Training/tests -p "test_*.py" -v
```

## Lizenz

[MPL-Lizenz](LICENSE).

Informationen zu Drittbibliotheken und Modell-Lizenzen: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

> **Hinweis:** Einige optionale Cross-Encoder-Modelle (z. B. Jina Reranker) stehen unter
> nicht-kommerziellen Lizenzen. Details siehe `THIRD_PARTY_NOTICES.md`.

