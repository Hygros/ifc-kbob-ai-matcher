# IFC-basierte Ökobilanzierung & Material-Matching

Automatisierte Zuordnung von IFC-Bauelementen zu Ökobilanzdaten (KBOB) mit Sentence-Transformer-basiertem Matching und Berechnung von Umweltindikatoren (UBP21, GWP, Primärenergie).

## Du möchtest das Tool ausprobieren und der Code interessiert dich weniger?
Leider hat HuggingFace den Space in der Vergangenheit gesperrt. Könnte sein das er nicht verfügbar ist oder etwas nicht funktioniert.  
Dann gehe auf diese Seite:  
[https://huggingface.co/spaces/Hygroskopisch/ifc-kbob-ai-matcher]  
oder auf  
[https://huggingface.co/spaces/Hygros-LCA/ifc-kbob-ai-matcher]  


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

Ergebnisse: `Evaluation/outputs/results/` (Pipeline-Outputs) sowie `Evaluation/outputs/single_model/` (Einzelmodell-Läufe).

**Metriken:** Hit@K, MRR, MAP@10, nDCG@10, Recall@10 — Details in [Evaluation/metric_explanations.md](Evaluation/metric_explanations.md).

### Evaluationsergebnisse

Die folgenden Ergebnisse basieren auf vier Single-Model-Läufen mit `models/Hygroskopisch/bge-m3-ifc-kbob-finetuned` ohne Cross-Encoder-Reranking.

| Queries | Cases | Hit@1 | Hit@10 | Hit@20 | Hit@30 | Hit@50 | MRR@10 | MAP@10 | nDCG@10 | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | 389 | 97.43% | 99.49% | 99.74% | 99.74% | 100.00% | 0.984 | 0.932 | 0.954 | 0.960 |
| Typos | 389 | 88.43% | 94.86% | 98.20% | 98.97% | 99.49% | 0.909 | 0.844 | 0.876 | 0.890 |
| Missing Attribute | 389 | 75.32% | 92.80% | 96.40% | 98.20% | 98.71% | 0.803 | 0.750 | 0.794 | 0.860 |
| Missing + Typos | 389 | 68.12% | 88.17% | 94.34% | 96.92% | 98.46% | 0.739 | 0.682 | 0.731 | 0.805 |

95%-Konfidenzintervalle (Bootstrap aus den Summary-Dateien):

| Queries | Hit@1 95% CI | Hit@10 95% CI | MRR@10 95% CI | nDCG@10 95% CI |
| --- | --- | --- | --- | --- |
| Normal | [95.37%, 98.97%] | [98.71%, 100.00%] | [0.971, 0.994] | [0.939, 0.968] |
| Typos | [84.83%, 91.77%] | [92.80%, 96.66%] | [0.881, 0.935] | [0.847, 0.902] |
| Missing Attribute | [70.69%, 79.18%] | [89.97%, 94.99%] | [0.766, 0.835] | [0.759, 0.824] |
| Missing + Typos | [63.36%, 72.49%] | [84.95%, 91.14%] | [0.695, 0.778] | [0.690, 0.767] |

### Query-Definitionen

Die vier unterschiedlichen Query-Dateien messen gezielt unterschiedliche Robustheitsachsen gegen verrauschte Eingaben.  
Queries sind hier: [Evaluation\ground_truth](Evaluation\ground_truth)

| Queries | Transformation | Harte Invarianten |
| --- | --- | --- |
| Normal | Unveränderte Query (Referenzlauf) | Keine Störung |
| Missing-Datei | Entfernt wird ein erlaubtes Token aus: `PredefinedType`, `Material`, `StrengthClass` oder `insitu/precast` bzw. `Ortbeton/Fertigteil` | `IfcEntity` wird nie entfernt |
| Typos-Datei | Pro Zeile 1 bis 2 Tippfehler, max. 1 Tippfehler pro Token/Wort | `IfcEntity` bleibt korrekt |
| Kombinierte Datei | Zuerst ein erlaubtes Token entfernen, danach 1 bis 2 Tippfehler auf verbleibenden erlaubten Tokens; max. 1 Tippfehler pro Token | `IfcEntity` bleibt korrekt |

Zusammenfassung der Queries:

| Datei | Geänderte Zeilen | Verteilung Tippfehler |
| --- | ---: | --- |
| Missing | 388 | - |
| Typos | 388 | 1 Tippfehler: 193, 2 Tippfehler: 195 |
| Missing + Typos | 388 | 1 Tippfehler: 309, 2 Tippfehler: 61 |

### Ausführliche Interpretation

Hinweis zur Lesart: Die Metriken basieren auf 389 Evaluationsfällen; die Tabelle oben zur Query-Erzeugung beschreibt die Anzahl geänderter Zeilen in den Stördateien.  
Artefakte: [Evaluation/outputs/single_model/bge-m3-ifc-kbob-finetuned/](Evaluation/outputs/single_model/bge-m3-ifc-kbob-finetuned/)

**Degradation gegenüber den Normalen Queries (quantifiziert)**

| Queries | Δ Hit@1 | Δ Hit@10 | Δ MRR@10 | Δ nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Typos | -9.00% | -4.63% | -0.075 | -0.078 |
| Missing Attribute | -22.11% | -6.69% | -0.181 | -0.160 |
| Missing + Typos | -29.31% | -11.32% | -0.245 | -0.223 |

Schlussfolgerung: Token-Entfernung verursacht den grösseren Schaden als reine Schreibfehler; die Kombination ist erwartungsgemäss am stärksten.

**Typos vs. Missing (direkter Vergleich der Fehlerarten)**
- Hit@1: Missing liegt 13.11 Prozentpunkte unter Typos (75.32% vs. 88.43%).
- Hit@10: Missing liegt 2.06 Prozentpunkte unter Typos (92.80% vs. 94.86%).
- MRR@10: Missing liegt 0.106 unter Typos (0.803 vs. 0.909).
- nDCG@10: Missing liegt 0.082 unter Typos (0.794 vs. 0.876).

Schlussfolgerung: Fehlende semantische Slots verschieben korrekte Treffer stärker aus den vorderen Rängen als Tippfehler.

**Top-1 vs. Top-10: konkretes Aufholpotenzial**
- Normal: Hit@10 - Hit@1 = 2.06%.
- Typos: Hit@10 - Hit@1 = 6.43%.
- Missing Attribute: Hit@10 - Hit@1 = 17.48%.
- Missing + Typos: Hit@10 - Hit@1 = 20.05%.

Schlussfolgerung: Unter Störungen bleibt das korrekte Material oft in den Top-10, fällt aber deutlich häufiger aus Rang 1.

**Statistische Trennschärfe (Hit@1-CIs)**
- Normal vs. Typos: keine Überlappung; Abstand zwischen Intervallen 3.60% (95.37% vs. 91.77%).
- Typos vs. Missing: keine Überlappung; Abstand 5.65% (84.83% vs. 79.18%).
- Missing vs. Missing + Typos: Überlappung 1.80% (70.69% bis 72.49%).

Schlussfolgerung: Die ersten beiden Verschlechterungsschritte sind klar separiert; der letzte Schritt ist kleiner, aber weiterhin negativ.

**Konsequenzen für Einsatz und UI**
- Für hohe Automatisierungspräzision ist die Stabilität der Slots `Material`, `StrengthClass` und `CastingMethod` entscheidend.
- Bei verrauschten IFC-Texten sollte die UI primär mit Top-10-Kandidaten arbeiten und Top-1 nicht als alleinige Entscheidung verwenden.
- Verbesserungshebel liegt weniger bei zusätzlicher Tippfehler-Toleranz als bei robuster Extraktion/Erhaltung semantischer Tokens.

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

