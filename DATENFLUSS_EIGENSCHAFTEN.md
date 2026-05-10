# Datenfluss: IFC-Export → SBERT Mapping → Dashboard

Stand: Abgleich mit aktuellem Code in Dashboard + angebundenen Kernmodulen.

## 1. IFC-EXPORT (core/ifc_extraction/ifc_extraction_core.py)

### Direkt aus IFC-Element extrahiert (Basis-Attribute)
```
IfcEntity
PredefinedType
Name
Description
GUID
```

### Zusätzliche technische Felder für Dashboard/Viewer-Linking
```
HasModeledRebar        (abgeleitet aus IfcReinforcingBar-Geschwisterelementen)
AggregateChildGUIDs    (Descendants für Viewer-Highlighting)
AggregateParentGUID    (Parent-GUID für Viewer-Highlighting)
```

### Aus PropertySets extrahiert
DEFAULT_PROPERTY_FIELDS:
```
Status
CastingMethod
StrengthClass
Length
NetVolume
GrossVolume
ReinforcementVolumeRatio
```

Zusätzlich extrahiert:
```
Height                 (für Flächenberechnung)
NetArea                (für Flächenberechnung)
Count                  (nur bei IfcReinforcingBar)
Weight                 (nur bei IfcReinforcingBar)
```

### Berechnete Felder
```
Durchmesser            (aktuell für IfcPile aus Length + NetVolume)
Ansichtsfläche         (IfcWall: Length × Height, IfcCovering: NetArea)
```

### Material-Informationen
```
Material               (als Liste, z.B. ["Beton C30/37"])
MaterialLayerIndex     (1..n bei mehrschichtigen Materialien)
MaterialLayerThickness (pro Layer)
```

Hinweis: Bei mehrschichtigen Materialien werden NetVolume/GrossVolume anteilig über die Layer-Dicken aufgeteilt (falls alle Dicken vorhanden und > 0).

---

## 2. SBERT-Mapping (vom Dashboard genutzt)

Primärer Runtime-Pfad:
```
Dashboard/services/ifc_pipeline.py
    -> core/sbert/sentence_transformer.py (run_sbert_matching)
```

### Bi-Encoder Query-Felder (core/sbert/sentence_transformer.py)
```
IfcEntity
PredefinedType
Material
Durchmesser
CastingMethod
StrengthClass
```

Diese Basisfelder werden zu einem Query-Text konkateniert und gegen die KBOB-Materialdatenbank gematcht.

`Name` wird nicht als Rohtext in die Basisfelder aufgenommen. Stattdessen wird `Name` gezielt für Baugrubensicherungs-Disambiguierung genutzt (fuzzy, mehrsprachig), und nur erkannte Tokens werden ergänzt, z. B.:
```
Schlitzwand | Bohrpfahlwand | Spundwand | Nagelwand | Rühlwand
verankert | gespriesst | auskragend | unverankert
400 | 800
```

### Cross-Encoder Reranking (optional im Upload-Tab)
```
TOP_K_RESULTS = 30
RERANK_TOP_N = 30
```

Bei aktivem Cross-Encoder werden die Top-N Treffer pro Query neu bewertet; die Scores werden auf [0, 1] normalisiert.

### Export/Training-Konsistenz
Evaluation/export_sbert_queries_to_txt.py nutzt dieselbe Query-Funktion (`ifc_entry_to_string`) wie das Laufzeit-Matching.

Dashboard/services/training_export.py (`record_to_query`) ist auf dieselbe Logik abgeglichen, inklusive Name-basierter Disambiguierungs-Tokens.

---

## 3. Dashboard-Darstellung (Dashboard/ui/tab_ai_mapping.py)

### Basis-Spalten für die Mapping-UI (base_cols)
```
IfcEntity
PredefinedType
Name
GUID
MaterialLayerIndex
Description
Material
Durchmesser
CastingMethod
StrengthClass
top_k_matches
AggregateChildGUIDs
AggregateParentGUID
```

### Element-Label in der linken Liste
Zusammengestellt aus gültigen Werten in:
```
IfcEntity | PredefinedType | Name | Description | Material | CastingMethod | StrengthClass | Ø Durchmesser
```

Zusatz:
```
(n Elemente)            (wenn eine Gruppe mehrere GUIDs enthält)
```

### Materialauswahl pro Gruppe
```
1) Top-K Treffer aus top_k_matches (mit Score)
2) Danach vollständige KBOB-Materialliste (Fallback/Manuell)
```

### Im Merge zusätzlich verwendete Spalten
```
Material KBOB
AI Score
```

### Übersicht-Tabelle (Subheader "Übersicht")
```
IfcEntity
PredefinedType
Name
Beschrieb              (aus Description)
Durchmesser
Material KBOB
AI Score               (3 Nachkommastellen)
```

---

## 4. Filtern, Gruppieren, Ausschlüsse

### Ausgeschlossene Zeilen
```
MaterialLayerIndex == "R"
MaterialLayerIndex == "Z"
```
Diese synthetischen Bewehrungs-/Verzinkungszeilen werden in AI-Mapping ausgeblendet und primär für Charts/Totals genutzt.

### Gruppierungslogik (Dashboard/domain/mapping.py)
Standard-Gruppierung über:
```
IfcEntity, PredefinedType, Name, Description, Material, Durchmesser, MaterialLayerIndex, top_k_matches
```

Spezialfall Bewehrungs-Entities:
```
IfcReinforcingBar, IfcReinforcingMesh, IfcTendon
```
Diese werden je Entity-Typ zusammengefasst; bei unterschiedlichen Feldwerten werden Konfliktfelder auf None gesetzt.

### In der AI-Mapping-Tabelle nicht direkt angezeigt
```
GUID
AggregateChildGUIDs
AggregateParentGUID
Status
CastingMethod
StrengthClass
Height
NetArea
Count
Weight
ReinforcementVolumeRatio
MaterialLayerThickness
```

---

## 5. Reinforcement-spezifische Felder

### Automatisch angereichert (add_reinforcement_info)
```
is_concrete
has_modeled_rebar
reinforcement_ratio_source      ("ifc" | "default" | None)
reinforcement_ratio_kg_m3
reinforcement_mass_kg
reinforcement_status            ("explicit" | "assumed" | "none" | "no_material")
```

Details:
```
- IfcReinforcingBar erhält Status "none" und keinen Ratio-Default.
- reinforcement_mass_kg wird nur für Status "assumed" geführt.
```

### Im Dashboard vom Nutzer gesetzt/überschrieben
```
reinforcement_accepted
reinforcement_ratio_kg_m3
reinforcement_source            (typisch "user", sonst aus Auto-Logik)
```

Persistiert wird pro GUID + MaterialLayerIndex in die JSONL-Datei.

---

## 6. Kurzüberblick Feldabdeckung

```
IFC-Export
    Basis + PSet + berechnete Felder + Material-Layer + Viewer-GUID-Hilfsfelder

SBERT-Matching (Runtime)
    6 Basis-Query-Felder + optionale Name-Disambiguierungs-Tokens -> top_k_matches (Top 30)
    optional: Cross-Encoder Reranking der Top 30

Dashboard AI-Mapping
    base_cols: 11 Felder
    Übersicht: 7 sichtbare Spalten
    Persistenz: Auswahlfelder + Reinforcement-Entscheidungen
```

---

## 7. Keine Werte: is_valid-Filter im Label

Werte werden nur angezeigt, wenn sie diese Checks bestehen:
```
- value NOT IN (None, "", [], {})
- string_value.strip() nicht leer
- normalized(value) NOT IN {"nan", "none", "null", "undefined", "notdefined", "n/a", "na", "-"}
```
