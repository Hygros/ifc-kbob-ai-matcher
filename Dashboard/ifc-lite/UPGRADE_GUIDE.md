# IFClite Update-Anleitung (Streamlit-Integration, Auto-Start stabil)

Diese Anleitung beschreibt den sicheren Update-Prozess fuer Dashboard/ifc-lite inklusive der Dashboard-Bridge, der Viewer-Autostart-Logik und der Selektionsfunktionen.

## Upstream-Credits

Dashboard/ifc-lite basiert auf dem Open-Source-Projekt ifc-lite (https://github.com/louistrue/ifc-lite) von Louis True (https://github.com/louistrue), Lizenz MPL-2.0.

Attribution und THIRD_PARTY_NOTICES.md beim Update immer beibehalten.

---

## 1) Ziel und Scope

Diese Schritte aktualisieren primaer:

- Dashboard/ifc-lite

Dabei muessen Integrationsdateien mit lokalen Anpassungen erhalten bleiben:

- Dashboard/services/viewer.py
- Dashboard/services/bootstrap.py
- Dashboard/ui/tab_ai_mapping.py
- Dashboard/ifc-lite/src/components/streamlit/StreamlitBridge.tsx
- Dashboard/ifc-lite/src/components/viewer/Viewport.tsx
- Dashboard/ifc-lite/src/App.tsx

Wenn Upstream-Dateien diese Bereiche ueberschreiben, lokale Patches nach dem Merge wieder einpflegen.

---

## 2) Vorbereitung

1. Terminal im Repo oeffnen.
2. In den Frontend-Ordner wechseln.

```powershell
cd Dashboard/ifc-lite
```

3. Arbeitsstand sichern.

```powershell
git status
git add .
git commit -m "chore: backup before ifc-lite update"
```

4. Versionslage prüfen.

```powershell
npm outdated
```

5. Vorab-Port-Check (wichtig fuer reproduzierbare Tests).

```powershell
if (Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue) { "8501 belegt" } else { "8501 frei" }
if (Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue) { "3000 belegt" } else { "3000 frei" }
```

---

## 3) Update durchfuehren

### Option A: Voll-Update

```powershell
npm update
npm install
```

### Option B: Gezieltes Update der Kernpakete (empfohlen)

```powershell
npm install @ifc-lite/parser@latest @ifc-lite/geometry@latest @ifc-lite/renderer@latest @ifc-lite/server-client@latest @ifc-lite/query@latest
npm install
```

Hinweis: npm install am Ende sorgt fuer konsistentes package-lock.json.

---

## 4) Build- und Dev-Validierung (Windows)

Fuer Build-Pruefung:

```powershell
npx vite build
```

Wichtig fuer Dev-Start auf Windows:

- Standard: nur npm run dev verwenden.
- Keine zusaetzlichen forwarded Args mit -- --host ... --port ..., wenn package.json den Port bereits fix setzt.

```powershell
npm run dev
```

Falls manuell Vite ohne Script gestartet werden soll:

```powershell
npx vite --host 127.0.0.1 --port 3000 --strictPort
```

---

## 5) Funktionstest im Dashboard (Pflicht)

Dashboard starten:

```powershell
python Dashboard/app_with_viewer.py
```

oder alternativ:

```powershell
python -m streamlit run Dashboard/app_with_viewer.py --server.port 8501
```

Dann im Tab AI-Mapping pruefen:

1. Auto-Start Viewer
   - Keine manuelle npm run dev-Session starten.
   - Viewer erscheint im iframe unter http://localhost:3000.

2. Selectbox -> Viewer
   - Material/Element waehlen.
   - Entsprechendes Element wird im Viewer selektiert.

3. Viewer -> linke Markierung
   - Element im Viewer anklicken.
   - Passende Zeile links wird hervorgehoben.

4. Outside Click -> Deselect
   - Ausserhalb Viewer/Selectbox klicken.
   - Viewer-Selektion wird aufgehoben.

5. Neue Auswahl nach Deselect
   - Danach erneut auswaehlen.
   - Viewer selektiert wieder korrekt.

---

## 6) Kritische Upgrade-Checks (aus aktuellen Erkenntnissen)

1. Renderer-Kamera-API

- Bei neuen @ifc-lite/renderer-Versionen kann setOrbitCenter fehlen.
- In Viewport-Logik setOrbitPivot verwenden.
- Falls mehrere Renderer-Versionen unterstuetzt werden muessen: Runtime-Guard mit Fallback auf Legacy-Methode.

2. Viewer-Autostart in Streamlit

- viewer_server_started nicht optimistisch auf True setzen.
- Nur True setzen, wenn Port 3000 wirklich erreichbar ist.
- Startup mit Retry-Intervall absichern (Timestamp in Session-State).

3. Startkommando fuer Viewer

- In der Python-Integration fuer npm run dev keine zusaetzlichen CLI-Weiterleitungen erzwingen, wenn das Dev-Script Host/Port schon enthaelt.

4. Error-Containment im Frontend

- App-level Error Boundary beibehalten, damit ein Runtime-Fehler im Viewport nicht die gesamte App-Oberflaeche entfernt.

5. Sauberer Neuaufbau nach Ordnerproblemen

- Ordnername und Pfad NICHT aendern (weiterhin exakt Dashboard/ifc-lite).
- Wenn der Viewer-Tree inkonsistent ist (fehlende src/lib-Dateien etc.), in-place neu erzeugen:

```powershell
cd Dashboard
npx --yes create-ifc-lite ifc-lite --template react
```

- Danach lokale Integrationsdateien wiederherstellen (mindestens StreamlitBridge, App, Viewport-spezifische Patches).

6. StreamlitBridge nach Neuaufbau wieder aktiv einhaengen

- StreamlitBridge-Datei aus dem Projekt-Remote holen.
- Sicherstellen, dass App.tsx die Bridge importiert und innerhalb von BimProvider rendert.

7. Windows-Startverhalten von npm run dev

- Wenn npm forwarded Args falsch an vite weitergibt (z. B. vite 127.0.0.1 3000), startet der Viewer auf falschem Port.
- Loesung: Dev-Script so definieren, dass Host/Port intern fix gesetzt sind und kein externer Forwarding-Zwang benoetigt wird.
- Danach immer pruefen: http://127.0.0.1:3000 muss HTTP 200 liefern.

8. Viewport-Runtime-Haertung

- Viewport nicht mounten, solange WebGPU noch geprueft wird oder nicht verfuegbar ist.
- Renderer-Erzeugung und renderer.init in try/catch absichern, damit der iframe nicht mit React-Fehlerseite endet.

---

## 7) Troubleshooting

Problem: python Dashboard/app_with_viewer.py meldet Port 8501 is not available.

- Ursache: Streamlit laeuft bereits.
- Loesung: existierende Session nutzen oder freien Port waehlen.

Problem: Viewer erscheint nicht und localhost:3000 ist down.

- npm im PATH pruefen.
- Dashboard/ifc-lite und node_modules pruefen.
- Wenn noetig manuell in Dashboard/ifc-lite npm run dev starten.

```powershell
Get-Command npm -ErrorAction SilentlyContinue
Test-Path Dashboard/ifc-lite
Test-Path Dashboard/ifc-lite/node_modules
if (Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue) { "3000 ON" } else { "3000 OFF" }
```

Problem: Viewer laeuft, Selektion aber nicht synchron.

- Bridge-Nachrichten pruefen:
  - ifc-lite-select-guid
  - ifc-lite-select-guids
  - ifc-lite-viewer-selection

Problem: iframe bleibt leer, Konsole zeigt React-Fehler in <Viewport>.

- Ursache haeufig: Renderer/WebGPU-Init wirft Runtime-Fehler vor stabilem Fallback.
- Loesung:
   - WebGPU-Guard in ViewportContainer aktiv halten (kein Viewport-Mount waehrend checking/unsupported).
   - Renderer-Konstruktion + renderer.init in Viewport per try/catch absichern.
   - Danach Dev-Server neu starten und Browser hart neu laden.

---

## 8) Rollback ohne Datenverlust

Im Projektordner:

```powershell
cd Dashboard/ifc-lite
git restore package.json package-lock.json
npm install
```

Wenn ein kompletter Rueckschritt noetig ist, bevorzugt per revert statt Hard-Reset:

```powershell
git log --oneline -n 10
git revert <commit_sha>
npm install
```

---

## 9) Checkliste fuer Copilot

Empfohlener Auftrag:

- Bitte fuehre die Schritte aus Dashboard/ifc-lite/UPGRADE_GUIDE.md aus,
  update ifc-lite sicher und validiere Auto-Start (3000) sowie die AI-Mapping-Selektionsfaelle.

Damit ist sichergestellt, dass nicht nur Build/Test, sondern auch die Dashboard-Integration stabil bleibt.
