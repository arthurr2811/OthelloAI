# Othello AI – Projektplan

AlphaZero-artige Othello-KI, lokal auf GPU trainiert, mit lokalem Web-Frontend zum
Selberspielen. Portfolio-Projekt. Ziel: ein Modell, das einen menschlichen
Hobbyspieler zuverlässig schlägt — und eine saubere, nachvollziehbare Pipeline,
die zeigt *wie* so etwas funktioniert.

## Workflow (Zusammenarbeit)

- **Claude committet nie selbst.** Ich (Claude) führe die Unterschritte eines
  großen Schritts aus und gebe dann eine Zusammenfassung, was passiert ist.
- **Arthur prüft und committet.** Du checkst gegen, committest selbst und sagst
  entweder „weiter geht's" oder meldest dich mit Anmerkungen.
- Erst nach deinem „OK" beginnt der nächste Schritt.

## Leitprinzipien

- **Jede Phase = ein Feierabend (2–3 h)** mit klarem „Fertig wenn"-Kriterium.
- **Immer lauffähig.** Nach jedem Schritt existiert etwas Testbares — nie ein
  halbfertiger Zwischenzustand über mehrere Abende.
- **Erst korrekt, dann schnell, dann schlau.** Engine → Baselines → MCTS ohne Netz
  → AlphaZero. Jede Stufe beweist, dass die darunter stimmt.
- **Kleine Version zuerst.** 6×6-Othello als Durchstich für die Pipeline, bevor das
  teure 8×8-Training läuft. Spart Tage an blindem Warten.
- **Tests sind kein Extra.** Die Engine ist das Fundament; ein Flip-Bug versaut
  sonst später tagelang das Training, ohne dass man es merkt.

## Stack

- Python 3.11+, PyTorch (CUDA)
- NumPy für die Board-/Bitboard-Logik (Self-Play ist der Flaschenhals)
- pytest für Tests
- FastAPI + simples HTML/JS für das Frontend
- Konfig über einfache Python-Dataclasses / YAML, Checkpoints als `.pt`

---

## Phase 0 – Setup

**Ziel:** Reproduzierbares Projektgerüst, alles läuft.

- [x] `venv` anlegen, `requirements.txt` (numpy, torch, pytest, fastapi, uvicorn).
- [x] CUDA prüfen: `torch.cuda.is_available()` → `True`, GPU-Name ausgeben.
- [x] Projektstruktur anlegen:
  ```
  othello/        # Engine (reine Spiellogik, kein ML)
  agents/         # Bots: random, greedy, mcts, alphazero
  az/             # AlphaZero: netz, mcts, selfplay, train, evaluate
  web/            # FastAPI-Backend + statisches Frontend
  tests/
  scripts/        # Einstiegspunkte (train.py, play.py, arena.py)
  config.py
  ```
- [x] `README.md` mit Kurzbeschreibung + Setup-Anleitung.
- [x] Git: `.gitignore` (venv, `__pycache__`, `*.pt`, Checkpoints, Logs).

**Fertig wenn:** `pytest` läuft (0 Tests ok), CUDA wird erkannt, Struktur committed.

---

## Phase 1 – Engine & Baselines

### Schritt 1.1 – Board & Spielzustand

**Ziel:** Datenmodell des Spiels steht.

- [x] Board-Repräsentation wählen: 8×8 NumPy-Array (`+1`/`-1`/`0`) für Klarheit;
      Bitboard-Optimierung später, wenn Speed nötig wird.
- [x] Startstellung, `current_player`, `to_string()` für Debug-Ausgabe.
- [x] Richtungs-Offsets (8 Richtungen) als Konstante.

**Fertig wenn:** Board lässt sich anlegen und lesbar ausgeben; Startstellung stimmt
(4 Steine in der Mitte).

### Schritt 1.2 – Zuggenerierung & Flip-Logik

**Ziel:** Herzstück der Engine — korrekte legale Züge und Umdrehen.

- [x] `legal_moves(board, player)` → Liste gültiger Felder.
- [x] `apply_move(board, player, move)` → neues Board mit korrekt geflippten Steinen
      in allen 8 Richtungen.
- [x] Unveränderlichkeit beachten (kopieren, nicht in-place mutieren).

**Fertig wenn:** Auf einer Handvoll manuell durchgerechneter Stellungen stimmen
legale Züge und Flips exakt.

### Schritt 1.3 – Pass, Game-Over & Regeln absichern

**Ziel:** Die kniffligen Randfälle lösen.

- [x] Pass-Regel: hat ein Spieler keinen legalen Zug, muss er passen (Gegner wieder
      dran). Nur wenn **beide** nicht ziehen können → Spielende.
- [x] `game_over(board)` + `winner(board)` (Steinmehrheit, inkl. Unentschieden).
- [x] Vollständige pytest-Suite für 1.2 + 1.3: Startzüge, Flips je Richtung,
      erzwungenes Pass, Spielende bei vollem Brett und bei beidseitigem Pass.
- [x] Property-Test: ein komplettes Random-Spiel läuft ohne Fehler bis zum Ende
      und Steinzahl bleibt konsistent.

**Fertig wenn:** `pytest` grün, inkl. der Pass-/Game-Over-Fälle. **Ab hier wird die
Engine nicht mehr angefasst, ohne dass Tests grün bleiben.**

### Schritt 1.4 – Baseline-Bots & Arena

**Ziel:** Gegner zum Messen + Infrastruktur, um Bots gegeneinander spielen zu lassen.

- [x] Einheitliches `Agent`-Interface: `select_move(state) -> move` (State statt
      board+player, damit Agenten die Pass-Logik nicht duplizieren).
- [x] `RandomAgent`, `GreedyAgent` (Zug, der die meisten Steine schlägt).
- [x] `arena.py`: spielt N Partien zwischen zwei Agents, wechselt Startspieler,
      loggt Win/Loss/Draw-Statistik.
- [x] Sanity-Check: Greedy schlägt Random (~61 % über 500 Partien; Greedy ist bei
      Othello nur mäßig stark, daher Schwelle 55 % statt harter 60 %-Kante).

**Fertig wenn:** `python scripts/arena.py` gibt eine saubere Bilanz aus; Greedy > Random.

### Schritt 1.5 – MCTS ohne Netz

**Ziel:** Reines UCT-MCTS mit Random-Rollouts — der Sanity-Check *vor* jedem ML.

- [x] `MCTSAgent`: Selection (UCB1), Expansion, Random-Rollout, Backprop.
- [x] Simulationsbudget pro Zug konfigurierbar.
- [x] Arena: MCTS schlägt Greedy klar (150 Sims: 30:0 = 100 %).

**Fertig wenn:** MCTS (z. B. 200 Sims/Zug) gewinnt deutlich gegen Greedy. Damit ist
bewiesen, dass Engine + Suche korrekt zusammenspielen — komplett ohne ML-Unsicherheit.

> **Meilenstein 1:** Getestete Engine, messbare Baselines, funktionierendes MCTS.
> Das allein ist schon ein vorzeigbares Mini-Projekt.

---

## Phase 2 – AlphaZero-Pipeline

> Diese Phase auf **6×6-Othello** entwickeln (kleiner, konvergiert in Stunden).
> Erst wenn die Pipeline dort nachweislich stärker wird, auf 8×8 skalieren.

### Schritt 2.1 – Neuronales Netz

**Ziel:** Netz-Architektur steht und läuft auf der GPU.

- [ ] Kleines ResNet: Input = Board-Ebenen (eigene Steine / gegnerische Steine /
      Spieler-am-Zug), ein paar Conv-Residual-Blöcke.
- [ ] **Policy-Head** (Logits über alle Felder + Pass) und **Value-Head** (tanh,
      Gewinnwahrscheinlichkeit aus Sicht des Ziehenden).
- [ ] Forward-Pass mit Dummy-Batch auf GPU testen (Shapes, keine NaNs).

**Fertig wenn:** Netz nimmt einen Batch Boards, liefert Policy + Value in korrekten
Shapes auf der GPU.

### Schritt 2.2 – MCTS mit Netz

**Ziel:** PUCT-MCTS, das statt Rollouts das Netz zur Bewertung nutzt.

- [ ] PUCT-Formel (Priors aus Policy-Head, Value statt Rollout).
- [ ] Dirichlet-Noise an der Wurzel (Exploration im Self-Play).
- [ ] Temperatur-Parameter für die Zugauswahl aus den Visit-Counts.
- [ ] Legal-Move-Masking auf die Policy.

**Fertig wenn:** MCTS+Netz (noch untrainiert) läuft fehlerfrei und liefert für eine
Stellung eine Visit-Count-Verteilung.

### Schritt 2.3 – Self-Play-Loop

**Ziel:** Der Agent erzeugt Trainingsdaten gegen sich selbst.

- [ ] Eine Self-Play-Partie: pro Zug MCTS laufen lassen, `(state, policy_target,
      spieler)` speichern; am Ende Value-Target aus dem Ergebnis rückwärts einfüllen.
- [ ] Symmetrie-Augmentierung (8 Dihedral-Varianten) — nahezu gratis, großer
      Sample-Efficiency-Gewinn.
- [ ] Daten in einen Replay-Buffer (deque mit Max-Größe) schreiben.

**Fertig wenn:** Eine Self-Play-Partie erzeugt korrekt geformte Trainingssamples
(inkl. augmentierter), im Buffer sichtbar.

### Schritt 2.4 – Trainingsschleife

**Ziel:** Netz aus Replay-Buffer-Daten trainieren.

- [ ] Loss = Policy-Cross-Entropy + Value-MSE (+ L2/Weight-Decay).
- [ ] Optimizer (Adam/SGD), Batches aus dem Buffer, Loss-Kurven loggen
      (TensorBoard oder simples CSV/matplotlib).
- [ ] Checkpoint speichern/laden.

**Fertig wenn:** Trainingsschritt läuft, Loss sinkt auf einem festen Datenbatch
(Overfit-Test als Korrektheitsbeweis).

### Schritt 2.5 – Evaluation & Gating

**Ziel:** Nur echte Verbesserungen werden übernommen — das Frühwarnsystem.

- [ ] Neues Modell vs. aktuell bestes über N Partien in der Arena.
- [ ] Gating: nur bei Siegquote über Schwelle (z. B. 55 %) wird das neue Modell
      „bestes" Modell.
- [ ] Zusätzlich gegen Greedy/MCTS-Baseline messen → absolute Stärke-Kurve.

**Fertig wenn:** Eval-Loop kürt einen Sieger und aktualisiert den „best model"-Zeiger.

### Schritt 2.6 – Orchestrierung & 6×6-Durchstich (Achtung lange Trainingszeit)

**Ziel:** Alles zu einem Loop verbinden und auf 6×6 nachweislich lernen.

- [x] `scripts/train.py`: Loop aus Self-Play → Train → Evaluate → (Gate) über N
      Iterationen (`az/pipeline.py`, CLI mit `--smoke`-Wiring-Check).
- [x] Zentrale Config (`RunConfig`: Sims, Iterationen, Buffer-Größe, LR,
      Batchgröße … bündelt alle Sub-Configs).
- [x] Resume aus Checkpoint (`--resume best.pt`), Logging pro Iteration
      (`logs/iterations.csv` + `loss_iter_XXX.csv`).
- [ ] **>>> NÄCHSTER SCHRITT (manuell, Arthur): 6×6-Lauf starten und beobachten.**
      `python scripts/train.py` (Defaults: 6×6, 40 Iterationen). Stärke gegen
      Baselines soll über die Iterationen steigen. Vorbereitung + Wiring stehen und
      sind getestet (`--smoke` läuft grün auf GPU); nur der eigentliche,
      langlaufende Trainingslauf fehlt noch.

**Fertig wenn:** Auf 6×6 schlägt das trainierte Modell die Baselines klar und wird
über die Iterationen messbar stärker. **Damit ist die Pipeline validiert.**

> **Meilenstein 2:** Funktionierende AlphaZero-Pipeline, auf 6×6 nachweislich
> lernend. Der schwierigste Teil des Projekts ist geschafft.

### Schritt 2.7 – 8×8-Training (Achtung lange Trainingszeit)

**Ziel:** Das eigentliche Zielmodell.

- [ ] Config hochskalieren (größeres Netz, mehr Sims, mehr Iterationen).
- [ ] Self-Play parallelisieren, falls zu langsam (mehrere Spiele gleichzeitig,
      Batched-Inferenz) — an die GPU angepasst.
- [ ] Lauf über mehrere Tage, regelmäßig Checkpoints + Stärke-Kurve sichern.

**Fertig wenn:** Ein 8×8-Checkpoint schlägt Greedy und das reine MCTS deutlich —
Kandidat für „schlägt Hobbyspieler".

---

## Phase 3 – Lokales Frontend (zum Selberspielen)

### Schritt 3.1 – Inferenz-Backend

**Ziel:** FastAPI-Server lädt das Modell und liefert Züge.

- [ ] Endpoints: `POST /new_game`, `POST /move` (Menschzug → KI-Antwortzug),
      `GET /state`.
- [ ] Best-Checkpoint laden, MCTS+Netz für den KI-Zug (kleines Sim-Budget für
      flottes Spiel).

**Fertig wenn:** Per curl/HTTP kann man eine Partie gegen das Modell durchspielen.

### Schritt 3.2 – Web-Board

**Ziel:** Klickbares 8×8-Board im Browser.

- [ ] Statisches HTML/JS: Board rendern, legale Züge markieren, Klick → `/move`,
      KI-Antwort anzeigen, Endstand.
- [ ] Reset-Button, Anzeige Steinzahl / wer am Zug ist.

**Fertig wenn:** Man spielt lokal im Browser eine komplette Partie gegen die KI.

### Schritt 3.3 – Feinschliff & Selbsttest

**Ziel:** Portfolio-Reife + Realitätscheck.

- [ ] Schwierigkeitsstufen über MCTS-Sim-Budget.
- [ ] Selbst (und Freunde) gegen die KI spielen → hält sie „schlägt Hobbyspieler"?
- [ ] README rund?

**Fertig wenn:** Rundes, im Browser spielbares Projekt mit dokumentierten Ergebnissen.

> **Meilenstein 3:** Lokal spielbare Othello-KI + vorzeigbares Portfolio-Repo.
> (Hetzner-Deployment später — bewusst außerhalb dieses Plans.)

---

### Schritt 4 Cleanup

**Ziel:** Sauberer Code

- [ ] Code aufgeräumt, nur für Zwischenschritte nötigen Code entfernt.
- [ ] Readme ist rund.
- [ ] Deliverable: Gut DOkumentiertes Projekt, das aufbau der KI zeigt und erklärt. Fertig trainiertes Modell, dass menschliche Spieler zuverlässig schlagen kann.
## Bewusst später / außerhalb des Scopes

- Hetzner-Deployment des fertigen Modells (eigene Aufgabe, wenn das Modell steht).
- Bitboard-Optimierung (nur falls Self-Play zum Engpass wird).
- Fortgeschrittenes: Gumbel-AlphaZero, größere Netze, Openings-Diversität.

## Fortschritt

- [x] Phase 0 – Setup
- [x] Phase 1 – Engine & Baselines (Meilenstein 1)
- [ ] Phase 2 – AlphaZero-Pipeline (Meilenstein 2)
- [ ] Phase 3 – Frontend (Meilenstein 3)
