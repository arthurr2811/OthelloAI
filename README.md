# Othello AI

Eine AlphaZero-artige Othello-KI – lokal auf GPU trainiert, mit Web-Frontend zum
Selberspielen. Die Pipeline geht von einer getesteten Spiel-Engine über klassische
Baselines und reines MCTS bis zum selbst-trainierten neuronalen Netz.

Details zum Vorgehen: siehe [`plan.md`](plan.md).

## Stack

- Python 3.11+ (entwickelt mit 3.13), PyTorch (CUDA)
- NumPy für die Board-Logik, pytest für Tests
- FastAPI + HTML/JS für das Frontend

## Setup

```bash
# 1. venv anlegen
py -3.13 -m venv .venv
.venv\Scripts\activate        # Windows (PowerShell/CMD)
# source .venv/bin/activate   # Linux/macOS

# 2. Basis-Dependencies
pip install -r requirements.txt

# 3. PyTorch mit CUDA passend zur GPU installieren.
#    RTX-50-Serie (Blackwell, sm_120) braucht das CUDA-12.8-Wheel:
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 4. Umgebung prüfen (erwartet: CUDA verfügbar = True, GPU-Name)
python scripts/check_env.py

# 5. Tests
pytest
```

## Projektstruktur

```
othello/   # Engine (reine Spiellogik, kein ML)
agents/    # Bots: random, greedy, mcts, alphazero
az/        # AlphaZero: netz, mcts, selfplay, train, evaluate
web/       # FastAPI-Backend + statisches Frontend
tests/
scripts/   # Einstiegspunkte (train.py, play.py, arena.py, check_env.py)
config.py  # zentrale Konfiguration
```

## Die Agenten – wie sie spielen

Mehrere Ansätze in Form von verschiedenen Agenten, von leicht umsetzbar und nur minimal stärker als Random bis zu 
wirklich intelligent.

### Baseline-Bots: Random & Greedy

- **RandomAgent** wählt gleichverteilt einen der legalen Züge. Er ist die
  absolute Untergrenze: nützlich als Messlatte und um die Engine zu stressen
  (tausende Zufallspartien decken Randfälle auf).
- **GreedyAgent** wählt kurzsichtig den Zug, der **die meisten gegnerischen
  Steine umdreht**: nur einen Halbzug tief gedacht.

Man würde erwarten, dass Greedy Random deutlich schlägt. Tatsächlich gewinnt er
nur **~61 %** der Partien (500 Spiele, `python scripts/arena.py`). Der Grund ist
lehrreich: **„so viele Steine wie möglich umdrehen" ist bei Othello eine schwache
Heuristik.** Wer früh viele Steine besitzt, steht oft *schlechter*, denn diese Steine
sind leicht zurückzuerobern. Es kommt nicht auf die Masse an, sondern auf
**Stabilität und Position**: Ecken (nie mehr umdrehbar), Kanten und die
Beweglichkeit (dem Gegner Züge nehmen). Die Steinzahl entscheidet erst am
Spielende.

Genau deshalb sind die folgenden Stufen so viel stärker: Sie optimieren nicht die
momentane Steinzahl, sondern die *Gewinnwahrscheinlichkeit*.

### MCTS – Monte Carlo Tree Search

MCTS findet gute Züge, ohne Othello-Wissen einprogrammiert zu bekommen. Statt den
(gigantischen) Spielbaum vollständig zu durchrechnen, spielt es von der aktuellen
Stellung aus **sehr oft zufällig zu Ende** und lernt daraus, welche Züge sich
lohnen. Die Rechenzeit fließt dorthin, wo es sich lohnt, d.h. vielversprechende Züge
werden tiefer untersucht, schlechte kaum.

Eine einzelne Simulation durchläuft vier Schritte, tausendfach wiederholt:

1. **Selection** – vom Wurzelknoten abwärts wandern, immer den „interessantesten"
   Kindknoten wählen. „Interessant" = bester **UCB1-Wert**: die bisherige
   Gewinnquote *plus* ein Bonus dafür, dass ein Zug selten probiert wurde (mehr
   dazu unten).
2. **Expansion** – am Rand des bekannten Baums einen neuen Knoten für einen noch
   nicht probierten Zug anhängen.
3. **Rollout** – von dort **komplett zufällig** bis zum Spielende weiterspielen;
   Ergebnis: Sieg oder Niederlage. (Der „Monte-Carlo"-Teil: Zufall statt teuer bis zum Ende rechnen.)
4. **Backpropagation** – das Ergebnis den durchlaufenen Pfad zurück nach oben
   tragen und je Knoten Besuchszahl und Gewinnbilanz aktualisieren.

Am Ende wird der **am häufigsten besuchte** Zug gespielt: die Besuchszahl ist das
robusteste Vertrauensmaß.

Der Kern steckt in der Auswahl (Schritt 1): die **UCB1**-Formel balanciert
*Exploitation* (Züge, die bisher gut liefen, weiterverfolgen) gegen *Exploration*
(selten probierte Züge trotzdem mal ansehen). Ein noch nie probierter Zug bekommt
den größtmöglichen Bonus und wird garantiert einmal ausprobiert; je öfter ein Zug
besucht wurde, desto kleiner sein Bonus und desto mehr zählt nur noch seine
Gewinnquote. So findet MCTS mit begrenzter Rechenzeit trotzdem starke Züge und
schlägt Greedy klar, ganz ohne ML. Damit ist bewiesen, dass Engine und Suche
korrekt zusammenspielen.

### AlphaZero (ML)

AlphaZero benutzt **denselben** MCTS-Kern wie oben: Selection, Expansion,
Backpropagation bleiben. Es ersetzt aber die zwei schwächsten Stellen durch ein
neuronales Netz:

- **Kein zufälliger Rollout mehr.** Statt von einem neuen Knoten aus zufällig zu
  Ende zu spielen, fragt AlphaZero das Netz *einmal*: „Wie gut steht dieser
  Spieler hier?" Diese gelernte Bewertung (**Value**) ersetzt tausende
  Zufallspartien durch einen einzigen, viel treffsichereren Blick.
- **Keine blinde Zug-Auswahl mehr.** Bei UCB1 startet jeder unprobierte Zug
  gleichberechtigt. AlphaZero bekommt vom Netz vorab eine Einschätzung, *welche*
  Züge überhaupt vielversprechend sind (**Policy**), und lenkt die Suche sofort
  dorthin. Die Auswahlformel heißt entsprechend **PUCT** (UCB1 + Policy-Prior).

Ergebnis: Wo reines MCTS Hunderte Simulationen braucht, reichen mit einem guten
Netz oft deutlich weniger für stärkere Züge. Die Rechenzeit wird nicht mehr in
Zufall verbrannt, sondern durch gelerntes Wissen geführt.

#### Das Netz

Das Modell ist ein kleines **ResNet** (Convolutional Neural Network mit
Residual-Blöcken), das eine Stellung in *zwei* Antworten übersetzt:

- **Eingabe:** die Stellung als 3 Ebenen à Brettgröße: eigene Steine,
  gegnerische Steine, und eine Ebene „wer ist am Zug". Wichtig: immer aus **Sicht
  des Ziehenden** kodiert, damit das Netz nur *eine* Bewertung lernen muss und
  nicht getrennt für Schwarz und Weiß.
- **Ausgabe 1 – Policy:** eine Wahrscheinlichkeit pro Feld (plus Pass = 0), also diese Züge sind nach dem, was das CNN gelernt hat am vielversprechendsten.
- **Ausgabe 2 – Value:** eine einzige Zahl in `[-1, +1]`: die vom CNN geschätzte
  Gewinnwahrscheinlichkeit aus Sicht des Ziehenden (`+1` = sicherer Sieg, `-1` =
  sichere Niederlage) von der Stellung aus, die nach diesem Zug bestehen würde.

Torso und beide Köpfe teilen sich dieselben Convolution-Schichten, das Netz baut
also *ein* Verständnis der Stellung auf und zapft es für beide Fragen an. Gelernt
werden ausschließlich die Gewichte dieser Schichten; kein Othello-Wissen
(Ecken, Stabilität …) ist vorgegeben.

#### Das Training – die KI spielt gegen sich selbst

Der Clou: Es gibt **keine menschlichen Partien** als Lehrmaterial. Das Netz
erzeugt seine Trainingsdaten selbst, in einem Kreislauf, der sich immer wieder
wiederholt (`scripts/train.py`):

1. **Self-Play.** Die KI spielt mit dem aktuell besten Netz gegen sich selbst.
   Pro Zug läuft ein MCTS-Suchlauf; gespeichert wird für jede Stellung die
   **Visit-Verteilung** der Suche (welche Züge wurden wie oft besucht) und später
   der **Ausgang** der Partie. Etwas Zufall an der Wurzel (*Dirichlet-Noise*) und
   eine „Temperatur" in der Eröffnung sorgen für Vielfalt, damit nicht immer
   dieselbe Partie entsteht.
2. **Die zwei Lernsignale.** Genau hier schließt sich der Kreis:
   - Die **Policy** lernt, die MCTS-Visit-Verteilung nachzuahmen. Die Suche ist
     stärker als das wenig trainierte Netz; das Netz destilliert also das Suchergebnis in
     sich hinein und trifft beim nächsten Mal schon *ohne* Suche bessere Vortipps.
   - Der **Value** lernt, den tatsächlichen Partie-Ausgang vorherzusagen: Stand
     die Stellung am Ende auf Sieg oder Niederlage?
3. **Training.** Aus einem **Replay-Buffer** der jüngsten Self-Play-Daten zieht
   der Optimizer zufällige Batches und passt die Netzgewichte an (Loss =
   Policy-Cross-Entropy + Value-MSE). Jedes Sample wird zusätzlich über die 8
   Symmetrien des Bretts gespiegelt/gedreht. Das bringt gratis das Achtfache an Daten.
4. **Gating.** Das frisch trainierte Netz muss sich beweisen: Es spielt gegen das
   bisher beste Netz. Nur wenn es eine klare Mehrheit der Partien gewinnt (Schwelle
   ~55 %), wird es zum neuen „besten" Netz befördert. Das verhindert, dass eine
   verrauschte Trainingsrunde die KI *schlechter* macht.

Warum wird sie dadurch besser? Stärkere Suche → bessere Trainingsdaten → stärkeres
Netz → das macht die *nächste* Suche noch stärker. Diese Rückkopplung schaukelt
sich hoch: Das Netz zieht sich an den eigenen Suchergebnissen selbst nach oben,
von planlosem Anfangsgeklopfe bis zu echtem Stellungsverständnis – ganz ohne
menschliche Vorlage.

#### Ergebnisse (6×6-Durchstich)

Setup: 6×6-Othello, kleines ResNet (64 Kanäle, 4 Residual-Blöcke), 64 MCTS-Sims pro
Zug, 40 Iterationen à 20 Self-Play-Partien (~23 s/Iteration, ~15 min gesamt auf einer
RTX-50-GPU). Gemessen mit `python scripts/measure.py` (40 Partien pro Match, Netz mit
64 Sims):

| Gegner | Quote des Netzes | W/L/D |
|---|---|---|
| Random | 100 % | 40/0/0 |
| Greedy | 100 % | 40/0/0 |
| reines MCTS, 50 Sims | 87.5 % | 35/5/0 |
| reines MCTS, 150 Sims | 85.0 % | 34/6/0 |
| reines MCTS, 400 Sims | 55.0 % | 21/17/2 |
| eigenes Netz nach 5 Iterationen | 91.2 % | 36/3/1 |

Zwei Dinge sind ablesbar:

- **Das gelernte Wissen ist echten Suchaufwand wert.** Das Netz spielt mit nur
  **64 Sims** ungefähr auf Augenhöhe mit reinem MCTS bei **400 Sims** (55 %) und
  schlägt es bei 150 Sims klar. Policy- und Value-Kopf ersetzen also grob den
  ~6-fachen Rollout-Aufwand – genau der Sinn von AlphaZero.
- **Es ist über die Iterationen messbar stärker geworden:** gegen ein *frühes
  eigenes Ich* (Iteration 5) gewinnt das Endmodell **91 %**. Das ist wichtig, weil
  die Quote gegen Greedy schon ab Iteration 1 bei 100 % klebt (Greedy ist zu
  schwach als Maßstab) und das Self-Play-Gating gegen Ende bei ~50 % pendelt – das
  ist die *Signatur von Konvergenz* (neues ≈ bestes Netz), kein Stillstand. Der
  Loss läuft entsprechend ab ~Iteration 20 in ein Plateau; die letzten Iterationen
  fügen wenig hinzu. **Damit ist die Pipeline validiert (Meilenstein 2).**

## Performance & Skalierung (6×6 → 8×8)

Der 6×6-Durchstich läuft in ~23 s pro Iteration (40 Iterationen ≈ 15 min). Das ist
erst nach zwei Batching-Schritten so schnell: Self-Play und Evaluation liefen
ursprünglich mit **Batch-Größe 1 pro MCTS-Simulation**, d. h. für jede einzelne
Stellung ging *ein* Brett auf die GPU und der Prozess wartete auf das Ergebnis
(CPU↔GPU-Sync). Bei einem winzigen Netz ist diese feste Latenz der ganze
Flaschenhals – die GPU langweilt sich (~30 % Auslastung).

**Was schon umgesetzt ist:** Self-Play (`az/selfplay_parallel.py`) und Eval-Arena
(`az/arena_parallel.py`) spielen viele Partien gleichzeitig und bündeln pro Runde
die Blatt-Bewertungen aller Partien in *einen* Forward-Pass. Messung: **~7×** beim
Self-Play, **~5×** bei der Eval.

**Warum die GPU trotzdem nur ~30 % zeigt:** Der Engpass ist jetzt die
**Single-Core-Python-Arbeit** (MCTS-Baumtraversierung + Engine-Ops), nicht die GPU.
Bei so kleinem Netz ist der Forward-Pass zu billig, um die GPU zu sättigen – und
das ist in Ordnung: Zielgröße ist die **Wall-Clock-Zeit pro Iteration**, nicht die
GPU-Prozentzahl.

Für **8×8** wird es spürbar langsamer – nicht wegen der GPU, sondern weil fast
alles, was 8×8 teurer macht, genau die CPU-Seite trifft: ~doppelt so lange Partien,
größerer Verzweigungsgrad, teurere `legal_moves`/`apply`, dazu mehr Sims und mehr
Iterationen. Realistisch Minuten pro Iteration, Stunden bis ~1 Tag gesamt. Die
Optimierungshebel, nach Wirkung/Aufwand:

| Hebel | Wirkung | Aufwand | Risiko | Status |
|---|---|---|---|---|
| **Batched Inferenz** (Self-Play + Eval parallel, Blätter bündeln) | hoch | – | – | **erledigt** |
| **A – `n_parallel` & `games_per_iteration` hoch** (z. B. 64–128) | mittel–hoch (nutzt GPU-Reserve + CPU/GPU-Überlappung) | null (Config) | keins | offen |
| **B – Sims/Iterationen bewusst wählen** (Plateau meiden, nicht über-trainieren) | mittel | null | keins | offen |
| **C – Multiprocessing-Self-Play** über CPU-Kerne | **hoch** (× Kernzahl; idle Cores sind gratis Leistung) | mittel | mittel | geplant |
| **D – Engine JIT-en (Numba)** auf `legal_moves`/`apply` | **hoch** (heißester Single-Core-Pfad) | mittel | mittel (Tests als Netz) | geplant |
| **E – Batched Engine** (alle Bretter als ein `(G,8,8)`-Array, Move-Gen vektorisiert) | hoch | hoch | mittel | Reserve |
| **F – Bitboard** statt NumPy (Move-Gen als Bit-Ops) | mittel–hoch (überlappt mit D) | hoch | hoch (Engine-Rewrite) | Reserve |
| **G – Größeres Netz** | *kein* Speedup, aber „gratis" dank GPU-Reserve → mehr Stärke | niedrig | keins | für 8×8 geplant |

**Empfohlenes Vorgehen für 8×8:** erst einen kurzen Mess-Lauf (5 Iterationen) für
echte Zahlen, dann **A + B** (gratis) und ein **größeres Netz (G)**. Reicht das
nicht, **C und/oder D** – beide greifen die Single-Core-Wurzel an, von
verschiedenen Seiten. E/F sind Reserve für den Fall, dass C+D nicht genügen; für
den Projektumfang voraussichtlich nicht nötig. Nicht lohnend: GPU-% direkt jagen
oder das Netz mikro-optimieren – beides zielt am Engpass vorbei.

## Status

Phase 2 (AlphaZero-Pipeline) validiert: 6×6-Training läuft und lernt nachweislich
(Meilenstein 2). Self-Play und Evaluation sind gebündelt/parallelisiert. Als
Nächstes: Skalierung auf 8×8 (Schritt 2.7).
