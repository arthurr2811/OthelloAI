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

#### Ergebnisse

*Folgt nach dem 6×6-Trainingslauf: Stärke-Kurven gegen die Baselines und über die
Iterationen, plus die entscheidenden Trainingsparameter.*

## Status

Phase 1 (Engine & Baselines) in Arbeit: Engine getestet, Random-/Greedy-Baselines
und Arena stehen. Als Nächstes: MCTS ohne Netz (Schritt 1.5).
