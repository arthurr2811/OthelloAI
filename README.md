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

### AlphaZero (ML) – *folgt in Phase 2*

AlphaZero benutzt **denselben** MCTS-Kern, ersetzt aber die zwei schwächsten
Stellen durch ein neuronales Netz: den zufälligen Rollout durch eine gelernte
Stellungsbewertung (*Value*) und die blinde Zug-Auswahl durch gelernte
Vorzugsrichtungen (*Policy*). Dieser Abschnitt wird ergänzt, sobald die Pipeline
steht.

## Status

Phase 1 (Engine & Baselines) in Arbeit: Engine getestet, Random-/Greedy-Baselines
und Arena stehen. Als Nächstes: MCTS ohne Netz (Schritt 1.5).
