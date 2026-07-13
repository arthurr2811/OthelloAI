# Othello AI

Eine AlphaZero-artige Othello-KI (8×8), lokal auf GPU trainiert – ohne
menschliche Partien, ohne einprogrammiertes Othello-Wissen. Dazu ein
Web-Frontend zum Selberspielen mit einstellbarer Spielstärke.

Vorgehen und Projektfortschritt: siehe [`plan.md`](plan.md).

## Setup

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# PyTorch passend zur GPU (RTX-50-Serie braucht das CUDA-12.8-Wheel):
pip install torch --index-url https://download.pytorch.org/whl/cu128

python scripts/check_env.py   # erwartet: CUDA verfügbar = True
pytest
```

## Projektstruktur

```
othello/   # Spiel-Engine (reine Logik, kein ML) + Numba-Kernel für die Hotpaths
agents/    # Referenzgegner: Random, Greedy, reines MCTS
az/        # AlphaZero: Netz, PUCT-MCTS, Self-Play, Training, Evaluation, Pipeline
web/       # FastAPI-Backend (server.py) + statisches Frontend (static/)
scripts/   # train.py (Training), measure.py (Stärke messen), serve.py (Frontend), check_env.py
tests/     # pytest-Suite (Engine, Suche, Pipeline, Kernel-Äquivalenz)
config.py  # zentrale Trainingskonfiguration (die Defaults = der echte 8x8-Lauf)
```

## Wie die KI funktioniert

**MCTS als Grundgerüst.** Statt den Spielbaum vollständig zu durchrechnen,
untersucht Monte-Carlo-Baumsuche gezielt die vielversprechendsten Züge und
steckt Rechenzeit dorthin, wo sie sich lohnt. Ein reines MCTS mit
Zufalls-Rollouts (in `agents/mcts.py`) schlägt die einfachen Baselines bereits
deutlich – der Beweis, dass Engine und Suche korrekt zusammenspielen, ganz ohne ML.

**AlphaZero = MCTS + gelerntes Netz.** Ein kleines ResNet (`az/net.py`) ersetzt
die zwei schwächsten Stellen der reinen Suche:

- **Value-Kopf** statt Zufalls-Rollouts: „Wie gut steht der Spieler am Zug?" –
  eine gelernte Bewertung in `[-1, 1]` statt tausender Zufallspartien.
- **Policy-Kopf** statt blinder Zugauswahl: „Welche Züge sind vielversprechend?" –
  Priors, die die Suche sofort in gute Richtungen lenken (PUCT-Formel).

Die Eingabe sind 3 Ebenen (eigene Steine, gegnerische Steine, wer am Zug ist),
immer aus Sicht des Ziehenden.

**Training durch Self-Play** (`az/pipeline.py`, pro Iteration):

1. **Self-Play:** Das beste Netz spielt gegen sich selbst. Pro Stellung wird die
   MCTS-Visit-Verteilung gespeichert, am Partieende der Ausgang. Dirichlet-Rauschen
   und eine Eröffnungs-Temperatur sorgen für Vielfalt.
2. **Training:** Die Policy lernt die (stärkere) Suchverteilung nachzuahmen, der
   Value den Partie-Ausgang vorherzusagen. Jedes Sample wird über die 8
   Brett-Symmetrien verachtfacht.
3. **Gating:** Der frisch trainierte Kandidat muss das bisherige Bestmodell über
   ≥ 55 % der Partien schlagen, sonst wird er verworfen – verrauschte Runden
   können die KI so nicht verschlechtern.

Die Rückkopplung „stärkere Suche → bessere Daten → stärkeres Netz → stärkere
Suche" schaukelt sich von Zufallsspiel zu echtem Stellungsverständnis hoch.

## Training ausführen

```bash
python scripts/train.py                                # Vollauf: 120 Iterationen, ~4 h
python scripts/train.py --resume checkpoints/best.pt   # abgebrochenen Lauf fortsetzen
python scripts/train.py --smoke                        # schneller Wiring-Check
python scripts/measure.py                              # Stärke: vs Random/Greedy/MCTS
```

Alle Parameter (Netzgröße, Sims, Partien, Worker …) liegen in `config.py`;
jede Iteration schreibt Checkpoint + Kennzahlen (`logs/iterations.csv`).

## Performance

Der Engpass des Trainings ist nicht die GPU, sondern Single-Core-Python
(MCTS-Baum + Engine-Operationen). Drei Maßnahmen zusammen machen den
8×8-Lauf praktikabel (~114 s/Iteration statt hochgerechnet > 5 min):

1. **Gebündelte Inferenz:** Viele Partien laufen verzahnt; alle anstehenden
   Blatt-Bewertungen einer Runde gehen als *ein* Batch auf die GPU
   (`az/selfplay_parallel.py`, `az/arena_parallel.py`).
2. **Numba-JIT-Kernel** für die heißen Engine-Pfade `legal_moves`/`apply_move`
   (`othello/_kernels.py`); die reine Python-Engine bleibt als Referenz, ein
   Äquivalenz-Test prüft beide gegeneinander.
3. **Multiprocessing-Self-Play:** Ein persistenter Pool aus 6 Worker-Prozessen
   teilt sich die Partien jeder Iteration (`az/selfplay_mp.py`).

## Ergebnisse

**6×6-Durchstich (Pipeline-Validierung):** Nach 40 Iterationen (~15 min) schlägt
das Netz mit 64 Sims/Zug Random und Greedy zu 100 %, reines MCTS mit 150 Sims zu
85 % und spielt gegen reines MCTS mit 400 Sims noch 55 % – das gelernte Wissen
ersetzt also grob den 6-fachen Suchaufwand. Gegen sein eigenes früheres Ich
(Iteration 5) gewinnt es 91 % → messbarer Fortschritt über die Iterationen.

**8×8 (Ziellauf):** 120 Iterationen in **4,1 h** Wall-Clock (Ryzen 7 7800X3D +
RTX 5070 Ti), Loss 5,15 → ~1,88, 52 der 120 Kandidaten via Gating angenommen.
Das fertige Netz (128 Sims/Zug, `scripts/measure.py`):

| Gegner              | Siegquote | W/L/D    |
|---------------------|-----------|----------|
| Random              | 100 %     | 40/0/0   |
| Greedy              | 100 %     | 40/0/0   |
| reines MCTS, 50 Sims  | 100 %   | 40/0/0   |
| reines MCTS, 150 Sims | 100 %   | 40/0/0   |
| reines MCTS, 400 Sims | 92,5 %  | 37/3/0   |

Der aussagekräftige Wert ist die letzte Zeile: Das Netz schlägt reines MCTS mit
dem **~3-fachen Suchbudget** klar – das gelernte Stellungsverständnis wiegt den
Rechenvorteil des Gegners mehr als auf.

## Selber spielen (Web-Frontend)

```bash
python scripts/serve.py --checkpoint scripts/checkpoints/best.pt
python scripts/serve.py                                   # Kurzform, lädt checkpoints/best.pt
```



Dann [http://127.0.0.1:8000](http://127.0.0.1:8000) im Browser öffnen. Ein
FastAPI-Backend (`web/server.py`) lädt den Checkpoint und liefert die KI-Züge
(PUCT-MCTS + Netz); das Frontend (`web/static/`) ist reines HTML/CSS/JS ohne
Build-Schritt.

## Status

- Engine, Baselines, reines MCTS: fertig und getestet (Meilenstein 1)
- AlphaZero-Pipeline auf 6×6 validiert (Meilenstein 2)
- 8×8-Ziellauf abgeschlossen: schlägt reines MCTS(400) zu 92,5 % (Schritt 2.7)
- Web-Frontend zum Selberspielen mit einstellbarer Stärke: spielbar (Phase 3)
