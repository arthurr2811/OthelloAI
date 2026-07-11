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

## Status

Phase 0 (Setup) abgeschlossen. Nächster Schritt: Phase 1 – Engine & Baselines.
