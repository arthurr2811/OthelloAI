"""Einstiegspunkt: lässt Baseline-Bots gegeneinander antreten.

Aufruf (aus dem Projektwurzelverzeichnis):
    python scripts/arena.py
    python scripts/arena.py --games 200 --size 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Projektwurzel in den Pfad, damit `python scripts/arena.py` direkt läuft.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.arena import play_match  # noqa: E402
from agents.simple import GreedyAgent, RandomAgent  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Othello-Baseline-Arena")
    parser.add_argument("--games", type=int, default=500, help="Anzahl Partien")
    parser.add_argument("--size", type=int, default=8, help="Brettgröße (gerade)")
    parser.add_argument("--seed", type=int, default=0, help="Basis-Seed")
    args = parser.parse_args()

    random_bot = RandomAgent(seed=args.seed)
    greedy_bot = GreedyAgent(seed=args.seed + 1)

    result = play_match(greedy_bot, random_bot, n_games=args.games, size=args.size)
    print(result.summary())

    # Greedy (max Flips) ist bei Othello nur mäßig stark und liegt real bei
    # ~61% gegen Random. Schwelle mit Varianz-Puffer statt harter 60%-Kante.
    threshold = 0.55
    ok = result.win_rate > threshold
    verdict = "OK" if ok else "ZU SCHWACH"
    print(f"Sanity-Check (Greedy > {threshold:.0%}): {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
