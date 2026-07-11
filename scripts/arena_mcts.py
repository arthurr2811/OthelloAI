"""Einstiegspunkt: MCTS gegen Greedy (voller Stärke-Check aus Schritt 1.5).

Bewusst getrennt von scripts/arena.py, weil MCTS deutlich langsamer ist.

Aufruf:
    python scripts/arena_mcts.py
    python scripts/arena_mcts.py --games 30 --sims 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.arena import play_match  # noqa: E402
from agents.mcts import MCTSAgent  # noqa: E402
from agents.simple import GreedyAgent  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="MCTS vs. Greedy")
    parser.add_argument("--games", type=int, default=30, help="Anzahl Partien")
    parser.add_argument("--sims", type=int, default=150, help="Simulationen pro Zug")
    parser.add_argument("--size", type=int, default=8, help="Brettgröße (gerade)")
    parser.add_argument("--seed", type=int, default=0, help="Basis-Seed")
    args = parser.parse_args()

    mcts = MCTSAgent(n_simulations=args.sims, seed=args.seed)
    greedy = GreedyAgent(seed=args.seed + 1)

    t = time.time()
    result = play_match(mcts, greedy, n_games=args.games, size=args.size)
    dt = time.time() - t

    print(result.summary())
    print(f"{dt:.1f}s ({dt / args.games * 1000:.0f} ms/Spiel, {args.sims} Sims/Zug)")

    threshold = 0.70
    ok = result.win_rate > threshold
    print(f"Sanity-Check (MCTS > {threshold:.0%}): {'OK' if ok else 'ZU SCHWACH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
