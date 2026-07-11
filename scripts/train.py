"""Einstiegspunkt für den AlphaZero-Trainings-Loop (Schritt 2.6).

Standard: 6x6-Durchstich mit den Defaults aus :class:`config.RunConfig`. Alle
wichtigen Größen sind per Flag überschreibbar, ohne die Config anzufassen.

Aufruf:
    python scripts/train.py                       # 6x6, Default-Config
    python scripts/train.py --iterations 60       # mehr Iterationen
    python scripts/train.py --resume checkpoints/best.pt
    python scripts/train.py --smoke               # winziger Wiring-Check (kein echtes Training)

Achtung: Der echte Lauf ist langlaufend (Stunden). Für einen schnellen Test der
Verkabelung ``--smoke`` benutzen.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from az.pipeline import run_training  # noqa: E402
from config import DEFAULT_RUN, EvalConfig, MCTSConfig, RunConfig  # noqa: E402


def _smoke_config() -> RunConfig:
    """Winzige Config: ein paar Iterationen, wenige Sims/Partien – nur zum Testen."""
    return RunConfig(
        board_size=6,
        n_iterations=2,
        games_per_iteration=2,
        train_steps_per_iteration=20,
        baseline_games=4,
        mcts=MCTSConfig(n_simulations=8),
        eval=EvalConfig(n_games=4, win_threshold=0.55, temperature_moves=4),
        checkpoint_dir="checkpoints/smoke",
        log_dir="logs/smoke",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="AlphaZero-Training (Othello)")
    parser.add_argument("--size", type=int, default=DEFAULT_RUN.board_size, help="Brettgröße")
    parser.add_argument("--iterations", type=int, default=DEFAULT_RUN.n_iterations)
    parser.add_argument("--games", type=int, default=DEFAULT_RUN.games_per_iteration,
                        help="Self-Play-Partien pro Iteration")
    parser.add_argument("--train-steps", type=int, default=DEFAULT_RUN.train_steps_per_iteration)
    parser.add_argument("--sims", type=int, default=DEFAULT_RUN.mcts.n_simulations,
                        help="MCTS-Simulationen pro Zug")
    parser.add_argument("--seed", type=int, default=DEFAULT_RUN.seed)
    parser.add_argument("--resume", type=str, default=None, help="Pfad zu best.pt zum Fortsetzen")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (Default: auto)")
    parser.add_argument("--smoke", action="store_true", help="Winziger Wiring-Check statt echtem Lauf")
    args = parser.parse_args()

    if args.smoke:
        config = _smoke_config()
    else:
        config = replace(
            DEFAULT_RUN,
            board_size=args.size,
            n_iterations=args.iterations,
            games_per_iteration=args.games,
            train_steps_per_iteration=args.train_steps,
            seed=args.seed,
            mcts=replace(DEFAULT_RUN.mcts, n_simulations=args.sims),
        )

    run_training(config, resume=args.resume, device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
