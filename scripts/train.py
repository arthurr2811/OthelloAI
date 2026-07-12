"""Einstiegspunkt für den AlphaZero-Trainings-Loop (8x8-Othello).

Die Defaults aus :class:`config.RunConfig` sind die echte Trainingskonfiguration;
alle wichtigen Größen sind per Flag überschreibbar.

Aufruf:
    python scripts/train.py                                # Vollauf (~3,8 h)
    python scripts/train.py --iterations 5                 # kurzer Mess-Lauf
    python scripts/train.py --resume checkpoints/best.pt   # Lauf fortsetzen
    python scripts/train.py --smoke                        # Wiring-Check (Minuten)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from az.pipeline import run_training  # noqa: E402
from config import DEFAULT_RUN, EvalConfig, MCTSConfig, NetConfig, RunConfig, SelfPlayConfig  # noqa: E402


def _smoke_config() -> RunConfig:
    """Winzige Config für den Wiring-Check: alle Pfade (inkl. Worker-Pool),
    aber Mini-Budgets und ein Mini-Netz – kein echtes Training."""
    return RunConfig(
        n_iterations=2,
        games_per_iteration=4,
        train_steps_per_iteration=20,
        baseline_games=4,
        net=NetConfig(channels=16, n_res_blocks=2, value_hidden=16),
        mcts=MCTSConfig(n_simulations=8),
        selfplay=SelfPlayConfig(temperature_moves=4, n_parallel=4, n_workers=2),
        eval=EvalConfig(n_games=4, win_threshold=0.55, temperature_moves=4),
        checkpoint_dir="checkpoints/smoke",
        log_dir="logs/smoke",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="AlphaZero-Training (Othello 8x8)")
    # Default None = Wert aus RunConfig gilt; nur explizit gesetzte Flags überschreiben.
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--games", type=int, default=None,
                        help="Self-Play-Partien pro Iteration")
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--sims", type=int, default=None,
                        help="MCTS-Simulationen pro Zug")
    parser.add_argument("--workers", type=int, default=None,
                        help="Self-Play-Prozesse (1 = kein Multiprocessing)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Pfad zu best.pt zum Fortsetzen")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (Default: auto)")
    parser.add_argument("--smoke", action="store_true", help="Winziger Wiring-Check statt echtem Lauf")
    args = parser.parse_args()

    if args.smoke:
        config = _smoke_config()
    else:
        overrides = {}
        if args.iterations is not None:
            overrides["n_iterations"] = args.iterations
        if args.games is not None:
            overrides["games_per_iteration"] = args.games
        if args.train_steps is not None:
            overrides["train_steps_per_iteration"] = args.train_steps
        if args.seed is not None:
            overrides["seed"] = args.seed
        if args.sims is not None:
            overrides["mcts"] = replace(DEFAULT_RUN.mcts, n_simulations=args.sims)
        if args.workers is not None:
            overrides["selfplay"] = replace(DEFAULT_RUN.selfplay, n_workers=args.workers)
        config = replace(DEFAULT_RUN, **overrides)

    run_training(config, resume=args.resume, device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
