"""Einstiegspunkt für den AlphaZero-Trainings-Loop (Schritt 2.6).

Standard: 6x6-Durchstich mit den Defaults aus :class:`config.RunConfig`. Alle
wichtigen Größen sind per Flag überschreibbar, ohne die Config anzufassen.

Aufruf:
    python scripts/train.py                       # 6x6, Default-Config
    python scripts/train.py --iterations 60       # mehr Iterationen
    python scripts/train.py --resume checkpoints/best.pt
    python scripts/train.py --smoke               # winziger Wiring-Check (kein echtes Training)

    python scripts/train.py --preset 8x8 --iterations 5   # 8x8-Mess-Lauf (echte Zahlen)
    python scripts/train.py --preset 8x8                  # 8x8-Vollauf (Schritt 2.7)
    python scripts/train.py --preset 8x8 --resume checkpoints/8x8/best.pt

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
from config import DEFAULT_RUN, RUN_8X8, EvalConfig, MCTSConfig, RunConfig  # noqa: E402

PRESETS = {"6x6": DEFAULT_RUN, "8x8": RUN_8X8}


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
    parser.add_argument("--preset", choices=sorted(PRESETS), default="6x6",
                        help="Basis-Config: 6x6-Durchstich oder 8x8-Zielkonfiguration")
    # Feineinstellungen: Default None = Preset-Wert gilt; nur explizit gesetzte
    # Flags überschreiben (sonst würden 6x6-Defaults das 8x8-Preset überdecken).
    parser.add_argument("--size", type=int, default=None, help="Brettgröße")
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
        base = PRESETS[args.preset]
        overrides = {}
        if args.size is not None:
            overrides["board_size"] = args.size
        if args.iterations is not None:
            overrides["n_iterations"] = args.iterations
        if args.games is not None:
            overrides["games_per_iteration"] = args.games
        if args.train_steps is not None:
            overrides["train_steps_per_iteration"] = args.train_steps
        if args.seed is not None:
            overrides["seed"] = args.seed
        if args.sims is not None:
            overrides["mcts"] = replace(base.mcts, n_simulations=args.sims)
        if args.workers is not None:
            overrides["selfplay"] = replace(base.selfplay, n_workers=args.workers)
        config = replace(base, **overrides)

    run_training(config, resume=args.resume, device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
