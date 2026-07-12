"""Orchestrierung des AlphaZero-Loops: Self-Play -> Train -> Evaluate -> Gate.

Eine Iteration:
    1. **Self-Play** mit dem aktuellen Bestmodell (Wurzel-Rauschen an) füllt den
       Replay-Buffer mit augmentierten Samples.
    2. **Training**: ein Kandidat (Kopie des Bestmodells) lernt N Schritte auf dem
       Buffer.
    3. **Gating**: Kandidat vs. Bestmodell. Übersteigt die Quote die Schwelle,
       wird der Kandidat neues Bestmodell (Checkpoint ``best.pt``).
    4. **Baseline**: Kandidat vs. Greedy → absolute Stärke-Kurve.
    5. Iterations-Zeile ins CSV loggen.

Der Loop lässt sich aus ``best.pt`` fortsetzen (``resume``); die Iterationsnummer
steckt in den Checkpoint-Metadaten.
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from az.checkpoint import load_checkpoint, save_checkpoint
from az.evaluate import evaluate_vs_baseline, gate
from az.net import OthelloNet
from az.replay import ReplayBuffer
from az.selfplay_mp import SelfPlayPool
from az.selfplay_parallel import generate_games_parallel
from az.train import Trainer
from config import DEFAULT_RUN, RunConfig

from agents.simple import GreedyAgent


def clone_net(net: OthelloNet) -> OthelloNet:
    """Erzeugt eine unabhängige Kopie eines Netzes (gleiche Gewichte, gleiches Device)."""
    clone = OthelloNet(net.board_size, net.config)
    clone.load_state_dict(net.state_dict())
    clone.to(next(net.parameters()).device)
    return clone


def _select_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _IterationLogger:
    """CSV-Logger für die Iterations-Zusammenfassung (append-fähig für Resume)."""

    FIELDS = (
        "iteration", "buffer_size", "loss_start", "loss_end",
        "gate_win_rate", "accepted", "baseline_win_rate", "seconds",
    )

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(self.FIELDS)

    def write(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(row)


def _init_best_net(
    config: RunConfig, device: torch.device, resume: str | Path | None, log
) -> tuple[OthelloNet, int]:
    """Lädt das Bestmodell (Resume) oder legt ein frisches an. Rückgabe: (net, start_iteration)."""
    checkpoint_dir = Path(config.checkpoint_dir)
    if resume is not None:
        best_net, extra = load_checkpoint(resume, device)
        start = int(extra.get("iteration", -1)) + 1
        log(f"Resume aus {resume} (weiter ab Iteration {start}).")
        return best_net, start

    best_net = OthelloNet(config.board_size, config.net).to(device)
    save_checkpoint(best_net, checkpoint_dir / "best.pt", extra={"iteration": -1})
    log(f"Frisches Netz angelegt, Startgewichte -> {checkpoint_dir / 'best.pt'}.")
    return best_net, 0


def run_training(
    config: RunConfig = DEFAULT_RUN,
    *,
    resume: str | Path | None = None,
    device: str | torch.device | None = None,
    log=print,
) -> OthelloNet:
    """Führt den vollständigen Trainings-Loop aus und gibt das finale Bestmodell zurück.

    Achtung: langlaufend. Für einen Wiring-Check eine ``RunConfig`` mit winzigen
    Werten (wenige Iterationen/Sims/Partien) übergeben.
    """
    device = _select_device(device)
    log(f"Device: {device} | Brett {config.board_size}x{config.board_size} | "
        f"{config.n_iterations} Iterationen")

    checkpoint_dir = Path(config.checkpoint_dir)
    log_dir = Path(config.log_dir)
    rng = np.random.default_rng(config.seed)

    best_net, start_iter = _init_best_net(config, device, resume, log)
    buffer = ReplayBuffer(config.selfplay.buffer_size)
    iter_logger = _IterationLogger(log_dir / "iterations.csv")

    # Hebel C: ab 2 Workern verteilt ein persistenter Prozess-Pool die
    # Self-Play-Partien über die CPU-Kerne (az/selfplay_mp.py). try/finally,
    # damit die Worker-Prozesse auch bei Abbruch (Ctrl+C) beendet werden.
    pool = SelfPlayPool(config.selfplay.n_workers) if config.selfplay.n_workers > 1 else None
    if pool is not None:
        log(f"Self-Play-Pool: {config.selfplay.n_workers} Worker-Prozesse")

    try:
        for it in range(start_iter, config.n_iterations):
            t0 = time.time()
            log(f"\n=== Iteration {it}/{config.n_iterations - 1} ===")

            # 1. Self-Play mit dem Bestmodell – gebündelt/parallel, damit die GPU
            #    ausgelastet ist (statt Batch-1-Inferenz pro Simulation); ab 2
            #    Workern zusätzlich über CPU-Kerne verteilt.
            if pool is not None:
                pool.generate(
                    best_net, config.board_size, config.games_per_iteration,
                    mcts_config=config.mcts, config=config.selfplay, device=device,
                    seed=config.seed + it * 1000, buffer=buffer,
                )
            else:
                generate_games_parallel(
                    best_net, config.board_size, config.games_per_iteration,
                    mcts_config=config.mcts, config=config.selfplay, device=device,
                    seed=config.seed + it * 1000, buffer=buffer,
                )
            log(f"Self-Play: {config.games_per_iteration} Partien, Buffer={len(buffer)}")

            # 2. Training eines Kandidaten (Kopie des Bestmodells).
            candidate = clone_net(best_net)
            trainer = Trainer(candidate, config.train, device=device)
            history = trainer.train(
                buffer, config.train_steps_per_iteration, rng,
                log_path=log_dir / f"loss_iter_{it:03d}.csv",
            )
            loss_start = float(np.mean([h["total"] for h in history[:5]]))
            loss_end = float(np.mean([h["total"] for h in history[-5:]]))
            log(f"Training: {config.train_steps_per_iteration} Schritte, "
                f"Loss {loss_start:.3f} -> {loss_end:.3f}")

            # 3. Gating gegen das Bestmodell.
            gating = gate(
                candidate, best_net, config.board_size,
                mcts_config=config.mcts, eval_config=config.eval, device=device,
                seed=config.seed + it,
            )
            log(f"Gating: {gating.summary()}")

            # 4. Baseline-Stärke (Kandidat vs. Greedy).
            baseline = evaluate_vs_baseline(
                candidate, GreedyAgent(seed=config.seed + it), config.board_size,
                n_games=config.baseline_games, mcts_config=config.mcts,
                eval_config=config.eval, device=device, seed=config.seed + it,
            )
            log(f"Baseline vs. Greedy: {baseline.win_rate:.1%}")

            # 5. Bestmodell aktualisieren, falls angenommen.
            if gating.accepted:
                best_net = candidate
                save_checkpoint(best_net, checkpoint_dir / "best.pt", extra={"iteration": it})
                log("-> neues Bestmodell gespeichert.")
            save_checkpoint(candidate, checkpoint_dir / f"iter_{it:03d}.pt", extra={"iteration": it})

            iter_logger.write({
                "iteration": it,
                "buffer_size": len(buffer),
                "loss_start": round(loss_start, 4),
                "loss_end": round(loss_end, 4),
                "gate_win_rate": round(gating.win_rate, 4),
                "accepted": int(gating.accepted),
                "baseline_win_rate": round(baseline.win_rate, 4),
                "seconds": round(time.time() - t0, 1),
            })
    finally:
        if pool is not None:
            pool.close()

    log("\nTraining abgeschlossen.")
    log(f"Konfiguration: {asdict(config)}")
    return best_net
