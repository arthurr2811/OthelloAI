"""Tests für Schritt 2.6: Orchestrierung des Trainings-Loops.

Nur Verkabelung, kein echtes Training: winzige Config, wenige Sims/Partien. Prüft,
dass eine Iteration durchläuft, Checkpoints entstehen und ein Resume funktioniert.
"""

from __future__ import annotations

import torch

from az.checkpoint import load_checkpoint
from az.net import OthelloNet
from az.pipeline import clone_net, run_training
from config import EvalConfig, MCTSConfig, RunConfig


def _tiny_config(tmp_path, n_iterations=1) -> RunConfig:
    return RunConfig(
        board_size=6,
        n_iterations=n_iterations,
        games_per_iteration=1,
        train_steps_per_iteration=3,
        baseline_games=2,
        mcts=MCTSConfig(n_simulations=4),
        eval=EvalConfig(n_games=2, win_threshold=0.55, temperature_moves=3),
        checkpoint_dir=str(tmp_path / "ckpt"),
        log_dir=str(tmp_path / "logs"),
    )


def test_clone_net_is_independent():
    net = OthelloNet(board_size=6)
    clone = clone_net(net)
    x = torch.randn(2, 3, 6, 6)
    net.eval(); clone.eval()
    with torch.no_grad():
        assert torch.allclose(net(x)[0], clone(x)[0], atol=1e-6)
    # Verändert man das Original, bleibt der Klon unberührt.
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
        assert not torch.allclose(net(x)[0], clone(x)[0])


def test_single_iteration_runs_and_writes_artifacts(tmp_path):
    config = _tiny_config(tmp_path, n_iterations=1)
    net = run_training(config, device="cpu", log=lambda *a, **k: None)
    assert isinstance(net, OthelloNet)
    # Checkpoints + Logs sind entstanden.
    assert (tmp_path / "ckpt" / "best.pt").exists()
    assert (tmp_path / "ckpt" / "iter_000.pt").exists()
    iterations_csv = (tmp_path / "logs" / "iterations.csv").read_text().strip().splitlines()
    assert iterations_csv[0].startswith("iteration,")
    assert len(iterations_csv) == 2                       # Header + 1 Iteration


def test_resume_continues_from_checkpoint(tmp_path):
    config = _tiny_config(tmp_path, n_iterations=1)
    run_training(config, device="cpu", log=lambda *a, **k: None)

    # best.pt trägt die letzte Iterationsnummer; Resume muss danach weiterzählen.
    _, extra = load_checkpoint(tmp_path / "ckpt" / "best.pt", "cpu")
    assert "iteration" in extra

    resume_config = _tiny_config(tmp_path, n_iterations=2)
    run_training(resume_config, resume=str(tmp_path / "ckpt" / "best.pt"),
                device="cpu", log=lambda *a, **k: None)
    # Nach dem Resume-Lauf existiert der Checkpoint der zweiten Iteration.
    assert (tmp_path / "ckpt" / "iter_001.pt").exists()
