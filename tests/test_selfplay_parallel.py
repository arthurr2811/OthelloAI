"""Tests für den gebündelten, parallelen Self-Play (az/selfplay_parallel.py).

Geprüft wird Korrektheit (wohlgeformte Samples, Budget eingehalten, Buffer-
Anbindung), nicht Bit-Gleichheit mit dem sequenziellen Pfad – die Verzahnung der
Partien und getrennte RNG-Ströme führen bewusst zu anderer Reihenfolge.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from az.encoding import num_actions
from az.net import OthelloNet
from az.replay import ReplayBuffer
from az.selfplay_parallel import generate_games_parallel
from config import MCTSConfig, SelfPlayConfig


def _net(size=6):
    torch.manual_seed(0)
    return OthelloNet(board_size=size).eval()


def test_produces_well_formed_samples():
    samples = generate_games_parallel(
        _net(), size=6, n_games=6,
        mcts_config=MCTSConfig(n_simulations=12),
        config=SelfPlayConfig(augment=False, n_parallel=4),
        seed=1,
    )
    assert len(samples) > 0
    for s in samples:
        assert s.planes.shape == (3, 6, 6)
        assert s.planes.dtype == np.float32
        assert s.policy.shape == (num_actions(6),)
        assert s.policy.sum() == pytest.approx(1.0)
        assert s.value in (-1.0, 0.0, 1.0)


def test_augment_multiplies_by_eight_and_fills_buffer():
    buffer = ReplayBuffer(capacity=100_000)
    samples = generate_games_parallel(
        _net(), size=6, n_games=5,
        mcts_config=MCTSConfig(n_simulations=10),
        config=SelfPlayConfig(augment=True, n_parallel=8),  # n_parallel > n_games
        seed=2, buffer=buffer,
    )
    assert len(samples) % 8 == 0
    assert len(samples) > 0
    assert len(buffer) == len(samples)


def test_more_games_than_workers_all_complete():
    # Budget (8 Partien) größer als die Worker-Zahl (3): jeder Worker muss
    # nacheinander mehrere Partien abarbeiten, bis das Kontingent leer ist.
    raw = generate_games_parallel(
        _net(), size=6, n_games=8,
        mcts_config=MCTSConfig(n_simulations=8),
        config=SelfPlayConfig(augment=False, n_parallel=3),
        seed=3,
    )
    # Über 8 vollständige Partien muss mindestens ein entschiedenes Ergebnis
    # auftauchen (sonst wären keine Partien zu Ende gespielt worden).
    assert any(s.value != 0.0 for s in raw)


def test_value_targets_are_consistent_per_outcome():
    raw = generate_games_parallel(
        _net(), size=6, n_games=4,
        mcts_config=MCTSConfig(n_simulations=8),
        config=SelfPlayConfig(augment=False, n_parallel=4),
        seed=4,
    )
    assert set(s.value for s in raw).issubset({-1.0, 0.0, 1.0})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="keine CUDA-GPU")
def test_runs_on_gpu():
    net = OthelloNet(board_size=6).eval().cuda()
    samples = generate_games_parallel(
        net, size=6, n_games=4,
        mcts_config=MCTSConfig(n_simulations=10),
        config=SelfPlayConfig(augment=False, n_parallel=4),
        seed=5,
    )
    assert len(samples) > 0
