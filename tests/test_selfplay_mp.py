"""Tests für das Multiprocessing-Self-Play (Hebel C, az/selfplay_mp.py).

Winzige Budgets und CPU-Device: geprüft wird die Verteil-Mechanik (Spawn,
Gewichte-Transfer, Sample-Rückgabe), nicht die Spielstärke.
"""

from __future__ import annotations

import pytest

from az.net import OthelloNet
from az.replay import ReplayBuffer
from az.selfplay_mp import SelfPlayPool
from config import MCTSConfig, NetConfig, SelfPlayConfig


def test_selfplay_pool_generates_samples_across_workers():
    net = OthelloNet(6, NetConfig(channels=8, n_res_blocks=1, value_hidden=8))
    buffer = ReplayBuffer(10_000)
    pool = SelfPlayPool(2)
    try:
        samples = pool.generate(
            net, 6, 3,   # 3 Partien auf 2 Worker -> Chunks 2 + 1
            mcts_config=MCTSConfig(n_simulations=4),
            config=SelfPlayConfig(temperature_moves=4, n_parallel=4),
            device="cpu",
            seed=0,
            buffer=buffer,
        )
    finally:
        pool.close()

    assert len(samples) > 0
    assert len(buffer) == len(samples)
    assert len(samples) % 8 == 0            # 8 Dihedral-Augmentierungen je Stellung
    sample = samples[0]
    assert sample.planes.shape == (3, 6, 6)
    assert sample.policy.shape == (37,)     # 36 Felder + Pass
    assert -1.0 <= sample.value <= 1.0


def test_selfplay_pool_rejects_single_worker():
    with pytest.raises(ValueError):
        SelfPlayPool(1)
