"""Tests für den Replay-Buffer (az/replay.py)."""

from __future__ import annotations

import numpy as np

from az.encoding import num_actions
from az.replay import ReplayBuffer, Sample


def _sample(size=6, value=1.0):
    planes = np.zeros((3, size, size), dtype=np.float32)
    policy = np.zeros(num_actions(size), dtype=np.float32)
    policy[0] = 1.0
    return Sample(planes, policy, value)


def test_buffer_respects_capacity():
    buffer = ReplayBuffer(capacity=10)
    buffer.add([_sample() for _ in range(25)])
    assert len(buffer) == 10                        # älteste fielen heraus


def test_buffer_sample_batch_shapes():
    buffer = ReplayBuffer(capacity=100)
    buffer.add([_sample(value=-1.0) for _ in range(20)])
    b_planes, b_policy, b_value = buffer.sample_batch(8, np.random.default_rng(0))
    assert b_planes.shape == (8, 3, 6, 6)
    assert b_policy.shape == (8, num_actions(6))
    assert b_value.shape == (8,)
    assert b_planes.dtype == np.float32
