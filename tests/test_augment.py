"""Tests für die Dihedral-Augmentierung (az/augment.py)."""

from __future__ import annotations

import numpy as np
import pytest

from az.augment import augment, symmetries, transform_planes, transform_policy
from az.encoding import encode_state, num_actions
from az.replay import Sample
from othello.board import GameState


def test_symmetries_returns_eight_valid_variants():
    state = GameState.initial(size=8)
    planes = encode_state(state)
    policy = np.zeros(num_actions(8), dtype=np.float32)
    policy[0] = 0.7          # Feld (0,0)
    policy[-1] = 0.3         # Pass
    variants = symmetries(planes, policy, 8)
    assert len(variants) == 8
    for p, pi in variants:
        assert p.shape == (3, 8, 8)
        assert pi.shape == (num_actions(8),)
        assert pi.sum() == pytest.approx(1.0)      # Normierung erhalten
        assert pi[-1] == pytest.approx(0.3)        # Pass bleibt an Ort und Stelle


def test_augmentation_matches_engine_equivariance():
    # Kern-Korrektheit: die transformierten Ebenen einer Stellung müssen der
    # Kodierung des geometrisch transformierten Boards entsprechen.
    state = GameState.initial(size=8)
    for m in state.legal_moves()[:1]:
        state = state.apply(m)                     # eine asymmetrischere Stellung
    planes = encode_state(state)
    for k in range(4):
        for flip in (False, True):
            transformed = transform_planes(planes, k, flip)
            board = np.rot90(state.board, k)
            if flip:
                board = board[:, ::-1]
            ref = encode_state(GameState(board=board, current_player=state.current_player))
            assert np.array_equal(transformed, ref)


def test_transform_policy_moves_corner_consistently():
    size = 8
    policy = np.zeros(num_actions(size), dtype=np.float32)
    policy[0] = 1.0                                # Ecke (0,0)
    # 90-Grad-Rotation gegen den Uhrzeigersinn: (0,0) -> (size-1, 0).
    rotated = transform_policy(policy, size, k=1, flip=False)
    grid = rotated[: size * size].reshape(size, size)
    assert grid[size - 1, 0] == pytest.approx(1.0)
    assert grid.sum() == pytest.approx(1.0)


def test_augment_multiplies_by_eight_and_keeps_value():
    rng = np.random.default_rng(0)
    raw = []
    for _ in range(3):
        planes = rng.standard_normal((3, 8, 8)).astype(np.float32)
        policy = np.zeros(num_actions(8), dtype=np.float32)
        policy[int(rng.integers(64))] = 1.0
        raw.append(Sample(planes, policy, float(rng.choice([-1.0, 1.0]))))
    aug = augment(raw, size=8)
    assert len(aug) == 8 * len(raw)
    assert {s.value for s in aug} == {s.value for s in raw}
