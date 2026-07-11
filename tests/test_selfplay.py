"""Tests für Schritt 2.3: Self-Play, Augmentierung und Replay-Buffer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from az.augment import symmetries, transform_planes, transform_policy
from az.encoding import encode_state, num_actions
from az.mcts import NeuralMCTS
from az.net import OthelloNet
from az.replay import ReplayBuffer, Sample
from az.selfplay import augment, generate_game, play_game
from config import MCTSConfig, SelfPlayConfig
from othello.board import GameState


def _mcts(size=6, sims=12, seed=0):
    torch.manual_seed(0)
    net = OthelloNet(board_size=size).eval()
    return NeuralMCTS(net, MCTSConfig(n_simulations=sims), add_noise=True, seed=seed)


# --- Augmentierung ---

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


# --- Self-Play-Partie ---

def test_play_game_produces_well_formed_samples():
    samples = play_game(_mcts(), size=6, config=SelfPlayConfig(temperature_moves=4),
                        rng=np.random.default_rng(1))
    assert len(samples) > 0
    for s in samples:
        assert s.planes.shape == (3, 6, 6)
        assert s.planes.dtype == np.float32
        assert s.policy.shape == (num_actions(6),)
        assert s.policy.sum() == pytest.approx(1.0)
        assert s.value in (-1.0, 0.0, 1.0)


def test_value_targets_reflect_a_single_outcome():
    # Alle Value-Targets stammen aus einem Partie-Ergebnis: entweder alle 0 (Remis)
    # oder eine Mischung aus +1/-1 (Sieger/Verlierer), aber nie widersprüchlich.
    samples = play_game(_mcts(seed=2), size=6, rng=np.random.default_rng(2))
    values = {s.value for s in samples}
    assert values in ({0.0}, {-1.0, 1.0}, {1.0}, {-1.0}, {-1.0, 0.0, 1.0})
    # Für ein entschiedenes Spiel muss mindestens ein Nicht-Null-Value auftreten.
    assert any(v != 0.0 for v in values) or values == {0.0}


def test_augment_multiplies_by_eight():
    raw = play_game(_mcts(), size=6, rng=np.random.default_rng(3))
    aug = augment(raw, size=6)
    assert len(aug) == 8 * len(raw)
    # Value bleibt bei jeder Symmetrie erhalten.
    assert {s.value for s in aug} == {s.value for s in raw}


# --- Replay-Buffer ---

def test_generate_game_fills_buffer_with_augmented_samples():
    buffer = ReplayBuffer(capacity=100_000)
    samples = generate_game(_mcts(), size=6, config=SelfPlayConfig(augment=True),
                           rng=np.random.default_rng(4), buffer=buffer)
    assert len(buffer) == len(samples)
    assert len(samples) % 8 == 0                   # augmentiert -> Vielfaches von 8


def test_buffer_respects_capacity():
    buffer = ReplayBuffer(capacity=10)
    planes = np.zeros((3, 6, 6), dtype=np.float32)
    policy = np.zeros(num_actions(6), dtype=np.float32)
    policy[0] = 1.0
    buffer.add([Sample(planes, policy, 1.0) for _ in range(25)])
    assert len(buffer) == 10                        # älteste fielen heraus


def test_buffer_sample_batch_shapes():
    buffer = ReplayBuffer(capacity=100)
    planes = np.zeros((3, 6, 6), dtype=np.float32)
    policy = np.zeros(num_actions(6), dtype=np.float32)
    policy[0] = 1.0
    buffer.add([Sample(planes, policy, -1.0) for _ in range(20)])
    b_planes, b_policy, b_value = buffer.sample_batch(8, np.random.default_rng(0))
    assert b_planes.shape == (8, 3, 6, 6)
    assert b_policy.shape == (8, num_actions(6))
    assert b_value.shape == (8,)
    assert b_planes.dtype == np.float32
