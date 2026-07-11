"""Tests für Schritt 2.1: Netz-Architektur und State-Kodierung."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from az.encoding import (
    encode_state,
    index_to_move,
    legal_move_mask,
    move_to_index,
    num_actions,
)
from az.net import OthelloNet
from othello.board import PASS, GameState


# --- Kodierung: Zug <-> Index ---

@pytest.mark.parametrize("size", [6, 8])
def test_move_index_roundtrip(size):
    for r in range(size):
        for c in range(size):
            idx = move_to_index((r, c), size)
            assert 0 <= idx < size * size
            assert index_to_move(idx, size) == (r, c)


@pytest.mark.parametrize("size", [6, 8])
def test_pass_is_last_index(size):
    idx = move_to_index(PASS, size)
    assert idx == num_actions(size) - 1
    assert index_to_move(idx, size) == PASS


# --- Kodierung: State -> Ebenen ---

def test_encode_state_shape_and_planes():
    state = GameState.initial(size=8)  # Schwarz am Zug
    planes = encode_state(state)
    assert planes.shape == (3, 8, 8)
    assert planes.dtype == np.float32
    # Startstellung: 2 eigene, 2 gegnerische Steine.
    assert planes[0].sum() == 2
    assert planes[1].sum() == 2
    # Ebene 2 konstant 1.0, weil Schwarz beginnt.
    assert np.all(planes[2] == 1.0)


def test_encode_state_is_from_movers_perspective():
    state = GameState.initial(size=8)
    after = state.apply(state.legal_moves()[0])  # jetzt Weiß am Zug
    planes = encode_state(after)
    # Ebene 0 (eigene) zählt nun die Weiß-Steine, Ebene 2 ist 0.0.
    assert np.all(planes[2] == 0.0)
    assert planes[0].sum() == (after.board == after.current_player).sum()


def test_legal_move_mask_matches_legal_moves():
    state = GameState.initial(size=8)
    mask = legal_move_mask(state)
    assert mask.shape == (num_actions(8),)
    expected = {move_to_index(m, 8) for m in state.legal_moves()}
    assert set(np.flatnonzero(mask)) == expected


# --- Netz: Forward-Pass ---

@pytest.mark.parametrize("size", [6, 8])
def test_forward_shapes(size):
    net = OthelloNet(board_size=size).eval()
    batch = torch.zeros(4, 3, size, size)
    policy, value = net(batch)
    assert policy.shape == (4, size * size + 1)
    assert value.shape == (4,)


def test_value_in_tanh_range_and_no_nan():
    net = OthelloNet(board_size=8).eval()
    batch = torch.randn(16, 3, 8, 8)
    policy, value = net(batch)
    assert torch.isfinite(policy).all()
    assert torch.isfinite(value).all()
    assert value.min() >= -1.0 and value.max() <= 1.0


def test_forward_on_real_state():
    net = OthelloNet(board_size=8).eval()
    state = GameState.initial(size=8)
    x = torch.from_numpy(encode_state(state)).unsqueeze(0)
    policy, value = net(x)
    assert policy.shape == (1, 65)
    assert value.shape == (1,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="keine CUDA-GPU")
def test_forward_on_gpu():
    net = OthelloNet(board_size=8).eval().cuda()
    batch = torch.randn(8, 3, 8, 8, device="cuda")
    policy, value = net(batch)
    assert policy.is_cuda and value.is_cuda
    assert torch.isfinite(policy).all() and torch.isfinite(value).all()
    assert value.shape == (8,)
