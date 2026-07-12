"""Äquivalenz-Tests: Numba-Kernel vs. reine Python-Engine (Sicherheitsnetz für Hebel D).

Die JIT-Kernel (othello/_kernels.py) müssen sich auf *jeder* erreichbaren
Stellung exakt wie die Python-Referenz verhalten – Zufallspartien decken
Randfälle (Pass, volle Kanten, Endspiel) automatisch mit ab.
"""

from __future__ import annotations

import numpy as np
import pytest

from othello import board as b

pytestmark = pytest.mark.skipif(
    b._kernels is None, reason="numba nicht installiert – Kernel-Pfad inaktiv"
)


@pytest.mark.parametrize("size", [6, 8])
def test_kernels_match_python_reference_over_random_games(size):
    rng = np.random.default_rng(42)
    for _ in range(15):
        state = b.GameState.initial(size)
        while not state.is_terminal():
            for player in (b.BLACK, b.WHITE):
                assert b.legal_moves(state.board, player) == \
                    b._legal_moves_py(state.board, player)
                assert b.has_legal_move(state.board, player) == \
                    b._has_legal_move_py(state.board, player)

            moves = state.legal_moves()
            move = moves[int(rng.integers(len(moves)))]
            if move != b.PASS:
                assert np.array_equal(
                    b.apply_move(state.board, state.current_player, move),
                    b._apply_move_py(state.board, state.current_player, move),
                )
            state = state.apply(move)


def test_kernel_apply_move_rejects_illegal_moves():
    board = b.initial_board(8)
    with pytest.raises(ValueError):
        b.apply_move(board, b.BLACK, (3, 3))   # Feld besetzt
    with pytest.raises(ValueError):
        b.apply_move(board, b.BLACK, (0, 0))   # dreht nichts um


def test_kernel_apply_move_does_not_mutate_input():
    board = b.initial_board(8)
    before = board.copy()
    b.apply_move(board, b.BLACK, (2, 3))
    assert np.array_equal(board, before)
