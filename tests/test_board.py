"""Tests für Schritt 1.1 – Board & Spielzustand."""

import numpy as np
import pytest

from othello.board import (
    BLACK,
    WHITE,
    EMPTY,
    GameState,
    disc_counts,
    initial_board,
    opponent,
)


def test_initial_board_has_four_center_discs():
    board = initial_board(8)
    assert np.count_nonzero(board) == 4
    counts = disc_counts(board)
    assert counts[BLACK] == 2
    assert counts[WHITE] == 2


def test_initial_center_layout_8x8():
    board = initial_board(8)
    # Diagonal gleichfarbig, Standard-Othello-Startstellung.
    assert board[3, 3] == WHITE
    assert board[4, 4] == WHITE
    assert board[3, 4] == BLACK
    assert board[4, 3] == BLACK
    # Alles außerhalb des Zentrums ist leer.
    board[3:5, 3:5] = EMPTY
    assert np.count_nonzero(board) == 0


def test_initial_layout_scales_to_6x6():
    board = initial_board(6)
    assert np.count_nonzero(board) == 4
    assert board[2, 2] == WHITE
    assert board[3, 3] == WHITE
    assert board[2, 3] == BLACK
    assert board[3, 2] == BLACK


@pytest.mark.parametrize("bad_size", [3, 5, 1, 0, -2])
def test_initial_board_rejects_invalid_size(bad_size):
    with pytest.raises(ValueError):
        initial_board(bad_size)


def test_opponent():
    assert opponent(BLACK) == WHITE
    assert opponent(WHITE) == BLACK


def test_gamestate_initial_defaults_to_black():
    state = GameState.initial(8)
    assert state.current_player == BLACK
    assert state.size == 8


def test_gamestate_copies_board_on_init():
    board = initial_board(8)
    state = GameState(board=board, current_player=BLACK)
    board[0, 0] = BLACK  # externe Mutation darf den State nicht beeinflussen
    assert state.board[0, 0] == EMPTY


def test_to_string_is_readable():
    text = GameState.initial(8).to_string()
    # Kopfzeile mit Spaltenbuchstaben und Steinsymbolen vorhanden.
    assert "a b c d e f g h" in text
    assert "X" in text and "O" in text
    assert "Am Zug: X" in text
