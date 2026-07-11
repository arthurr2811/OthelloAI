"""Tests für Schritt 1.2 (Zuggenerierung & Flips) und 1.3 (Pass, Game-Over)."""

import random

import numpy as np
import pytest

from othello.board import (
    BLACK,
    WHITE,
    EMPTY,
    PASS,
    GameState,
    apply_move,
    disc_counts,
    flips_for_move,
    game_over,
    has_legal_move,
    initial_board,
    is_legal_move,
    legal_moves,
    winner,
)


# --- 1.2: Zuggenerierung -----------------------------------------------------

def test_opening_moves_black_8x8():
    board = initial_board(8)
    moves = set(legal_moves(board, BLACK))
    assert moves == {(2, 3), (3, 2), (4, 5), (5, 4)}


def test_opening_moves_white_8x8():
    # Symmetrisch: Weiß hat zu Beginn ebenfalls vier Züge.
    board = initial_board(8)
    moves = set(legal_moves(board, WHITE))
    assert moves == {(2, 4), (4, 2), (3, 5), (5, 3)}


def test_flip_in_each_of_eight_directions():
    # Zugfeld in der Mitte; je Richtung ein gegnerischer Stein + eigener Abschluss.
    from othello.board import DIRECTIONS

    for dr, dc in DIRECTIONS:
        board = np.zeros((8, 8), dtype=np.int8)
        board[4 + dr, 4 + dc] = WHITE      # gegnerisch
        board[4 + 2 * dr, 4 + 2 * dc] = BLACK  # eigener Abschluss
        flips = flips_for_move(board, BLACK, (4, 4))
        assert (4 + dr, 4 + dc) in flips
        assert len(flips) == 1


def test_move_flips_multiple_directions_at_once():
    # Schwarz spielt in ein Feld, das in mehreren Richtungen einklammert.
    board = np.zeros((8, 8), dtype=np.int8)
    # rechts: W dann B
    board[4, 5] = WHITE
    board[4, 6] = BLACK
    # unten: W dann B
    board[5, 4] = WHITE
    board[6, 4] = BLACK
    flips = set(flips_for_move(board, BLACK, (4, 4)))
    assert flips == {(4, 5), (5, 4)}


def test_no_flip_without_own_terminator():
    board = np.zeros((8, 8), dtype=np.int8)
    board[4, 5] = WHITE  # gegnerisch, aber kein eigener Abschluss dahinter
    assert flips_for_move(board, BLACK, (4, 4)) == []
    assert not is_legal_move(board, BLACK, (4, 4))


def test_no_flip_over_a_gap():
    board = np.zeros((8, 8), dtype=np.int8)
    board[4, 5] = WHITE
    board[4, 6] = EMPTY  # Lücke bricht die Kette
    board[4, 7] = BLACK
    assert flips_for_move(board, BLACK, (4, 4)) == []


def test_apply_move_flips_and_is_immutable():
    board = initial_board(8)
    original = board.copy()
    new_board = apply_move(board, BLACK, (2, 3))
    # Original unverändert.
    assert np.array_equal(board, original)
    # Neuer Stein gesetzt, eingeklammerter Stein umgedreht.
    assert new_board[2, 3] == BLACK
    assert new_board[3, 3] == BLACK  # war Weiß
    counts = disc_counts(new_board)
    assert counts[BLACK] == 4 and counts[WHITE] == 1


def test_apply_illegal_move_raises():
    board = initial_board(8)
    with pytest.raises(ValueError):
        apply_move(board, BLACK, (0, 0))  # dreht nichts um


def test_cannot_play_on_occupied_square():
    board = initial_board(8)
    assert not is_legal_move(board, BLACK, (3, 3))  # bereits belegt


# --- 1.3: Pass & Game-Over ---------------------------------------------------

def _forced_pass_board() -> np.ndarray:
    """Board, auf dem Schwarz nicht ziehen kann, Weiß aber schon."""
    board = np.array(
        [
            [EMPTY, BLACK, WHITE, WHITE],
            [BLACK, BLACK, WHITE, WHITE],
            [WHITE, WHITE, WHITE, WHITE],
            [WHITE, WHITE, WHITE, WHITE],
        ],
        dtype=np.int8,
    )
    return board


def test_forced_pass_detection():
    board = _forced_pass_board()
    assert not has_legal_move(board, BLACK)
    assert has_legal_move(board, WHITE)
    assert not game_over(board)


def test_gamestate_offers_pass_when_no_move():
    state = GameState(board=_forced_pass_board(), current_player=BLACK)
    assert state.legal_moves() == [PASS]
    # Passen wechselt nur den Spieler.
    after = state.apply(PASS)
    assert after.current_player == WHITE
    assert np.array_equal(after.board, state.board)


def test_pass_illegal_when_moves_exist():
    state = GameState.initial(8)
    with pytest.raises(ValueError):
        state.apply(PASS)


def test_game_over_on_full_board():
    board = np.full((4, 4), BLACK, dtype=np.int8)
    board[0, 0] = WHITE
    assert game_over(board)
    assert winner(board) == BLACK


def test_game_over_by_mutual_no_move_not_full():
    # Ein leeres Feld, aber weder Schwarz noch Weiß kann es bespielen.
    board = np.full((4, 4), BLACK, dtype=np.int8)
    board[0, 0] = EMPTY
    assert not has_legal_move(board, BLACK)
    assert not has_legal_move(board, WHITE)
    assert game_over(board)
    assert winner(board) == BLACK
    assert GameState(board=board, current_player=BLACK).legal_moves() == []


def test_winner_draw():
    board = np.zeros((4, 4), dtype=np.int8)
    board[:2, :] = BLACK
    board[2:, :] = WHITE
    assert disc_counts(board) == {BLACK: 8, WHITE: 8}
    assert winner(board) == EMPTY


# --- Property-Test: komplettes Random-Spiel ----------------------------------

@pytest.mark.parametrize("size", [6, 8])
def test_random_game_runs_to_completion(size):
    rng = random.Random(1234 + size)
    state = GameState.initial(size)
    steps = 0
    max_steps = size * size + 5  # weit mehr als je nötig
    while not state.is_terminal():
        options = state.legal_moves()
        assert options, "nicht-terminaler Zustand ohne Optionen"
        move = rng.choice(options)
        prev_filled = int(np.count_nonzero(state.board))
        state = state.apply(move)
        filled = int(np.count_nonzero(state.board))
        if move == PASS:
            assert filled == prev_filled  # Pass ändert keine Steine
        else:
            assert filled == prev_filled + 1  # genau ein neuer Stein
        # Board enthält immer nur gültige Werte.
        assert set(np.unique(state.board)).issubset({BLACK, WHITE, EMPTY})
        steps += 1
        assert steps <= max_steps

    # Am Ende stimmen die Steinzahlen mit dem Gewinner überein.
    counts = disc_counts(state.board)
    w = winner(state.board)
    if w == BLACK:
        assert counts[BLACK] > counts[WHITE]
    elif w == WHITE:
        assert counts[WHITE] > counts[BLACK]
    else:
        assert counts[BLACK] == counts[WHITE]
