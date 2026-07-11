"""Umwandlung zwischen Spielzustand und Netz-Tensoren.

Zwei Aufgaben, die Netz, MCTS und Self-Play teilen:

1. **State -> Eingabe-Ebenen.** Das Netz sieht den Zustand immer aus Sicht des
   Spielers am Zug (kanonische Perspektive), damit es nur *eine* Bewertung lernen
   muss und nicht getrennt für Schwarz und Weiß.
2. **Zug <-> Index.** Die Policy ist ein flacher Vektor der Länge ``size*size + 1``.
   Feld ``(r, c)`` liegt an Index ``r*size + c``, der Pass-Zug am letzten Index.
"""

from __future__ import annotations

import numpy as np

from othello.board import PASS, GameState, Move


def num_actions(size: int) -> int:
    """Größe des Policy-Vektors: jedes Feld plus ein Pass-Zug."""
    return size * size + 1


def move_to_index(move: Move, size: int) -> int:
    """Wandelt einen Zug in seinen Policy-Index um (Pass = letzter Index)."""
    if move == PASS:
        return size * size
    r, c = move
    return r * size + c


def index_to_move(index: int, size: int) -> Move:
    """Kehrt ``move_to_index`` um: Index -> Feld ``(r, c)`` oder ``PASS``."""
    if index == size * size:
        return PASS
    return divmod(index, size)


def encode_state(state: GameState) -> np.ndarray:
    """Kodiert ``state`` als ``(3, size, size)``-float32-Array.

    Ebenen, alle aus Sicht des Spielers am Zug:
        0 – eigene Steine (1.0 wo der ziehende Spieler liegt)
        1 – gegnerische Steine
        2 – konstante Ebene: 1.0 wenn Schwarz am Zug, sonst 0.0

    Ebene 2 erhält die absolute Farbe (Othello ist nicht farbsymmetrisch: Schwarz
    beginnt), obwohl die Steine-Ebenen bereits relativ kodiert sind.
    """
    board = state.board
    player = state.current_player
    own = (board == player).astype(np.float32)
    opp = (board == -player).astype(np.float32)
    color = np.full(board.shape, 1.0 if player == 1 else 0.0, dtype=np.float32)
    return np.stack([own, opp, color], axis=0)


def legal_move_mask(state: GameState) -> np.ndarray:
    """Boolesche Maske der Länge ``num_actions`` über die legalen Züge.

    Wird auf die Policy-Logits gelegt, damit das Netz keine illegalen Züge wählt.
    """
    size = state.size
    mask = np.zeros(num_actions(size), dtype=bool)
    for move in state.legal_moves():
        mask[move_to_index(move, size)] = True
    return mask
