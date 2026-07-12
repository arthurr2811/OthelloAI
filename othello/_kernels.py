"""Numba-JIT-Kernel für die heißen Engine-Pfade (Hebel D aus dem README).

Profil des Trainings: Der Engpass ist Single-Core-Python – und davon entfällt
der Löwenanteil auf ``legal_moves``/``apply_move``/``has_legal_move`` (pro
MCTS-Simulation mehrfach aufgerufen). Diese drei Pfade sind hier als
``@njit``-Kernel implementiert; :mod:`othello.board` behält die reinen
Python-Implementierungen als Fallback (numba nicht installiert) und als
Referenz für den Äquivalenz-Test (``tests/test_kernels.py``).

Semantik ist exakt die der Python-Pfade – inklusive Zug-Reihenfolge
(zeilenweise) und "0 Flips = illegal".
"""

from __future__ import annotations

import numpy as np
from numba import njit

# Die 8 Richtungen als Array-Konstante (numba behandelt globale Arrays als
# eingefrorene Konstanten).
_DIRS = np.array(
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
    dtype=np.int64,
)


@njit(cache=True)
def _is_legal(board: np.ndarray, player: int, r: int, c: int) -> bool:
    """True, wenn ein Zug auf das (leere) Feld (r, c) mindestens einen Stein dreht."""
    size = board.shape[0]
    for i in range(8):
        dr, dc = _DIRS[i, 0], _DIRS[i, 1]
        rr, cc = r + dr, c + dc
        n = 0
        while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == -player:
            rr += dr
            cc += dc
            n += 1
        if n > 0 and 0 <= rr < size and 0 <= cc < size and board[rr, cc] == player:
            return True
    return False


@njit(cache=True)
def has_legal_move(board: np.ndarray, player: int) -> bool:
    size = board.shape[0]
    for r in range(size):
        for c in range(size):
            if board[r, c] == 0 and _is_legal(board, player, r, c):
                return True
    return False


@njit(cache=True)
def legal_moves_mask(board: np.ndarray, player: int) -> np.ndarray:
    """Boolesche (S, S)-Maske aller legalen Felder für ``player``."""
    size = board.shape[0]
    mask = np.zeros((size, size), dtype=np.bool_)
    for r in range(size):
        for c in range(size):
            if board[r, c] == 0 and _is_legal(board, player, r, c):
                mask[r, c] = True
    return mask


@njit(cache=True)
def apply_move(board: np.ndarray, player: int, r: int, c: int):
    """Führt den Zug aus. Rückgabe: ``(n_flips, neues Board)``.

    ``n_flips == 0`` heißt: illegaler Zug (Feld besetzt oder nichts
    eingeklammert) – das Board ist dann eine unveränderte Kopie; die
    ``ValueError``-Behandlung übernimmt der Python-Aufrufer.
    """
    size = board.shape[0]
    out = board.copy()
    total = 0
    if board[r, c] == 0:
        for i in range(8):
            dr, dc = _DIRS[i, 0], _DIRS[i, 1]
            rr, cc = r + dr, c + dc
            n = 0
            while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == -player:
                rr += dr
                cc += dc
                n += 1
            if n > 0 and 0 <= rr < size and 0 <= cc < size and board[rr, cc] == player:
                rr, cc = r + dr, c + dc
                for _ in range(n):
                    out[rr, cc] = player
                    rr += dr
                    cc += dc
                total += n
        if total > 0:
            out[r, c] = player
    return total, out
