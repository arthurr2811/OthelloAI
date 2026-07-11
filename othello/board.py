"""Board-Repräsentation und Spielzustand für Othello.

Konvention:
    +1  = Schwarz (X), zieht zuerst
    -1  = Weiß (O)
     0  = leeres Feld

Das Board ist ein 2D-NumPy-Array (int8) der Kantenlänge ``size``. Bewusst ein
Array statt Bitboard – Klarheit vor Speed. Bitboard-Optimierung erst, falls
Self-Play zum Engpass wird (siehe plan.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --- Spieler-/Feld-Konstanten ---
BLACK = 1
WHITE = -1
EMPTY = 0

# Die 8 Richtungen (dr, dc): orthogonal + diagonal.
DIRECTIONS: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)

# Anzeige-Symbole für to_string().
_SYMBOLS = {BLACK: "X", WHITE: "O", EMPTY: "."}


def initial_board(size: int = 8) -> np.ndarray:
    """Erzeugt die Othello-Startstellung: vier Steine im Zentrum.

    ``size`` muss gerade und >= 2 sein, damit das Zentrum wohldefiniert ist.
    """
    if size < 2 or size % 2 != 0:
        raise ValueError(f"board_size muss gerade und >= 2 sein, war {size}")

    board = np.zeros((size, size), dtype=np.int8)
    lo = size // 2 - 1
    hi = size // 2
    # Diagonal gleichfarbig: Weiß auf (lo,lo)/(hi,hi), Schwarz auf (lo,hi)/(hi,lo).
    board[lo, lo] = WHITE
    board[hi, hi] = WHITE
    board[lo, hi] = BLACK
    board[hi, lo] = BLACK
    return board


def opponent(player: int) -> int:
    """Gibt den Gegenspieler zurück (+1 <-> -1)."""
    return -player


def board_to_string(board: np.ndarray, current_player: int | None = None) -> str:
    """Menschenlesbare Darstellung mit Zeilen-/Spaltenkoordinaten.

    Spalten sind mit Buchstaben (a, b, c, ...), Zeilen mit Zahlen (1..N)
    beschriftet – wie in der Othello-Notation üblich.
    """
    size = board.shape[0]
    col_labels = "  " + " ".join(chr(ord("a") + c) for c in range(size))
    lines = [col_labels]
    for r in range(size):
        cells = " ".join(_SYMBOLS[int(board[r, c])] for c in range(size))
        lines.append(f"{r + 1:>2} {cells}")

    out = "\n".join(lines)
    if current_player is not None:
        who = _SYMBOLS[current_player]
        counts = disc_counts(board)
        out += f"\nAm Zug: {who}  (X={counts[BLACK]}, O={counts[WHITE]})"
    return out


def disc_counts(board: np.ndarray) -> dict[int, int]:
    """Zählt Steine je Spieler. Praktisch für Anzeige und Gewinnermittlung."""
    return {
        BLACK: int(np.count_nonzero(board == BLACK)),
        WHITE: int(np.count_nonzero(board == WHITE)),
    }


@dataclass
class GameState:
    """Vollständiger Spielzustand: Brett + wer am Zug ist.

    Der ``board`` wird beim Anlegen kopiert, damit die Startstellung nicht
    versehentlich von außen mutiert wird.
    """

    board: np.ndarray
    current_player: int = BLACK

    def __post_init__(self) -> None:
        self.board = np.array(self.board, dtype=np.int8, copy=True)

    @classmethod
    def initial(cls, size: int = 8) -> "GameState":
        """Frische Startstellung, Schwarz am Zug."""
        return cls(board=initial_board(size), current_player=BLACK)

    @property
    def size(self) -> int:
        return self.board.shape[0]

    def to_string(self) -> str:
        return board_to_string(self.board, self.current_player)

    def __str__(self) -> str:
        return self.to_string()
