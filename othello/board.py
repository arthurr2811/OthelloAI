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

# Sentinel für den Pass-Zug: ein Spieler ohne legalen Zug muss passen.
PASS = "PASS"

# Ein Zug ist entweder ein Feld (Zeile, Spalte) oder PASS.
Move = "tuple[int, int] | str"


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


def _flips_in_direction(
    board: np.ndarray, r: int, c: int, player: int, dr: int, dc: int
) -> list[tuple[int, int]]:
    """Steine, die ein Zug auf (r, c) in Richtung (dr, dc) umdrehen würde.

    Läuft vom Feld aus in eine Richtung: sammelt eine ununterbrochene Kette
    gegnerischer Steine, die von einem eigenen Stein abgeschlossen wird. Bricht
    die Kette an einem leeren Feld oder am Rand ab, wird nichts umgedreht.
    """
    size = board.shape[0]
    opp = -player
    line: list[tuple[int, int]] = []
    rr, cc = r + dr, c + dc
    while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == opp:
        line.append((rr, cc))
        rr += dr
        cc += dc
    # Nur gültig, wenn nach >=1 gegnerischem Stein ein eigener Stein folgt.
    if line and 0 <= rr < size and 0 <= cc < size and board[rr, cc] == player:
        return line
    return []


def flips_for_move(
    board: np.ndarray, player: int, move: tuple[int, int]
) -> list[tuple[int, int]]:
    """Alle Steine, die ``move`` in allen 8 Richtungen umdrehen würde.

    Leere Liste, wenn der Zug nichts umdreht (also illegal ist).
    """
    r, c = move
    if board[r, c] != EMPTY:
        return []
    flips: list[tuple[int, int]] = []
    for dr, dc in DIRECTIONS:
        flips.extend(_flips_in_direction(board, r, c, player, dr, dc))
    return flips


def is_legal_move(board: np.ndarray, player: int, move: tuple[int, int]) -> bool:
    """True, wenn ``move`` für ``player`` mindestens einen Stein umdreht."""
    r, c = move
    if not (0 <= r < board.shape[0] and 0 <= c < board.shape[0]):
        return False
    return bool(flips_for_move(board, player, move))


def legal_moves(board: np.ndarray, player: int) -> list[tuple[int, int]]:
    """Alle legalen Felder für ``player`` (ohne Pass).

    Ein Feld ist legal, wenn es leer ist und in mindestens einer Richtung
    gegnerische Steine einklammert.
    """
    size = board.shape[0]
    moves: list[tuple[int, int]] = []
    for r in range(size):
        for c in range(size):
            if board[r, c] == EMPTY and flips_for_move(board, player, (r, c)):
                moves.append((r, c))
    return moves


def has_legal_move(board: np.ndarray, player: int) -> bool:
    """True, wenn ``player`` irgendein legales Feld hat (schneller Abbruch)."""
    size = board.shape[0]
    for r in range(size):
        for c in range(size):
            if board[r, c] == EMPTY and flips_for_move(board, player, (r, c)):
                return True
    return False


def apply_move(
    board: np.ndarray, player: int, move: tuple[int, int]
) -> np.ndarray:
    """Wendet ``move`` an und gibt ein **neues** Board zurück (nicht in-place).

    Setzt den Stein auf das Feld und dreht alle eingeklammerten gegnerischen
    Steine um. Wirft ``ValueError`` bei einem illegalen Zug.
    """
    flips = flips_for_move(board, player, move)
    if not flips:
        raise ValueError(f"Illegaler Zug {move} für Spieler {player}")
    new_board = board.copy()
    r, c = move
    new_board[r, c] = player
    for fr, fc in flips:
        new_board[fr, fc] = player
    return new_board


def game_over(board: np.ndarray) -> bool:
    """Spiel vorbei, wenn **kein** Spieler mehr ziehen kann."""
    return not has_legal_move(board, BLACK) and not has_legal_move(board, WHITE)


def winner(board: np.ndarray) -> int:
    """Gewinner nach Steinmehrheit: BLACK, WHITE oder EMPTY (Unentschieden).

    Nur bei Spielende sinnvoll interpretierbar; die Funktion selbst zählt
    lediglich die Steine.
    """
    counts = disc_counts(board)
    if counts[BLACK] > counts[WHITE]:
        return BLACK
    if counts[WHITE] > counts[BLACK]:
        return WHITE
    return EMPTY


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

    def legal_moves(self):
        """Optionen für den aktuellen Spieler.

        - Reale Felder, wenn welche existieren.
        - ``[PASS]``, wenn der Spieler nicht ziehen kann, aber der Gegner schon
          (erzwungenes Passen).
        - ``[]``, wenn das Spiel vorbei ist (keiner kann ziehen).
        """
        moves = legal_moves(self.board, self.current_player)
        if moves:
            return moves
        if has_legal_move(self.board, opponent(self.current_player)):
            return [PASS]
        return []

    def is_terminal(self) -> bool:
        return game_over(self.board)

    def apply(self, move) -> "GameState":
        """Führt ``move`` aus und gibt einen **neuen** Zustand zurück.

        Bei PASS wechselt nur der Spieler (nur erlaubt, wenn keine realen Züge
        existieren). Ansonsten wird das Board aktualisiert und gewechselt.
        """
        if move == PASS:
            if legal_moves(self.board, self.current_player):
                raise ValueError("PASS unzulässig: es gibt legale Züge")
            return GameState(board=self.board, current_player=opponent(self.current_player))
        new_board = apply_move(self.board, self.current_player, move)
        return GameState(board=new_board, current_player=opponent(self.current_player))

    def winner(self) -> int:
        return winner(self.board)

    def to_string(self) -> str:
        return board_to_string(self.board, self.current_player)

    def __str__(self) -> str:
        return self.to_string()
