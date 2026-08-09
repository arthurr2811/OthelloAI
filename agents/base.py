"""Einheitliches Agent-Interface.

Ein Agent bekommt den aktuellen ``GameState`` und wählt daraus einen Zug. Der
State liefert über ``legal_moves()`` bereits die gültigen Optionen (inkl. PASS,
falls erzwungen), sodass Agenten die Pass-Logik nicht selbst nachbauen müssen.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from othello.board import GameState, Move


class Agent(ABC):
    """Basisklasse für alle Bots."""

    name: str = "Agent"

    @abstractmethod
    def select_move(self, state: GameState) -> Move:
        """Wählt einen Zug aus ``state.legal_moves()``.
        """

    def __str__(self) -> str:
        return self.name
