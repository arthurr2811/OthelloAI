"""Zentrale Konfiguration für Engine, Self-Play und Training.

Bewusst schlank gehalten – wird in Phase 2 (AlphaZero-Pipeline) ausgebaut.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    """Spiel-Parameter. Brettgröße hier zentral, damit 6x6-Durchstich und
    8x8-Training über dieselbe Config laufen."""

    board_size: int = 8


DEFAULT_GAME = GameConfig()
