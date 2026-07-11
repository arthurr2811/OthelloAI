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


@dataclass(frozen=True)
class NetConfig:
    """Architektur des AlphaZero-Netzes.

    Bewusst klein gehalten – reicht für 6x6 und lässt sich für 8x8 über mehr
    Blöcke/Kanäle hochskalieren, ohne dass sich der Code ändert.
    """

    # Anzahl Eingabe-Ebenen: eigene Steine / gegnerische Steine / Spieler-am-Zug.
    input_planes: int = 3
    channels: int = 64          # Feature-Kanäle im Torso
    n_res_blocks: int = 4       # Anzahl Residual-Blöcke
    value_hidden: int = 64      # Breite der versteckten Value-Schicht


DEFAULT_GAME = GameConfig()
DEFAULT_NET = NetConfig()
