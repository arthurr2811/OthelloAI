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


@dataclass(frozen=True)
class MCTSConfig:
    """Parameter für das PUCT-MCTS mit Netz (Schritt 2.2).

    Die Dirichlet-/Temperatur-Werte sind Self-Play-Exploration; beim reinen
    Spielen/Evaluieren wird ohne Rausch und mit Temperatur ~0 gesucht.
    """

    n_simulations: int = 200
    c_puct: float = 1.5            # Explorations-Gewicht in der PUCT-Formel
    dirichlet_alpha: float = 0.3   # Konzentration des Wurzel-Rauschens
    dirichlet_epsilon: float = 0.25  # Mischungsanteil des Rauschens
    temperature: float = 1.0       # >0: proportional zu Visit-Counts, 0: argmax


@dataclass(frozen=True)
class SelfPlayConfig:
    """Parameter für den Self-Play-Loop (Schritt 2.3)."""

    # So viele Anfangszüge werden mit Temperatur 1 (explorativ) gesampelt, danach
    # deterministisch (Temperatur 0). Sorgt für Eröffnungsvielfalt in den Daten.
    temperature_moves: int = 15
    augment: bool = True           # 8 Dihedral-Symmetrien pro Sample
    buffer_size: int = 100_000     # Max-Größe des Replay-Buffers (Samples)
    max_moves: int = 1000          # Endlosschutz pro Partie


@dataclass(frozen=True)
class TrainConfig:
    """Parameter der Trainingsschleife (Schritt 2.4)."""

    lr: float = 1e-3
    weight_decay: float = 1e-4       # L2-Regularisierung über den Optimizer
    batch_size: int = 64
    value_loss_weight: float = 1.0   # Gewicht des Value-MSE relativ zur Policy-CE


@dataclass(frozen=True)
class EvalConfig:
    """Parameter für Evaluation & Gating (Schritt 2.5)."""

    n_games: int = 40
    win_threshold: float = 0.55      # ab dieser Punktquote wird der Kandidat neues Bestmodell
    # Explorative Eröffnung (Temperatur 1 für so viele Züge), damit die Partien
    # variieren – sonst spielen zwei deterministische Netze immer dieselbe Partie.
    temperature_moves: int = 10


DEFAULT_GAME = GameConfig()
DEFAULT_NET = NetConfig()
DEFAULT_MCTS = MCTSConfig()
DEFAULT_SELFPLAY = SelfPlayConfig()
DEFAULT_TRAIN = TrainConfig()
DEFAULT_EVAL = EvalConfig()
