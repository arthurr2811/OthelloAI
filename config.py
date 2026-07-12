"""Zentrale Konfiguration der AlphaZero-Trainingspipeline (8x8-Othello).

Alle Defaults sind die *echten* Trainingswerte des 8x8-Laufs – ein Aufruf von
``python scripts/train.py`` ohne Flags startet genau diese Konfiguration.
Kleinere Werte (Tests, Smoke-Check) werden explizit übergeben.

Gemessen auf Ryzen 7 7800X3D + RTX 5070 Ti: ~114 s pro Iteration,
120 Iterationen ≈ 3,8 h.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NetConfig:
    """Architektur des AlphaZero-Netzes (ResNet mit Policy- und Value-Kopf)."""

    # Anzahl Eingabe-Ebenen: eigene Steine / gegnerische Steine / Spieler-am-Zug.
    input_planes: int = 3
    channels: int = 128         # Feature-Kanäle im Torso
    n_res_blocks: int = 8       # Anzahl Residual-Blöcke
    value_hidden: int = 128     # Breite der versteckten Value-Schicht


@dataclass(frozen=True)
class MCTSConfig:
    """Parameter für das PUCT-MCTS mit Netz.

    Die Dirichlet-/Temperatur-Werte sind Self-Play-Exploration; beim reinen
    Spielen/Evaluieren wird ohne Rauschen und mit Temperatur ~0 gesucht.
    """

    n_simulations: int = 128
    c_puct: float = 1.5            # Explorations-Gewicht in der PUCT-Formel
    dirichlet_alpha: float = 0.3   # Konzentration des Wurzel-Rauschens
    dirichlet_epsilon: float = 0.25  # Mischungsanteil des Rauschens
    temperature: float = 1.0       # >0: proportional zu Visit-Counts, 0: argmax


@dataclass(frozen=True)
class SelfPlayConfig:
    """Parameter für den Self-Play-Loop."""

    # So viele Anfangszüge werden mit Temperatur 1 (explorativ) gesampelt, danach
    # deterministisch. Sorgt für Eröffnungsvielfalt in den Trainingsdaten.
    temperature_moves: int = 20
    augment: bool = True           # 8 Dihedral-Symmetrien pro Sample
    buffer_size: int = 400_000     # Replay-Buffer (~9 Iterationen Self-Play-Daten)
    max_moves: int = 1000          # Endlosschutz pro Partie
    # Partien, die im gebündelten Self-Play gleichzeitig laufen: ihre
    # Blatt-Bewertungen werden pro Runde zu *einem* Netz-Forward gebündelt.
    n_parallel: int = 96
    # Self-Play-Prozesse (1 = kein Multiprocessing). Ab 2 teilen sich
    # Worker-Prozesse die Partien einer Iteration (az/selfplay_mp.py).
    # Sinnvoll: Kernzahl minus ~2 (Rest für Hauptprozess + OS).
    n_workers: int = 6


@dataclass(frozen=True)
class TrainConfig:
    """Parameter der Trainingsschleife."""

    lr: float = 1e-3
    weight_decay: float = 1e-4       # L2-Regularisierung über den Optimizer
    batch_size: int = 256
    value_loss_weight: float = 1.0   # Gewicht des Value-MSE relativ zur Policy-CE


@dataclass(frozen=True)
class EvalConfig:
    """Parameter für Evaluation & Gating."""

    n_games: int = 24
    win_threshold: float = 0.55      # ab dieser Punktquote wird der Kandidat neues Bestmodell
    # Explorative Eröffnung (Temperatur 1 für so viele Züge), damit die Partien
    # variieren – sonst spielen zwei deterministische Netze immer dieselbe Partie.
    temperature_moves: int = 10


@dataclass(frozen=True)
class RunConfig:
    """Config für den ganzen AlphaZero-Loop: Self-Play -> Train -> Gate.

    Bündelt Brettgröße, Loop-Umfang und alle Sub-Configs an *einer* Stelle,
    damit ein Lauf über ein einziges Objekt reproduzierbar beschrieben ist.
    """

    board_size: int = 8
    n_iterations: int = 120
    games_per_iteration: int = 96         # Self-Play-Partien pro Iteration
    train_steps_per_iteration: int = 500  # ~3 Sichtungen je neuem Sample bei Batch 256
    baseline_games: int = 10              # Partien gegen Greedy für die Stärke-Kurve
    seed: int = 0

    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Sub-Configs (unveränderlich, daher als Default-Instanzen teilbar).
    net: NetConfig = NetConfig()
    mcts: MCTSConfig = MCTSConfig()
    selfplay: SelfPlayConfig = SelfPlayConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()


DEFAULT_NET = NetConfig()
DEFAULT_MCTS = MCTSConfig()
DEFAULT_SELFPLAY = SelfPlayConfig()
DEFAULT_TRAIN = TrainConfig()
DEFAULT_EVAL = EvalConfig()
DEFAULT_RUN = RunConfig()
