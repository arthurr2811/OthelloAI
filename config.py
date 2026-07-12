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
    # Anzahl Partien, die im parallelen Self-Play gleichzeitig laufen. Alle ihre
    # Blatt-Bewertungen werden pro Runde zu *einem* Netz-Forward gebündelt – das
    # ist der Hebel, der die GPU auslastet (Batch-1-Inferenz war der Flaschenhals).
    n_parallel: int = 32
    # Hebel C: Anzahl Self-Play-Prozesse (1 = kein Multiprocessing). Ab 2 teilen
    # sich Worker-Prozesse die Partien einer Iteration (az/selfplay_mp.py) –
    # greift den Single-Core-Engpass direkt an. Sinnvoll: Kernzahl minus ~2.
    n_workers: int = 1


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


@dataclass(frozen=True)
class RunConfig:
    """Zentrale Config für den ganzen AlphaZero-Loop (Schritt 2.6).

    Bündelt Brettgröße, Loop-Umfang und alle Sub-Configs an *einer* Stelle, damit
    ein Lauf reproduzierbar über ein einziges Objekt beschrieben ist. Defaults
    zielen auf den 6x6-Durchstich (klein, konvergiert in Stunden).
    """

    board_size: int = 6
    n_iterations: int = 40
    games_per_iteration: int = 20        # Self-Play-Partien pro Iteration
    train_steps_per_iteration: int = 200  # Gradientenschritte pro Iteration
    baseline_games: int = 10             # Partien gegen Greedy für die Stärke-Kurve
    seed: int = 0

    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Sub-Configs (unveränderlich, daher als Default-Instanzen teilbar).
    net: NetConfig = NetConfig()
    mcts: MCTSConfig = MCTSConfig(n_simulations=64)
    selfplay: SelfPlayConfig = SelfPlayConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig(n_games=20)


# 8x8-Zielkonfiguration (Schritt 2.7). Skaliert die drei "Gratis"-Hebel aus dem
# README: G (größeres Netz – die GPU hat Reserve), A (mehr parallele Partien pro
# Iteration – bessere CPU/GPU-Überlappung, mehr Daten pro Iteration) und
# B (Sims/Iterationen bewusst gewählt statt blind hochgedreht).
# Eigene Checkpoint-/Log-Verzeichnisse, damit der 6x6-Stand unangetastet bleibt.
RUN_8X8 = RunConfig(
    board_size=8,
    n_iterations=120,
    games_per_iteration=96,          # A: mehr Partien je Iteration (6x6: 20)
    train_steps_per_iteration=500,   # ~3 Sichtungen je neuem Sample bei Batch 256
    baseline_games=10,
    checkpoint_dir="checkpoints/8x8",
    log_dir="logs/8x8",
    net=NetConfig(channels=128, n_res_blocks=8, value_hidden=128),  # G (6x6: 64ch/4)
    mcts=MCTSConfig(n_simulations=128),                             # B (6x6: 64)
    selfplay=SelfPlayConfig(
        temperature_moves=20,        # 8x8-Partien sind ~doppelt so lang wie 6x6
        buffer_size=400_000,         # ~9 Iterationen Self-Play-Daten
        n_parallel=96,               # A: alle Partien einer Iteration gleichzeitig
        n_workers=6,                 # C: 6 von 8 Kernen (Rest: Hauptprozess + OS)
    ),
    train=TrainConfig(batch_size=256),
    eval=EvalConfig(n_games=24),
)


DEFAULT_GAME = GameConfig()
DEFAULT_NET = NetConfig()
DEFAULT_MCTS = MCTSConfig()
DEFAULT_SELFPLAY = SelfPlayConfig()
DEFAULT_TRAIN = TrainConfig()
DEFAULT_EVAL = EvalConfig()
DEFAULT_RUN = RunConfig()
