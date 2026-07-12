"""Gebündelter, paralleler Self-Play – der GPU-Auslastungs-Fix.

Das sequenzielle Self-Play (:mod:`az.selfplay`) spielt eine Partie nach der
anderen und bewertet in jeder MCTS-Simulation *ein* Brett mit Batch-Größe 1.
Bei einem winzigen Netz dominiert dann der CPU↔GPU-Sync – die GPU wartet die
meiste Zeit.

Hier laufen viele Partien **gleichzeitig**. Der Scheduler treibt in jeder Runde
jede aktive Partie bis genau zu dem Punkt, an dem sie eine Netz-Bewertung braucht
(ein Blatt im Suchbaum), sammelt diese Blätter über *alle* Partien ein und wertet
sie in **einem** Forward-Pass aus (:func:`az.mcts.evaluate_batch`).
"""

from __future__ import annotations

import numpy as np

from az.encoding import encode_state, index_to_move
from az.mcts import NeuralMCTS, _Node, evaluate_batch
from az.net import OthelloNet
from az.replay import ReplayBuffer, Sample
from az.selfplay import augment
from config import DEFAULT_SELFPLAY, MCTSConfig, SelfPlayConfig
from othello.board import EMPTY, PASS, GameState


class _Worker:
    """Zustandsmaschine für *eine* laufende Self-Play-Partie im Scheduler.

    ``collect`` treibt die Partie bis zur nächsten nötigen Netz-Bewertung und gibt
    die zu bewertende Stellung zurück; ``apply`` speist das Ergebnis zurück
    (Kind-Expansion + Backprop). Ist die Partie fertig und das Partie-Budget
    aufgebraucht, liefert ``collect`` ``None`` – der Worker ist erledigt.
    """

    def __init__(
        self,
        mcts: NeuralMCTS,
        size: int,
        config: SelfPlayConfig,
        rng: np.random.Generator,
        budget: "_Budget",
        out: list[Sample],
    ) -> None:
        self.mcts = mcts
        self.size = size
        self.config = config
        self.rng = rng
        self.budget = budget
        self.out = out            # gemeinsame Sammelliste roher (un-augmentierter) Samples

        self.state: GameState | None = None
        self.root: _Node | None = None
        self.pending: _Node | None = None    # Blatt, das gerade auf eine Bewertung wartet
        self.sims_done = 0
        self.move_count = 0
        self.history: list[tuple[np.ndarray, np.ndarray, int]] = []
        self.done = False

    # --- Scheduler-Schnittstelle ---

    def collect(self) -> GameState | None:
        """Treibt die Partie bis zur nächsten Netz-Bewertung. Rückgabe: Stellung oder None."""
        while True:
            if self.done:
                return None

            if self.root is None:
                # Kein aktiver Suchbaum: (ggf. neue) Partie bis zum nächsten
                # Entscheidungspunkt vorspulen und dort die Wurzel anlegen.
                if not self._begin_search():
                    continue          # Partie beendet/gewechselt -> Schleife erneut
                # Die Wurzel-Bewertung zählt als erste Simulation.
                self.pending = self.root
                self.sims_done = 1
                return self.root.state

            # Aktiver Suchbaum: entweder Zug abschließen oder nächstes Blatt suchen.
            if self.sims_done >= self.mcts.config.n_simulations:
                self._finish_move()
                continue

            leaf = self.mcts._select_leaf(self.root)
            self.sims_done += 1
            if leaf.state.is_terminal():
                # Terminale Blätter brauchen kein Netz – direkt zurückpropagieren.
                self.mcts._backprop(leaf, self.mcts._terminal_value(leaf.state))
                continue
            self.pending = leaf
            return leaf.state

    def apply(self, priors: np.ndarray, value: float) -> None:
        """Speist die Bewertung des zuletzt gemeldeten Blatts zurück."""
        node = self.pending
        assert node is not None
        self.mcts._expand_with_priors(node, priors)
        # Wurzel-Rauschen erst nach der Expansion mischen (wie im sequenziellen run).
        if node.parent is None and self.mcts.add_noise:
            self.mcts._add_dirichlet_noise(node)
        self.mcts._backprop(node, value)
        self.pending = None

    # --- interne Ablaufsteuerung ---

    def _begin_search(self) -> bool:
        """Legt die Suchwurzel am nächsten Entscheidungspunkt an.

        Startet bei Bedarf eine neue Partie, spult erzwungene Pässe/terminale
        Stellungen ab und finalisiert eine beendete Partie. Rückgabe: ``True``,
        wenn eine Wurzel für eine echte Suche steht, sonst ``False`` (Aufrufer
        soll ``collect`` erneut durchlaufen – neue Partie oder Worker fertig).
        """
        if self.state is None:
            if not self.budget.take():
                self.done = True
                return False
            self.state = GameState.initial(self.size)
            self.move_count = 0
            self.history = []

        while True:
            if self.state.is_terminal():
                self._finalize_game()
                self.state = None
                return False
            if self.move_count >= self.config.max_moves:
                raise RuntimeError("max_moves überschritten – vermutlich ein Engine-Bug")
            if self.state.legal_moves() == [PASS]:   # erzwungenes Pass: kein Lernsignal
                self.state = self.state.apply(PASS)
                continue
            self.root = _Node(self.state, parent=None, mover=None, prior=1.0)
            return True

    def _finish_move(self) -> None:
        """Schließt die Suche für den aktuellen Zug ab: Sample sichern, Zug spielen."""
        assert self.root is not None and self.state is not None
        policy_target = NeuralMCTS._visit_distribution(self.root, temperature=1.0)
        self.history.append((encode_state(self.state), policy_target, self.state.current_player))

        temperature = 1.0 if self.move_count < self.config.temperature_moves else 0.0
        pi = NeuralMCTS._visit_distribution(self.root, temperature)
        if temperature == 0:
            index = int(np.argmax(pi))
        else:
            index = int(self.rng.choice(len(pi), p=pi))
        self.state = self.state.apply(index_to_move(index, self.size))
        self.move_count += 1
        self.root = None
        self.sims_done = 0

    def _finalize_game(self) -> None:
        """Füllt die Value-Targets aus dem Partie-Ergebnis und hängt Samples an ``out``."""
        winner = self.state.winner()
        for planes, policy, player in self.history:
            if winner == EMPTY:
                value = 0.0
            else:
                value = 1.0 if winner == player else -1.0
            self.out.append(Sample(planes=planes, policy=policy, value=value))
        self.history = []


class _Budget:
    """Gemeinsames Partie-Kontingent: deckelt die Gesamtzahl gespielter Partien."""

    def __init__(self, total: int) -> None:
        self.remaining = total

    def take(self) -> bool:
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True


def generate_games_parallel(
    net: OthelloNet,
    size: int,
    n_games: int,
    *,
    mcts_config: MCTSConfig,
    config: SelfPlayConfig = DEFAULT_SELFPLAY,
    device: str | np.generic | None = None,
    seed: int = 0,
    buffer: ReplayBuffer | None = None,
) -> list[Sample]:
    """Spielt ``n_games`` Self-Play-Partien gebündelt und gibt die Samples zurück.

    Bis zu ``config.n_parallel`` Partien laufen gleichzeitig; ihre Blatt-Bewertungen
    werden pro Runde in einem Forward-Pass gebündelt. Die (bei ``config.augment``)
    augmentierten Samples werden zurückgegeben und – falls übergeben – an ``buffer``
    angehängt.
    """
    if device is None:
        device = next(net.parameters()).device

    budget = _Budget(n_games)
    raw: list[Sample] = []
    n_workers = min(config.n_parallel, n_games)
    workers = [
        _Worker(
            NeuralMCTS(net, mcts_config, device=device, add_noise=True, seed=seed + i),
            size,
            config,
            np.random.default_rng(seed + 10_000 + i),
            budget,
            raw,
        )
        for i in range(n_workers)
    ]

    while True:
        states: list[GameState] = []
        pending_workers: list[_Worker] = []
        for w in workers:
            if w.done:
                continue
            state = w.collect()
            if state is None:
                continue
            states.append(state)
            pending_workers.append(w)

        if not states:
            break

        results = evaluate_batch(net, states, device)
        for w, (priors, value) in zip(pending_workers, results):
            w.apply(priors, value)

    samples = augment(raw, size) if config.augment else raw
    if buffer is not None:
        buffer.add(samples)
    return samples
