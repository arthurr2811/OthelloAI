"""Reines UCT-MCTS mit Random-Rollouts – ohne neuronales Netz.

Wert-Konvention (wichtig, weil hier gern Vorzeichenfehler passieren):
    Jeder Knoten speichert ``W`` als Summe von Rollout-Ergebnissen aus Sicht des
    Spielers, der den Zug *in diesen Knoten hinein* gemacht hat (``mover``).
    Ergebnis je Rollout in [0, 1]: Sieg=1, Remis=0.5, Niederlage=0.
    Damit ist ``Q = W / N`` direkt die Gewinnquote dieses Zugs aus Sicht dessen,
    der ihn gewählt hat – genau der Wert, den die Selektion maximieren will.
"""

from __future__ import annotations

import math
import random

from othello.board import EMPTY, GameState, Move

from .base import Agent


class _Node:
    """Ein Knoten im Suchbaum: eine Stellung plus Statistik."""

    __slots__ = ("state", "parent", "mover", "player_to_move",
                 "children", "untried", "N", "W")

    def __init__(self, state: GameState, parent: "_Node | None", mover: int | None):
        self.state = state
        self.parent = parent
        # Spieler, der den Zug in diesen Knoten hinein gemacht hat. None = Wurzel.
        self.mover = mover
        self.player_to_move = state.current_player
        self.children: dict[Move, _Node] = {}
        # Noch nicht expandierte Optionen (reale Züge oder [PASS]); leer = terminal.
        self.untried: list[Move] = list(state.legal_moves())
        self.N = 0
        self.W = 0.0

    @property
    def is_terminal(self) -> bool:
        return not self.untried and not self.children

    @property
    def is_fully_expanded(self) -> bool:
        return not self.untried


class MCTSAgent(Agent):
    """UCT-MCTS-Agent mit konfigurierbarem Simulationsbudget."""

    name = "MCTS"

    def __init__(
        self,
        n_simulations: int = 200,
        c: float = math.sqrt(2),
        seed: int | None = None,
    ) -> None:
        self.n_simulations = n_simulations
        self.c = c
        self._rng = random.Random(seed)

    def select_move(self, state: GameState) -> Move:
        options = state.legal_moves()
        # Erzwungener Zug oder Pass: keine Suche nötig.
        if len(options) == 1:
            return options[0]

        root = _Node(state, parent=None, mover=None)
        for _ in range(self.n_simulations):
            self._simulate(root)

        # Zug mit den meisten Besuchen (robustestes Maß); Gleichstand zufällig.
        max_visits = max(child.N for child in root.children.values())
        best = [m for m, child in root.children.items() if child.N == max_visits]
        return self._rng.choice(best)

    def _simulate(self, root: _Node) -> None:
        # 1. Selection: durch voll expandierte Knoten zum Rand absteigen.
        node = root
        while not node.is_terminal and node.is_fully_expanded:
            node = self._best_uct_child(node)

        # 2. Expansion: einen noch nicht probierten Zug anhängen.
        if not node.is_terminal:
            move = self._rng.choice(node.untried)
            node.untried.remove(move)
            child_state = node.state.apply(move)
            child = _Node(child_state, parent=node, mover=node.player_to_move)
            node.children[move] = child
            node = child

        # 3. Rollout: von hier zufällig bis zum Spielende.
        winner = self._rollout(node.state)

        # 4. Backpropagation: Ergebnis den Pfad hinauf tragen.
        while node is not None:
            node.N += 1
            if node.mover is not None:
                node.W += self._value(winner, node.mover)
            node = node.parent

    def _best_uct_child(self, node: _Node) -> _Node:
        ln_parent = math.log(node.N)
        best_score = -math.inf
        best_child: _Node | None = None
        for child in node.children.values():
            q = child.W / child.N               # Gewinnquote aus mover-Sicht
            u = self.c * math.sqrt(ln_parent / child.N)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        assert best_child is not None
        return best_child

    def _rollout(self, state: GameState) -> int:
        while not state.is_terminal():
            move = self._rng.choice(state.legal_moves())
            state = state.apply(move)
        return state.winner()

    @staticmethod
    def _value(winner: int, mover: int) -> float:
        """Rollout-Ergebnis in [0, 1] aus Sicht von ``mover``."""
        if winner == EMPTY:
            return 0.5
        return 1.0 if winner == mover else 0.0
