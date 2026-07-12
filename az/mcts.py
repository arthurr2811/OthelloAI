"""PUCT-MCTS mit neuronalem Netz (AlphaZero-Stil).

Unterschied zu :mod:`agents.mcts` (reines UCT): Statt Random-Rollouts bewertet
das Netz jede neu expandierte Stellung direkt (Value), und die Policy-Priors
lenken die Suche (PUCT statt UCB1).

Wert-Konvention (Vorzeichen ist hier die Hauptfehlerquelle):
    Das Netz liefert ``value`` aus Sicht des Spielers **am Zug** in der bewerteten
    Stellung. Jeder Knoten speichert ``W`` aus Sicht seines ``mover`` – also des
    Spielers, der den Zug *in diesen Knoten hinein* gemacht hat. ``Q = W / N`` ist
    damit der erwartete Ausgang in ``[-1, 1]`` für den, der den Zug wählte – genau
    das, was die PUCT-Selektion am Elternknoten maximieren will. Beim Backprop
    wird das Vorzeichen pro Ebene gekippt, weil sich der Spieler am Zug abwechselt
    (auch bei PASS wechselt ``current_player``).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from az.encoding import (
    encode_state,
    index_to_move,
    legal_move_mask,
    move_to_index,
    num_actions,
)
from az.net import OthelloNet
from config import DEFAULT_MCTS, MCTSConfig
from othello.board import EMPTY, GameState, Move

from agents.base import Agent


class _Node:
    """Ein Knoten im Suchbaum: Stellung plus PUCT-Statistik."""

    __slots__ = ("state", "parent", "mover", "prior", "children", "N", "W", "is_expanded")

    def __init__(self, state: GameState, parent: "_Node | None", mover: int | None, prior: float):
        self.state = state
        self.parent = parent
        # Spieler, der den Zug in diesen Knoten hinein machte. None = Wurzel.
        self.mover = mover
        self.prior = prior              # P(a) aus der Policy des Elternteils
        self.children: dict[Move, _Node] = {}
        self.N = 0
        self.W = 0.0
        self.is_expanded = False

    @property
    def q(self) -> float:
        """Mittlerer Wert aus Sicht des ``mover`` (0, solange unbesucht)."""
        return self.W / self.N if self.N > 0 else 0.0


class NeuralMCTS:
    """PUCT-Suche über ein :class:`OthelloNet`.

    Trennt Suche (``run`` -> Wurzelknoten mit Visit-Counts) von der Zugauswahl
    (``policy_target`` / ``select_move``), damit der Self-Play-Loop (Schritt 2.3)
    an die Visit-Verteilung als Trainingsziel kommt.
    """

    def __init__(
        self,
        net: OthelloNet,
        config: MCTSConfig = DEFAULT_MCTS,
        *,
        device: str | torch.device | None = None,
        add_noise: bool = False,
        seed: int | None = None,
    ) -> None:
        self.net = net
        self.config = config
        self.device = torch.device(device) if device is not None else next(net.parameters()).device
        self.add_noise = add_noise
        self._rng = np.random.default_rng(seed)

    # --- Öffentliche API ---

    def run(self, state: GameState) -> _Node:
        """Führt das Simulationsbudget aus und gibt den Wurzelknoten zurück."""
        root = _Node(state, parent=None, mover=None, prior=1.0)
        value = self._expand_and_evaluate(root)
        self._backprop(root, value)  # Wurzel-Evaluation zählt als erste Simulation
        if self.add_noise:
            self._add_dirichlet_noise(root)
        for _ in range(self.config.n_simulations - 1):
            self._simulate(root)
        return root

    def policy_target(self, state: GameState, temperature: float | None = None) -> np.ndarray:
        """Visit-Count-Verteilung über alle Aktionen (Länge ``S*S+1``).

        Mit ``temperature`` skaliert: ``pi ∝ N^(1/tau)``. ``tau=0`` legt die ganze
        Masse auf den meistbesuchten Zug. Das ist das Policy-Trainingsziel.
        """
        tau = self.config.temperature if temperature is None else temperature
        root = self.run(state)
        return self._visit_distribution(root, tau)

    def select_move(self, state: GameState, temperature: float | None = None) -> Move:
        """Wählt einen Zug: bei ``tau>0`` gesampelt, bei ``tau=0`` der beste."""
        options = state.legal_moves()
        if len(options) == 1:            # erzwungener Zug/Pass: keine Suche nötig
            return options[0]
        tau = self.config.temperature if temperature is None else temperature
        root = self.run(state)
        pi = self._visit_distribution(root, tau)
        size = state.size
        if tau == 0:
            index = int(np.argmax(pi))
        else:
            index = int(self._rng.choice(len(pi), p=pi))
        return index_to_move(index, size)

    # --- Suche ---

    def _simulate(self, root: _Node) -> None:
        # 1. Selection: PUCT-Abstieg zu einem Blatt.
        node = self._select_leaf(root)

        # 2. Auswertung des Blatts: terminal -> echtes Ergebnis, sonst Netz.
        if node.state.is_terminal():
            value = self._terminal_value(node.state)
        else:
            value = self._expand_and_evaluate(node)

        # 3. Backpropagation.
        self._backprop(node, value)

    def _select_leaf(self, root: _Node) -> _Node:
        """PUCT-Abstieg durch expandierte Knoten bis zu einem Blatt.

        Blatt = noch nicht expandierter Knoten (braucht eine Netz-Bewertung) oder
        eine terminale Stellung. Diese Trennung von Auswahl und Bewertung ist die
        Basis fürs gebündelte Self-Play: der Scheduler sammelt die Blätter vieler
        Partien und bewertet sie in *einem* Forward-Pass.
        """
        node = root
        while node.is_expanded and node.children:
            node = self._select_child(node)
        return node

    def _select_child(self, node: _Node) -> _Node:
        """PUCT: ``Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))``."""
        sqrt_parent = math.sqrt(node.N)
        c_puct = self.config.c_puct
        best_score = -math.inf
        best_child: _Node | None = None
        for child in node.children.values():
            u = c_puct * child.prior * sqrt_parent / (1 + child.N)
            score = child.q + u
            if score > best_score:
                best_score = score
                best_child = child
        assert best_child is not None
        return best_child

    def _expand_and_evaluate(self, node: _Node) -> float:
        """Bewertet ``node`` mit dem Netz (Batch-1) und expandiert es.

        Rückgabe: Value aus Sicht des Spielers am Zug in ``node.state``.
        """
        priors, value = self._evaluate(node.state)
        self._expand_with_priors(node, priors)
        return value

    @staticmethod
    def _expand_with_priors(node: _Node, priors: np.ndarray) -> None:
        """Legt alle legalen Kinder von ``node`` mit ihren Policy-Priors an.
        """
        mover = node.state.current_player
        for move in node.state.legal_moves():
            idx = move_to_index(move, node.state.size)
            child_state = node.state.apply(move)
            node.children[move] = _Node(child_state, parent=node, mover=mover, prior=float(priors[idx]))
        node.is_expanded = True

    def _backprop(self, leaf: _Node, value: float) -> None:
        """Trägt ``value`` (Sicht: ``leaf.current_player``) den Pfad hinauf.

        ``W`` jedes Knotens ist aus Sicht seines ``mover`` = Gegner des Spielers am
        Zug in diesem Knoten, daher ``-value``. Pro Ebene kippt das Vorzeichen.
        """
        node: _Node | None = leaf
        while node is not None:
            node.N += 1
            if node.mover is not None:
                node.W += -value
            value = -value
            node = node.parent

    # --- Netz-Auswertung ---

    def _evaluate(self, state: GameState) -> tuple[np.ndarray, float]:
        """Netz-Forward für eine Stellung.

        Rückgabe: ``(priors, value)`` mit über die legalen Züge maskierter und
        normalisierter Policy (0 auf illegalen Feldern) und Value in ``[-1, 1]``.
        """
        mask = legal_move_mask(state)
        x = torch.from_numpy(encode_state(state)).unsqueeze(0).to(self.device)
        was_training = self.net.training
        self.net.eval()
        with torch.no_grad():
            logits, value = self.net(x)
        if was_training:
            self.net.train()
        logits = logits[0].detach().cpu().numpy()
        priors = self._masked_softmax(logits, mask)
        return priors, float(value[0])

    @staticmethod
    def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Softmax nur über die legalen Aktionen; Rest exakt 0."""
        masked = np.where(mask, logits, -np.inf)
        masked = masked - masked.max()          # numerische Stabilität
        exp = np.where(mask, np.exp(masked), 0.0)
        total = exp.sum()
        if total == 0:                           # Fallback: gleichverteilt über legal
            return mask / mask.sum()
        return exp / total

    @staticmethod
    def _terminal_value(state: GameState) -> float:
        """Echtes Partie-Ergebnis aus Sicht des Spielers am Zug in ``state``."""
        w = state.winner()
        if w == EMPTY:
            return 0.0
        return 1.0 if w == state.current_player else -1.0

    # --- Wurzel-Rauschen & Zugverteilung ---

    def _add_dirichlet_noise(self, root: _Node) -> None:
        """Mischt Dirichlet-Rauschen in die Wurzel-Priors (Self-Play-Exploration)."""
        moves = list(root.children.keys())
        if not moves:
            return
        eps = self.config.dirichlet_epsilon
        noise = self._rng.dirichlet([self.config.dirichlet_alpha] * len(moves))
        for move, n in zip(moves, noise):
            child = root.children[move]
            child.prior = (1 - eps) * child.prior + eps * float(n)

    @staticmethod
    def _visit_distribution(root: _Node, temperature: float) -> np.ndarray:
        """Normalisierte Visit-Counts über alle Aktionen, mit Temperatur."""
        size = root.state.size
        counts = np.zeros(num_actions(size), dtype=np.float64)
        for move, child in root.children.items():
            counts[move_to_index(move, size)] = child.N
        if counts.sum() == 0:                    # Wurzel ohne Sims: gleichverteilt legal
            counts[[move_to_index(m, size) for m in root.children]] = 1.0

        if temperature == 0:
            pi = np.zeros_like(counts)
            pi[int(np.argmax(counts))] = 1.0
            return pi
        scaled = counts ** (1.0 / temperature)
        return scaled / scaled.sum()


def evaluate_batch(
    net: OthelloNet,
    states: list[GameState],
    device: str | torch.device,
) -> list[tuple[np.ndarray, float]]:
    """Bewertet viele Stellungen in *einem* Forward-Pass
    """
    planes = np.stack([encode_state(s) for s in states])
    x = torch.from_numpy(planes).to(device)
    was_training = net.training
    net.eval()
    with torch.no_grad():
        logits, values = net(x)
    if was_training:
        net.train()
    logits_np = logits.detach().cpu().numpy()
    values_np = values.detach().cpu().numpy()
    out: list[tuple[np.ndarray, float]] = []
    for i, state in enumerate(states):
        mask = legal_move_mask(state)
        priors = NeuralMCTS._masked_softmax(logits_np[i], mask)
        out.append((priors, float(values_np[i])))
    return out


class NeuralMCTSAgent(Agent):
    """Adapter, der ``NeuralMCTS`` in das :class:`~agents.base.Agent`-Interface steckt.

    Temperatur-Schedule: die ersten ``temperature_moves`` Züge werden mit
    ``temperature`` gesampelt, danach deterministisch (bestes Spiel). Für reines
    Spielen genügt ``temperature_moves=0`` (immer der beste Zug); in der Evaluation
    sorgt ein kleines Zeitfenster mit ``temperature=1`` für Partievielfalt, ohne die
    Endspielstärke zu verwässern.

    Die Zugnummer wird statuslos aus der Steinzahl abgeleitet (``nonzero - 4``), da
    ``select_move`` keinen Partieverlauf mitführt.
    """

    name = "NeuralMCTS"

    def __init__(
        self,
        net: OthelloNet,
        config: MCTSConfig = DEFAULT_MCTS,
        *,
        device: str | torch.device | None = None,
        temperature: float = 0.0,
        temperature_moves: int = 0,
        seed: int | None = None,
    ) -> None:
        self._mcts = NeuralMCTS(net, config, device=device, add_noise=False, seed=seed)
        self._temperature = temperature
        self._temperature_moves = temperature_moves

    def select_move(self, state: GameState) -> Move:
        move_number = int(np.count_nonzero(state.board)) - 4  # 4 Startsteine
        temp = self._temperature if move_number < self._temperature_moves else 0.0
        return self._mcts.select_move(state, temperature=temp)
