"""Gebündelte, parallele Arena für die Evaluation (Gating + Baseline).

Analog zu :mod:`az.selfplay_parallel`, aber für *Matches* zweier Spieler statt
Self-Play.

Zwei Spielertypen:
    * :class:`NetPlayer`  – PUCT-MCTS über ein Netz (gebündelte Bewertung).
    * :class:`AgentPlayer` – beliebiger :class:`~agents.base.Agent` (Zug sofort).

Die Farbrotation und die Bilanz-Attribution entsprechen exakt
:func:`agents.arena.play_match` (Ergebnis aus Sicht von Spieler A).
"""

from __future__ import annotations

import numpy as np
import torch

from az.encoding import index_to_move
from az.mcts import NeuralMCTS, _Node, evaluate_batch
from az.net import OthelloNet
from config import DEFAULT_MCTS, MCTSConfig
from othello.board import BLACK, EMPTY, WHITE, GameState

from agents.arena import MatchResult
from agents.base import Agent


class NetPlayer:
    """Ein Netz-Spieler: PUCT-MCTS mit gebündelter Blatt-Bewertung.

    Kein Wurzel-Rauschen (reines Spiel). Temperatur-Schedule wie
    :class:`~az.mcts.NeuralMCTSAgent`: die ersten ``temperature_moves`` Züge werden
    mit ``temperature`` gesampelt, danach deterministisch. Die Zugnummer wird aus
    der Steinzahl abgeleitet (``nonzero - 4``), passend zum sequenziellen Pfad.
    """

    is_net = True

    def __init__(
        self,
        net: OthelloNet,
        name: str,
        mcts_config: MCTSConfig = DEFAULT_MCTS,
        *,
        temperature: float = 1.0,
        temperature_moves: int = 0,
        device: str | torch.device | None = None,
        seed: int | None = None,
    ) -> None:
        self.net = net
        self.name = name
        self.mcts = NeuralMCTS(net, mcts_config, device=device, add_noise=False, seed=seed)
        self.temperature = temperature
        self.temperature_moves = temperature_moves

    def __str__(self) -> str:
        return self.name


class AgentPlayer:
    """Adapter für einen beliebigen :class:`~agents.base.Agent` (zieht sofort, kein Netz)."""

    is_net = False

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.name = str(agent)

    def __str__(self) -> str:
        return self.name


class _Worker:
    """Spielt *eine* Partie im Scheduler; wechselt die Seite je nach Spieler am Zug.

    ``collect`` treibt die Partie bis zur nächsten nötigen Netz-Bewertung und gibt
    ``(net_player, state)`` zurück; einfache Züge (AgentPlayer, erzwungene Züge)
    werden dabei inline gespielt. Bei Spielende wird die Bilanz gebucht und
    ``None`` zurückgegeben.
    """

    def __init__(self, size: int, budget: "_Budget", result: MatchResult,
                 player_a, player_b) -> None:
        self.size = size
        self.budget = budget
        self.result = result
        self.player_a = player_a
        self.player_b = player_b

        self.state: GameState | None = None
        self.a_is_black = True
        self.black = player_a
        self.white = player_b

        self.root: _Node | None = None
        self.current: NetPlayer | None = None   # Netz-Spieler, der gerade sucht
        self.pending: _Node | None = None
        self.sims_done = 0
        self.plies = 0
        self.done = False

    # --- Scheduler-Schnittstelle ---

    def collect(self):
        """Treibt die Partie bis zur nächsten Netz-Bewertung. Rückgabe: (player, state) oder None."""
        while True:
            if self.done:
                return None

            if self.root is None:
                phase = self._advance()
                if phase == "terminal":
                    self._record_result()
                    self.state = None
                    self.root = None
                    continue                 # nächste Partie holen (oder fertig)
                if phase == "done":
                    self.done = True
                    return None
                if phase == "continue":
                    continue                 # inline-Zug gespielt -> weiter
                # phase == "search": Wurzel steht, Wurzel-Bewertung = erste Simulation.
                self.pending = self.root
                self.sims_done = 1
                return self.current, self.root.state

            # Aktiver Suchbaum eines Netz-Spielers.
            if self.sims_done >= self.current.mcts.config.n_simulations:
                self._finish_move()
                continue

            mcts = self.current.mcts
            leaf = mcts._select_leaf(self.root)
            self.sims_done += 1
            if leaf.state.is_terminal():
                mcts._backprop(leaf, mcts._terminal_value(leaf.state))
                continue
            self.pending = leaf
            return self.current, leaf.state

    def apply(self, priors: np.ndarray, value: float) -> None:
        """Speist die Bewertung des zuletzt gemeldeten Blatts zurück (Expansion + Backprop)."""
        node = self.pending
        assert node is not None and self.current is not None
        self.current.mcts._expand_with_priors(node, priors)
        self.current.mcts._backprop(node, value)
        self.pending = None

    # --- interne Ablaufsteuerung ---

    def _advance(self) -> str:
        """Spielt inline-Züge bis zum nächsten Netz-Entscheidungspunkt.

        Rückgabe:
            "search"   – Wurzel für einen Netz-Zug angelegt (``self.current`` gesetzt);
            "terminal" – Partie zu Ende (Bilanz buchen, neue Partie holen);
            "continue" – ein inline-Zug (AgentPlayer/erzwungen) wurde gespielt;
            "done"     – Partie-Budget leer, Worker fertig.
        """
        if self.state is None:
            idx = self.budget.take()
            if idx is None:
                return "done"
            self.state = GameState.initial(self.size)
            self.a_is_black = (idx % 2 == 0)
            self.black = self.player_a if self.a_is_black else self.player_b
            self.white = self.player_b if self.a_is_black else self.player_a
            self.plies = 0

        if self.state.is_terminal():
            return "terminal"
        if self.plies > self.size * self.size * 4:
            raise RuntimeError("zu viele Halbzüge ohne Spielende – vermutlich ein Engine-Bug")

        mover = self.black if self.state.current_player == BLACK else self.white
        options = self.state.legal_moves()

        # Erzwungener Einzelzug/Pass: keine Suche, kein Netz.
        if len(options) == 1:
            self.state = self.state.apply(options[0])
            self.plies += 1
            return "continue"

        if not mover.is_net:                 # einfacher Agent: Zug sofort
            self.state = self.state.apply(mover.agent.select_move(self.state))
            self.plies += 1
            return "continue"

        # Netz-Spieler mit echter Wahl: Suche aufsetzen.
        self.current = mover
        self.root = _Node(self.state, parent=None, mover=None, prior=1.0)
        return "search"

    def _finish_move(self) -> None:
        """Schließt die Netz-Suche ab: Zug gemäß Temperatur wählen und spielen."""
        assert self.root is not None and self.current is not None and self.state is not None
        player = self.current
        move_number = int(np.count_nonzero(self.state.board)) - 4
        temp = player.temperature if move_number < player.temperature_moves else 0.0
        pi = NeuralMCTS._visit_distribution(self.root, temp)
        if temp == 0:
            index = int(np.argmax(pi))
        else:
            index = int(player.mcts._rng.choice(len(pi), p=pi))
        self.state = self.state.apply(index_to_move(index, self.size))
        self.plies += 1
        self.root = None
        self.current = None
        self.sims_done = 0

    def _record_result(self) -> None:
        """Bucht das Partie-Ergebnis in die geteilte ``MatchResult`` (Sicht Spieler A)."""
        winner = self.state.winner()
        a_color = BLACK if self.a_is_black else WHITE
        if winner == EMPTY:
            self.result.draws += 1
        elif winner == a_color:
            self.result.wins += 1
        else:
            self.result.losses += 1


class _Budget:
    """Gemeinsames Partie-Kontingent: gibt aufsteigende Partie-Indizes aus (für die Farbrotation)."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.next = 0

    def take(self):
        if self.next >= self.total:
            return None
        idx = self.next
        self.next += 1
        return idx


def play_match_parallel(
    player_a,
    player_b,
    n_games: int,
    size: int,
    *,
    n_parallel: int = 32,
    device: str | torch.device | None = None,
) -> MatchResult:
    """Spielt ``n_games`` Partien A vs. B gebündelt; Startfarbe wechselt pro Partie.

    Netz-Blätter werden pro Runde nach Netz gruppiert und je in *einem* Forward-Pass
    ausgewertet. Rückgabe: :class:`~agents.arena.MatchResult` aus Sicht von A –
    identische Semantik zu :func:`agents.arena.play_match`.
    """
    if device is None:
        for p in (player_a, player_b):
            if getattr(p, "is_net", False):
                device = next(p.net.parameters()).device
                break
    if device is None:
        device = "cpu"

    result = MatchResult(agent_a=str(player_a), agent_b=str(player_b))
    budget = _Budget(n_games)
    n_workers = min(n_parallel, n_games)
    workers = [_Worker(size, budget, result, player_a, player_b) for _ in range(n_workers)]

    while True:
        # Anfragen pro Runde nach Netz gruppieren: {id(net_player): [player, workers, states]}.
        groups: dict[int, list] = {}
        for w in workers:
            if w.done:
                continue
            req = w.collect()
            if req is None:
                continue
            player, state = req
            g = groups.setdefault(id(player), [player, [], []])
            g[1].append(w)
            g[2].append(state)

        if not groups:
            break

        for player, ws, states in groups.values():
            results = evaluate_batch(player.net, states, device)
            for w, (priors, value) in zip(ws, results):
                w.apply(priors, value)

    return result
