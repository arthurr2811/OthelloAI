"""Baseline-Bots: RandomAgent und GreedyAgent.

Beide dienen als Messlatte für stärkere Agenten (MCTS, AlphaZero). Greedy soll
Random deutlich schlagen – das ist der Sanity-Check in Schritt 1.4.
"""

from __future__ import annotations

import random

from othello.board import PASS, GameState, Move, flips_for_move

from .base import Agent


class RandomAgent(Agent):
    """Wählt gleichverteilt einen der legalen Züge."""

    name = "Random"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select_move(self, state: GameState) -> Move:
        return self._rng.choice(state.legal_moves())


class GreedyAgent(Agent):
    """Wählt den Zug, der die meisten gegnerischen Steine umdreht.

    Kurzsichtig (nur ein Halbzug tief), aber deutlich besser als Zufall.
    Gleichstände werden zufällig aufgelöst, damit keine systematische
    Feld-Verzerrung entsteht.
    """

    name = "Greedy"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select_move(self, state: GameState) -> Move:
        options = state.legal_moves()
        if options == [PASS]:
            return PASS

        best_gain = -1
        best_moves: list[Move] = []
        for move in options:
            gain = len(flips_for_move(state.board, state.current_player, move))
            if gain > best_gain:
                best_gain = gain
                best_moves = [move]
            elif gain == best_gain:
                best_moves.append(move)
        return self._rng.choice(best_moves)
