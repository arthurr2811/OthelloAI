"""Tests für Schritt 1.5 – MCTS ohne Netz."""

import numpy as np

from agents.arena import play_match
from agents.mcts import MCTSAgent
from agents.simple import GreedyAgent
from othello.board import BLACK, EMPTY, WHITE, PASS, GameState


def test_mcts_returns_legal_move():
    agent = MCTSAgent(n_simulations=50, seed=0)
    state = GameState.initial(8)
    move = agent.select_move(state)
    assert move in state.legal_moves()


def test_mcts_returns_pass_when_forced():
    board = np.array(
        [
            [EMPTY, BLACK, WHITE, WHITE],
            [BLACK, BLACK, WHITE, WHITE],
            [WHITE, WHITE, WHITE, WHITE],
            [WHITE, WHITE, WHITE, WHITE],
        ],
        dtype=np.int8,
    )
    state = GameState(board=board, current_player=BLACK)
    assert MCTSAgent(n_simulations=10, seed=0).select_move(state) == PASS


def test_mcts_takes_immediate_winning_corner():
    # Endstellung: Schwarz kann mit einem Zug in die Ecke die Partie klar drehen.
    # Board so gebaut, dass (0,0) viele Steine flippt und das Spiel beendet.
    board = np.array(
        [
            [EMPTY, WHITE, WHITE, BLACK],
            [WHITE, WHITE, WHITE, WHITE],
            [WHITE, WHITE, WHITE, WHITE],
            [BLACK, WHITE, WHITE, WHITE],
        ],
        dtype=np.int8,
    )
    state = GameState(board=board, current_player=BLACK)
    # (0,0) klammert die obere Reihe (0,1),(0,2) gegen (0,3)=Schwarz ein.
    move = MCTSAgent(n_simulations=200, seed=0).select_move(state)
    assert move == (0, 0)


def test_mcts_beats_greedy():
    # Der zentrale Sanity-Check aus dem Plan: MCTS schlägt Greedy klar.
    # Seed-fest und daher deterministisch; wenige Sims/Spiele halten den Test
    # schnell, MCTS dominiert Greedy trotzdem deutlich (Referenzmessung:
    # 150 Sims, 30 Spiele -> 100 %).
    mcts = MCTSAgent(n_simulations=30, seed=0)
    greedy = GreedyAgent(seed=1)
    result = play_match(mcts, greedy, n_games=12, size=8)
    assert result.win_rate > 0.70, result.summary()
