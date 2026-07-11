"""Tests für Schritt 1.4 – Baseline-Bots & Arena."""

import numpy as np

from agents.arena import MatchResult, play_game, play_match
from agents.simple import GreedyAgent, RandomAgent
from othello.board import BLACK, EMPTY, WHITE, PASS, GameState, flips_for_move


def test_random_agent_only_picks_legal_moves():
    agent = RandomAgent(seed=1)
    state = GameState.initial(8)
    for _ in range(5):
        if state.is_terminal():
            break
        move = agent.select_move(state)
        assert move in state.legal_moves()
        state = state.apply(move)


def test_random_agent_is_deterministic_with_seed():
    s = GameState.initial(8)
    m1 = RandomAgent(seed=42).select_move(s)
    m2 = RandomAgent(seed=42).select_move(s)
    assert m1 == m2


def test_greedy_picks_move_with_most_flips():
    # Konstruierte Stellung: ein Zug dreht 2 Steine, ein anderer nur 1.
    board = np.zeros((8, 8), dtype=np.int8)
    # Zug (4,4): rechts W,W dann B -> 2 Flips
    board[4, 5] = WHITE
    board[4, 6] = WHITE
    board[4, 7] = BLACK
    # Zug (0,4): unten W dann B -> 1 Flip
    board[1, 4] = WHITE
    board[2, 4] = BLACK
    state = GameState(board=board, current_player=BLACK)

    chosen = GreedyAgent(seed=0).select_move(state)
    assert chosen == (4, 4)
    assert len(flips_for_move(board, BLACK, chosen)) == 2


def test_greedy_returns_pass_when_forced():
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
    assert state.legal_moves() == [PASS]
    assert GreedyAgent(seed=0).select_move(state) == PASS


def test_play_game_returns_valid_winner():
    w = play_game(RandomAgent(seed=1), RandomAgent(seed=2), size=8)
    assert w in (BLACK, WHITE, EMPTY)


def test_match_result_accounting():
    r = MatchResult("A", "B", wins=6, losses=3, draws=1)
    assert r.games == 10
    assert r.win_rate == (6 + 0.5) / 10


def test_match_alternates_colors_and_counts_add_up():
    result = play_match(RandomAgent(seed=1), RandomAgent(seed=2), n_games=20, size=8)
    assert result.games == 20
    assert result.wins + result.losses + result.draws == 20


def test_greedy_beats_random():
    # Der zentrale Sanity-Check aus dem Plan. Greedy (max Flips) ist bei Othello
    # nur mäßig stark (~61% gegen Random), daher genug Partien für ein stabiles
    # Ergebnis und eine Schwelle mit Varianz-Puffer.
    greedy = GreedyAgent(seed=0)
    random_bot = RandomAgent(seed=1)
    result = play_match(greedy, random_bot, n_games=400, size=8)
    assert result.win_rate > 0.55, result.summary()
