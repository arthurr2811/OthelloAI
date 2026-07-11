"""Tests für Schritt 2.5: Evaluation & Gating."""

from __future__ import annotations

import copy

import numpy as np
import torch

from az.evaluate import evaluate_vs_baseline, gate, make_agent
from az.net import OthelloNet
from config import EvalConfig, MCTSConfig
from othello.board import GameState

from agents.simple import RandomAgent


def _net(size=6, seed=0):
    torch.manual_seed(seed)
    return OthelloNet(board_size=size).eval()


_SMALL_MCTS = MCTSConfig(n_simulations=8)
_SMALL_EVAL = EvalConfig(n_games=4, win_threshold=0.55, temperature_moves=6)


def test_gate_returns_decision_and_matchresult():
    cand = _net(seed=1)
    best = _net(seed=2)
    res = gate(cand, best, size=6, mcts_config=_SMALL_MCTS, eval_config=_SMALL_EVAL, seed=0)
    assert res.result.games == 4
    assert 0.0 <= res.win_rate <= 1.0
    assert res.accepted == (res.win_rate >= _SMALL_EVAL.win_threshold)
    assert "candidate" in res.summary()


def test_gate_promotes_on_threshold():
    # Deterministische Kontrolle der Gating-Logik über einen konstruierten Fall:
    # win_threshold 0.0 -> jeder Kandidat wird angenommen.
    cfg = EvalConfig(n_games=2, win_threshold=0.0, temperature_moves=4)
    res = gate(_net(seed=1), _net(seed=2), size=6, mcts_config=_SMALL_MCTS, eval_config=cfg, seed=0)
    assert res.accepted is True

    # win_threshold > 1.0 -> unmöglich zu erreichen, immer abgelehnt.
    cfg_hard = EvalConfig(n_games=2, win_threshold=1.01, temperature_moves=4)
    res_hard = gate(_net(seed=1), _net(seed=2), size=6, mcts_config=_SMALL_MCTS, eval_config=cfg_hard, seed=0)
    assert res_hard.accepted is False


def test_openings_vary_between_games():
    # Mit temperature_moves > 0 dürfen zwei Partien desselben Netz-Paars nicht
    # zwangsläufig identisch sein – sonst wäre die Evaluation wertlos.
    agent = make_agent(_net(seed=1), "a", _SMALL_MCTS, _SMALL_EVAL, seed=0)
    state1 = GameState.initial(6)
    first_moves = set()
    for _ in range(6):
        first_moves.add(agent.select_move(state1))
    assert len(first_moves) > 1


def test_evaluate_vs_baseline_runs():
    res = evaluate_vs_baseline(
        _net(seed=1), RandomAgent(seed=0), size=6,
        n_games=4, mcts_config=_SMALL_MCTS, eval_config=_SMALL_EVAL, seed=0,
    )
    assert res.games == 4
    assert 0.0 <= res.win_rate <= 1.0
