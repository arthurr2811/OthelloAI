"""Tests für Schritt 2.2: PUCT-MCTS mit (untrainiertem) Netz.

Das Netz ist zufällig initialisiert – hier geht es nur um Korrektheit der Suche
(gültige Züge, saubere Visit-Verteilung, Masking, Rauschen, Temperatur), nicht
um Spielstärke.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from az.encoding import index_to_move, move_to_index, num_actions
from az.mcts import NeuralMCTS, NeuralMCTSAgent, _Node
from az.net import OthelloNet
from config import MCTSConfig, NetConfig
from othello.board import PASS, GameState

# Mini-Netz: hier geht es um die Suche, nicht um Kapazität – hält die Suite schnell.
_TINY_NET = NetConfig(channels=16, n_res_blocks=2, value_hidden=16)


def _net(size=8):
    torch.manual_seed(0)
    return OthelloNet(board_size=size, config=_TINY_NET).eval()


def test_run_produces_visit_counts():
    state = GameState.initial(size=8)
    mcts = NeuralMCTS(_net(), MCTSConfig(n_simulations=50), seed=1)
    root = mcts.run(state)
    total_child_visits = sum(c.N for c in root.children.values())
    # Wurzel wurde einmal ausgewertet + 49 weitere Simulationen laufen in Kinder.
    assert root.N == 50
    assert total_child_visits == 49
    # Nur legale Züge tauchen als Kinder auf.
    legal = set(state.legal_moves())
    assert set(root.children.keys()) == legal


def test_policy_target_is_distribution_over_legal_moves():
    state = GameState.initial(size=8)
    mcts = NeuralMCTS(_net(), MCTSConfig(n_simulations=40, temperature=1.0), seed=2)
    pi = mcts.policy_target(state)
    assert pi.shape == (num_actions(8),)
    assert pytest.approx(pi.sum(), abs=1e-6) == 1.0
    # Masse liegt ausschließlich auf legalen Zügen.
    legal_idx = {move_to_index(m, 8) for m in state.legal_moves()}
    assert set(np.flatnonzero(pi > 0)).issubset(legal_idx)


def test_temperature_zero_is_argmax_visits():
    state = GameState.initial(size=8)
    mcts = NeuralMCTS(_net(), MCTSConfig(n_simulations=60), seed=3)
    root = mcts.run(state)
    pi = mcts._visit_distribution(root, temperature=0.0)
    assert pi.sum() == pytest.approx(1.0)
    assert np.count_nonzero(pi) == 1
    best_move = max(root.children.items(), key=lambda kv: kv[1].N)[0]
    assert index_to_move(int(np.argmax(pi)), 8) == best_move


def test_select_move_returns_legal_move():
    state = GameState.initial(size=8)
    mcts = NeuralMCTS(_net(), MCTSConfig(n_simulations=30), seed=4)
    move = mcts.select_move(state, temperature=0.0)
    assert move in state.legal_moves()


def test_dirichlet_noise_changes_root_priors():
    state = GameState.initial(size=8)
    cfg = MCTSConfig(n_simulations=30)
    net = _net()
    # Gleiche Wurzel-Expansion, einmal mit und ohne Rauschen.
    clean = NeuralMCTS(net, cfg, add_noise=False, seed=5).run(state)
    noisy = NeuralMCTS(net, cfg, add_noise=True, seed=5).run(state)
    clean_priors = np.array([clean.children[m].prior for m in sorted(clean.children)])
    noisy_priors = np.array([noisy.children[m].prior for m in sorted(noisy.children)])
    assert not np.allclose(clean_priors, noisy_priors)
    # Rauschen ist eine Mischung -> immer noch eine gültige Verteilung.
    assert noisy_priors.sum() == pytest.approx(1.0, abs=1e-6)


def test_forced_pass_is_returned_without_search():
    # Suche per Random-Spiel eine echte Stellung, in der nur PASS legal ist, und
    # prüfe, dass select_move sie ohne Suche direkt zurückgibt.
    rng = np.random.default_rng(0)
    mcts = NeuralMCTS(_net(size=6), MCTSConfig(n_simulations=5), seed=6)
    for _ in range(200):
        state = GameState.initial(size=6)
        while not state.is_terminal():
            options = state.legal_moves()
            if options == [PASS]:
                assert mcts.select_move(state) == PASS
                return
            move = options[int(rng.integers(len(options)))]
            state = state.apply(move)
    pytest.skip("keine erzwungene Pass-Stellung gefunden")


def test_works_on_6x6():
    state = GameState.initial(size=6)
    mcts = NeuralMCTS(_net(size=6), MCTSConfig(n_simulations=40), seed=7)
    root = mcts.run(state)
    assert root.N == 40
    pi = mcts._visit_distribution(root, temperature=1.0)
    assert pi.shape == (num_actions(6),)
    assert pi.sum() == pytest.approx(1.0)


def test_agent_interface_plays_full_game():
    # Zwei Netz-Agenten spielen eine komplette Partie bis zum Ende ohne Fehler.
    agent = NeuralMCTSAgent(_net(size=6), MCTSConfig(n_simulations=15), temperature=0.0, seed=8)
    state = GameState.initial(size=6)
    moves = 0
    while not state.is_terminal():
        move = agent.select_move(state)
        assert move in state.legal_moves()
        state = state.apply(move)
        moves += 1
        assert moves < 100  # Endlosschutz
    assert state.is_terminal()


def test_backprop_sign_convention():
    # Zwei-Knoten-Pfad: Wurzel (mover=None) -> Kind (mover=Schwarz). Ein Value von
    # +1 aus Sicht des Blatt-Spielers (= Weiß, weil Schwarz hineinzog) muss dem
    # Kind aus Schwarz-Sicht -1 gutschreiben.
    mcts = NeuralMCTS(_net(size=6), MCTSConfig(n_simulations=1), seed=0)
    root = _Node(GameState.initial(size=6), parent=None, mover=None, prior=1.0)
    child = _Node(root.state.apply(root.state.legal_moves()[0]), parent=root, mover=1, prior=1.0)
    root.children[0] = child
    # value ist aus Sicht von child.state.current_player (Weiß) = +1 (Weiß gewinnt).
    mcts._backprop(child, value=1.0)
    assert child.N == 1 and root.N == 1
    assert child.W == -1.0            # aus Schwarz-Sicht (mover) eine Niederlage
    assert root.W == 0.0             # Wurzel hat keinen mover


@pytest.mark.skipif(not torch.cuda.is_available(), reason="keine CUDA-GPU")
def test_runs_on_gpu():
    net = OthelloNet(board_size=8, config=_TINY_NET).eval().cuda()
    mcts = NeuralMCTS(net, MCTSConfig(n_simulations=20), seed=9)
    root = mcts.run(GameState.initial(size=8))
    assert root.N == 20
