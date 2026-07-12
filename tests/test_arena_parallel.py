"""Tests für die gebündelte, parallele Arena (az/arena_parallel.py).

Geprüft wird: korrekte Bilanz (Summe = n_games, Sicht Spieler A), Netz-vs-Netz
und Netz-vs-einfacher-Agent, sowie Konsistenz mit dem sequenziellen Match bei
deterministischem Spiel (Temperatur 0 -> gleiche Partien -> gleiche Bilanz).
"""

from __future__ import annotations

import pytest
import torch

from az.arena_parallel import AgentPlayer, NetPlayer, play_match_parallel
from az.net import OthelloNet
from config import MCTSConfig, NetConfig

from agents.arena import play_match
from agents.simple import RandomAgent
from az.mcts import NeuralMCTSAgent

_TINY_NET = NetConfig(channels=16, n_res_blocks=2, value_hidden=16)


def _net(size=6, seed=0):
    torch.manual_seed(seed)
    return OthelloNet(board_size=size, config=_TINY_NET).eval()


_MCTS = MCTSConfig(n_simulations=8)


def test_net_vs_net_balance_sums_to_games():
    a = NetPlayer(_net(seed=1), "a", _MCTS, temperature=1.0, temperature_moves=4, seed=0)
    b = NetPlayer(_net(seed=2), "b", _MCTS, temperature=1.0, temperature_moves=4, seed=1)
    res = play_match_parallel(a, b, n_games=6, size=6)
    assert res.games == 6
    assert res.wins + res.losses + res.draws == 6
    assert 0.0 <= res.win_rate <= 1.0
    assert res.agent_a == "a" and res.agent_b == "b"


def test_net_vs_agent_runs():
    a = NetPlayer(_net(seed=1), "net", _MCTS, temperature=1.0, temperature_moves=4, seed=0)
    res = play_match_parallel(a, AgentPlayer(RandomAgent(seed=0)), n_games=6, size=6)
    assert res.games == 6
    assert 0.0 <= res.win_rate <= 1.0


def test_more_games_than_workers():
    # n_parallel kleiner als n_games: Worker müssen mehrere Partien nacheinander spielen.
    a = NetPlayer(_net(seed=1), "a", _MCTS, temperature=1.0, temperature_moves=4, seed=0)
    b = NetPlayer(_net(seed=2), "b", _MCTS, temperature=1.0, temperature_moves=4, seed=1)
    res = play_match_parallel(a, b, n_games=7, size=6, n_parallel=3)
    assert res.games == 7


def test_deterministic_matches_sequential():
    # Temperatur 0 (deterministisch): parallele und sequenzielle Arena müssen
    # dieselbe Bilanz liefern – gleiche Partien, nur andere Verzahnung.
    netA, netB = _net(seed=1), _net(seed=2)

    a_par = NetPlayer(netA, "a", _MCTS, temperature=0.0, temperature_moves=0, seed=0)
    b_par = NetPlayer(netB, "b", _MCTS, temperature=0.0, temperature_moves=0, seed=1)
    par = play_match_parallel(a_par, b_par, n_games=6, size=6)

    a_seq = NeuralMCTSAgent(netA, _MCTS, temperature=0.0, temperature_moves=0, seed=0)
    a_seq.name = "a"
    b_seq = NeuralMCTSAgent(netB, _MCTS, temperature=0.0, temperature_moves=0, seed=1)
    b_seq.name = "b"
    seq = play_match(a_seq, b_seq, n_games=6, size=6)

    assert (par.wins, par.losses, par.draws) == (seq.wins, seq.losses, seq.draws)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="keine CUDA-GPU")
def test_runs_on_gpu():
    a = NetPlayer(_net(seed=1).cuda(), "a", _MCTS, temperature=1.0, temperature_moves=4, seed=0)
    b = NetPlayer(_net(seed=2).cuda(), "b", _MCTS, temperature=1.0, temperature_moves=4, seed=1)
    res = play_match_parallel(a, b, n_games=4, size=6)
    assert res.games == 4
