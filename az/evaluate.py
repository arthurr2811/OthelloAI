"""Evaluation & Gating: nur echte Verbesserungen werden Bestmodell.

Zwei Messungen:
    * **Gating** (relativ): Kandidat vs. aktuelles Bestmodell über N Partien.
      Übersteigt die Punktquote die Schwelle, wird der Kandidat neues Bestmodell.
    * **Baseline** (absolut): Netz vs. Greedy/MCTS → Stärke-Kurve über die
      Iterationen, unabhängig davon, wie stark das jeweilige Bestmodell ist.

Für Partievielfalt spielen die Netz-Agenten die Eröffnung mit Temperatur 1
(``temperature_moves``), danach deterministisch – sonst wäre jede Partie zwischen
zwei festen Netzen identisch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from az.arena_parallel import AgentPlayer, NetPlayer, play_match_parallel
from az.mcts import NeuralMCTSAgent
from az.net import OthelloNet
from config import DEFAULT_EVAL, DEFAULT_MCTS, EvalConfig, MCTSConfig
from othello.board import GameState  # noqa: F401  (dokumentiert die Spielabhängigkeit)

from agents.arena import MatchResult
from agents.base import Agent


def make_agent(
    net: OthelloNet,
    name: str,
    mcts_config: MCTSConfig = DEFAULT_MCTS,
    eval_config: EvalConfig = DEFAULT_EVAL,
    *,
    device: str | torch.device | None = None,
    seed: int | None = None,
) -> NeuralMCTSAgent:
    """Baut einen Evaluations-Agenten (explorative Eröffnung, sonst deterministisch)."""
    agent = NeuralMCTSAgent(
        net,
        mcts_config,
        device=device,
        temperature=1.0,
        temperature_moves=eval_config.temperature_moves,
        seed=seed,
    )
    agent.name = name
    return agent


@dataclass
class GatingResult:
    """Ergebnis eines Gating-Matches Kandidat vs. Bestmodell."""

    result: MatchResult
    win_rate: float
    accepted: bool

    def summary(self) -> str:
        verdict = "ANGENOMMEN" if self.accepted else "abgelehnt"
        return f"{self.result.summary()} -> {verdict}"


def gate(
    candidate: OthelloNet,
    best: OthelloNet,
    size: int,
    *,
    mcts_config: MCTSConfig = DEFAULT_MCTS,
    eval_config: EvalConfig = DEFAULT_EVAL,
    device: str | torch.device | None = None,
    seed: int = 0,
) -> GatingResult:
    """Kandidat vs. Bestmodell. Angenommen, wenn Quote >= ``win_threshold``."""
    cand_player = NetPlayer(
        candidate, "candidate", mcts_config, temperature=1.0,
        temperature_moves=eval_config.temperature_moves, device=device, seed=seed,
    )
    best_player = NetPlayer(
        best, "best", mcts_config, temperature=1.0,
        temperature_moves=eval_config.temperature_moves, device=device, seed=seed + 1,
    )
    result = play_match_parallel(cand_player, best_player, eval_config.n_games, size, device=device)
    accepted = result.win_rate >= eval_config.win_threshold
    return GatingResult(result=result, win_rate=result.win_rate, accepted=accepted)


def evaluate_vs_baseline(
    net: OthelloNet,
    baseline: Agent,
    size: int,
    *,
    n_games: int = 40,
    mcts_config: MCTSConfig = DEFAULT_MCTS,
    eval_config: EvalConfig = DEFAULT_EVAL,
    device: str | torch.device | None = None,
    seed: int = 0,
) -> MatchResult:
    """Misst die absolute Stärke des Netzes gegen einen Baseline-Agenten."""
    net_player = NetPlayer(
        net, "net", mcts_config, temperature=1.0,
        temperature_moves=eval_config.temperature_moves, device=device, seed=seed,
    )
    return play_match_parallel(net_player, AgentPlayer(baseline), n_games, size, device=device)
