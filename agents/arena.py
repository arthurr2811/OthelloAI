"""Arena: lässt zwei Agenten gegeneinander spielen und sammelt Statistik.

Wird später auch fürs AlphaZero-Gating gebraucht (neues Modell vs. bestes).
"""

from __future__ import annotations

from dataclasses import dataclass

from othello.board import BLACK, EMPTY, WHITE, GameState

from .base import Agent


def play_game(black_agent: Agent, white_agent: Agent, size: int = 8) -> int:
    """Spielt eine Partie und gibt den Gewinner zurück (BLACK/WHITE/EMPTY).

    ``black_agent`` zieht als Schwarz (beginnt), ``white_agent`` als Weiß.
    """
    state = GameState.initial(size)
    # Obergrenze als Sicherheitsnetz gegen Endlosschleifen (kann nie greifen,
    # da Othello-Partien endlich sind – dient nur der Robustheit).
    max_plies = size * size * 4
    for _ in range(max_plies):
        if state.is_terminal():
            break
        agent = black_agent if state.current_player == BLACK else white_agent
        move = agent.select_move(state)
        state = state.apply(move)
    return state.winner()


@dataclass
class MatchResult:
    """Bilanz aus Sicht von Agent A (dem ersten übergebenen Agenten)."""

    agent_a: str
    agent_b: str
    wins: int = 0     # A hat gewonnen
    losses: int = 0   # B hat gewonnen
    draws: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        """Punktquote von A: Sieg = 1, Remis = 0.5, Niederlage = 0."""
        if self.games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games

    def summary(self) -> str:
        return (
            f"{self.agent_a} vs. {self.agent_b} über {self.games} Partien: "
            f"{self.wins}W / {self.losses}L / {self.draws}D "
            f"(Quote {self.agent_a}: {self.win_rate:.1%})"
        )


def play_match(
    agent_a: Agent, agent_b: Agent, n_games: int = 100, size: int = 8
) -> MatchResult:
    """Spielt ``n_games`` Partien; Startfarbe wird abwechselnd getauscht.

    Fairness: bei geraden Spielindizes ist A Schwarz (beginnt), bei ungeraden
    ist B Schwarz. So gleicht sich der Anzugsvorteil aus.
    """
    result = MatchResult(agent_a=str(agent_a), agent_b=str(agent_b))
    for i in range(n_games):
        a_is_black = (i % 2 == 0)
        if a_is_black:
            w = play_game(agent_a, agent_b, size)
            a_color = BLACK
        else:
            w = play_game(agent_b, agent_a, size)
            a_color = WHITE

        if w == EMPTY:
            result.draws += 1
        elif w == a_color:
            result.wins += 1
        else:
            result.losses += 1
    return result
