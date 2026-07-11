"""Self-Play: der Agent erzeugt Trainingsdaten gegen sich selbst.

Ablauf einer Partie:
    1. Pro Zug MCTS (mit Wurzel-Rauschen) laufen lassen.
    2. ``(planes, policy_target, spieler)`` speichern – ``policy_target`` ist die
       normalisierte Visit-Verteilung (Temperatur 1), das Trainingsziel der Policy.
    3. Zug samplen: die ersten ``temperature_moves`` Züge explorativ (Temperatur 1),
       danach deterministisch (der meistbesuchte Zug).
    4. Am Ende den Value-Target aus dem Ergebnis rückwärts in alle Samples füllen:
       ``+1`` wenn der Spieler dieser Stellung gewann, ``-1`` bei Niederlage,
       ``0`` bei Remis.

Reine Pass-Stellungen (nur ``[PASS]`` legal) werden übersprungen – sie tragen kein
Lernsignal und würden die Policy nur auf den Pass-Index verzerren.
"""

from __future__ import annotations

import numpy as np

from az.augment import symmetries
from az.encoding import encode_state, index_to_move
from az.mcts import NeuralMCTS
from az.replay import ReplayBuffer, Sample
from config import DEFAULT_SELFPLAY, SelfPlayConfig
from othello.board import EMPTY, PASS, GameState


def _sample_move(root, size: int, temperature: float, rng: np.random.Generator):
    """Wählt einen Zug aus den Wurzel-Visit-Counts gemäß Temperatur."""
    pi = NeuralMCTS._visit_distribution(root, temperature)
    if temperature == 0:
        index = int(np.argmax(pi))
    else:
        index = int(rng.choice(len(pi), p=pi))
    return index_to_move(index, size)


def play_game(
    mcts: NeuralMCTS,
    size: int,
    config: SelfPlayConfig = DEFAULT_SELFPLAY,
    rng: np.random.Generator | None = None,
) -> list[Sample]:
    """Spielt eine komplette Self-Play-Partie und gibt die Trainingssamples zurück.

    Die Samples sind **nicht** augmentiert – das übernimmt :func:`augment` bzw.
    :func:`generate_game`, damit der rohe Spielverlauf testbar bleibt.
    """
    rng = rng if rng is not None else np.random.default_rng()
    state = GameState.initial(size)
    history: list[tuple[np.ndarray, np.ndarray, int]] = []
    move_count = 0

    while not state.is_terminal():
        if move_count >= config.max_moves:
            raise RuntimeError("max_moves überschritten – vermutlich ein Engine-Bug")

        options = state.legal_moves()
        if options == [PASS]:            # erzwungenes Pass: kein Lernsignal
            state = state.apply(PASS)
            continue

        root = mcts.run(state)
        policy_target = NeuralMCTS._visit_distribution(root, temperature=1.0)
        history.append((encode_state(state), policy_target, state.current_player))

        temperature = 1.0 if move_count < config.temperature_moves else 0.0
        move = _sample_move(root, size, temperature, rng)
        state = state.apply(move)
        move_count += 1

    winner = state.winner()
    samples = []
    for planes, policy, player in history:
        if winner == EMPTY:
            value = 0.0
        else:
            value = 1.0 if winner == player else -1.0
        samples.append(Sample(planes=planes, policy=policy, value=value))
    return samples


def augment(samples: list[Sample], size: int) -> list[Sample]:
    """Verachtfacht die Samples über die 8 Dihedral-Symmetrien."""
    out: list[Sample] = []
    for s in samples:
        for planes, policy in symmetries(s.planes, s.policy, size):
            out.append(Sample(planes=planes, policy=policy, value=s.value))
    return out


def generate_game(
    mcts: NeuralMCTS,
    size: int,
    config: SelfPlayConfig = DEFAULT_SELFPLAY,
    rng: np.random.Generator | None = None,
    buffer: ReplayBuffer | None = None,
) -> list[Sample]:
    """Spielt eine Partie, augmentiert (falls konfiguriert) und füllt den Buffer.

    Gibt die (ggf. augmentierten) Samples zurück und hängt sie an ``buffer`` an,
    falls einer übergeben wurde.
    """
    samples = play_game(mcts, size, config, rng)
    if config.augment:
        samples = augment(samples, size)
    if buffer is not None:
        buffer.add(samples)
    return samples
