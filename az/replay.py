"""Replay-Buffer für Self-Play-Samples.

Ein ``deque`` mit fester Maximalgröße: neue Self-Play-Daten schieben die ältesten
heraus (gleitendes Fenster über die jüngsten Partien). Das Training zieht daraus
zufällige Batches.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Sample:
    """Ein Trainingsbeispiel aus dem Self-Play.

    Attribute:
        planes: kodierte Stellung ``(3, S, S)`` float32, aus Sicht des Ziehenden.
        policy: MCTS-Visit-Verteilung ``(S*S + 1,)`` float32 (Trainingsziel).
        value:  Partie-Ausgang in ``[-1, 1]`` aus Sicht des Ziehenden dieser Stellung.
    """

    planes: np.ndarray
    policy: np.ndarray
    value: float


class ReplayBuffer:
    """Ringpuffer fester Größe über :class:`Sample`-Objekte."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._buf: deque[Sample] = deque(maxlen=capacity)

    def add(self, samples) -> None:
        """Hängt ein einzelnes Sample oder ein Iterable von Samples an."""
        if isinstance(samples, Sample):
            self._buf.append(samples)
        else:
            self._buf.extend(samples)

    def __len__(self) -> int:
        return len(self._buf)

    def sample_batch(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Zieht ``batch_size`` Samples (mit Zurücklegen) als gestapelte Arrays.

        Rückgabe: ``(planes, policy, value)`` mit Shapes ``(B, 3, S, S)``,
        ``(B, S*S+1)`` und ``(B,)`` – direkt fürs Training verwendbar.
        """
        if not self._buf:
            raise ValueError("Replay-Buffer ist leer")
        idx = rng.integers(0, len(self._buf), size=batch_size)
        chosen = [self._buf[int(i)] for i in idx]
        planes = np.stack([s.planes for s in chosen]).astype(np.float32)
        policy = np.stack([s.policy for s in chosen]).astype(np.float32)
        value = np.array([s.value for s in chosen], dtype=np.float32)
        return planes, policy, value
