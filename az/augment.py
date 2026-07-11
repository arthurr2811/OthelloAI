"""Dihedral-Symmetrien für Trainingssamples.

Othello ist unter allen 8 Symmetrien des Quadrats äquivariant: Die Flip-Regel
behandelt alle Richtungen gleich, also hat eine gedrehte/gespiegelte Stellung
exakt die gedrehte/gespiegelte Policy und denselben Value. Jedes Self-Play-Sample
lässt sich damit fast gratis verachtfachen (großer Sample-Efficiency-Gewinn).

Die 8 Transformationen: Rotationen um 0/90/180/270 Grad, jeweils optional mit
einer Spiegelung an der Spaltenachse. Board-Ebenen ``(C, S, S)`` und das
Policy-Gitter ``(S, S)`` werden mit **derselben** geometrischen Operation
transformiert; der Pass-Eintrag (letzter Policy-Index) bleibt unverändert.
"""

from __future__ import annotations

import numpy as np

# Die 8 Elemente der Dieder-Gruppe D4 als (Rotationen k, spiegeln?).
_SYMMETRIES: tuple[tuple[int, bool], ...] = (
    (0, False), (1, False), (2, False), (3, False),
    (0, True),  (1, True),  (2, True),  (3, True),
)


def transform_planes(planes: np.ndarray, k: int, flip: bool) -> np.ndarray:
    """Wendet Rotation (``k`` * 90 Grad) und optionale Spaltenspiegelung an.

    ``planes`` hat Form ``(C, S, S)``; rotiert/gespiegelt wird nur über die
    beiden Ortsachsen (1, 2).
    """
    out = np.rot90(planes, k, axes=(1, 2))
    if flip:
        out = out[:, :, ::-1]
    return np.ascontiguousarray(out)


def transform_policy(policy: np.ndarray, size: int, k: int, flip: bool) -> np.ndarray:
    """Transformiert einen Policy-Vektor ``(S*S + 1,)`` konsistent zu den Ebenen.

    Die Feld-Wahrscheinlichkeiten werden als ``(S, S)``-Gitter behandelt und
    identisch transformiert; der Pass-Eintrag am Ende bleibt an Ort und Stelle.
    """
    grid = policy[: size * size].reshape(size, size)
    grid = np.rot90(grid, k)
    if flip:
        grid = grid[:, ::-1]
    return np.concatenate([grid.reshape(-1), policy[size * size:]])


def symmetries(
    planes: np.ndarray, policy: np.ndarray, size: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Alle 8 Symmetrie-Varianten von ``(planes, policy)``.

    Der Value bleibt invariant und wird hier nicht angefasst (Aufrufer hängt ihn
    unverändert an jede Variante).
    """
    out = []
    for k, flip in _SYMMETRIES:
        out.append((transform_planes(planes, k, flip), transform_policy(policy, size, k, flip)))
    return out
