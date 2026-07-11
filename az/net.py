"""AlphaZero-Netz für Othello: gemeinsamer Torso, zwei Köpfe.

Architektur (klein gehalten, siehe :class:`config.NetConfig`):
    Eingabe (B, 3, S, S)
      -> Conv-Stamm (3x3) -> BN -> ReLU
      -> N Residual-Blöcke (je 2x Conv-BN mit Skip-Verbindung)
      -> Policy-Head:  1x1-Conv -> flach -> Linear -> Logits (S*S + 1)
      -> Value-Head:   1x1-Conv -> flach -> Linear -> ReLU -> Linear -> tanh

Konventionen:
    * Die Policy liefert **Logits** (kein Softmax) über alle Felder plus Pass –
      Masking und Normalisierung passieren beim Aufrufer.
    * Der Value liegt in ``[-1, 1]`` und ist die erwartete Partie-Bewertung aus
      Sicht des Spielers am Zug (+1 = sicherer Sieg, -1 = sichere Niederlage).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DEFAULT_NET, NetConfig


class _ResidualBlock(nn.Module):
    """Zwei 3x3-Convs mit BatchNorm und einer Skip-Verbindung."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class OthelloNet(nn.Module):
    """Policy-Value-Netz. Brettgröße wird beim Anlegen festgelegt.

    Die Kopf-Dimensionen hängen von ``board_size`` ab, deshalb muss das Netz zur
    Brettgröße passen (6x6-Netz != 8x8-Netz).
    """

    def __init__(self, board_size: int, config: NetConfig = DEFAULT_NET) -> None:
        super().__init__()
        self.board_size = board_size
        self.config = config
        n_actions = board_size * board_size + 1

        # --- Gemeinsamer Torso ---
        self.stem = nn.Sequential(
            nn.Conv2d(config.input_planes, config.channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(config.channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.ModuleList(
            _ResidualBlock(config.channels) for _ in range(config.n_res_blocks)
        )

        # --- Policy-Head ---
        self.policy_conv = nn.Conv2d(config.channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, n_actions)

        # --- Value-Head ---
        self.value_conv = nn.Conv2d(config.channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, config.value_hidden)
        self.value_fc2 = nn.Linear(config.value_hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Vorwärtslauf.

        Args:
            x: Eingabe-Batch ``(B, input_planes, S, S)``.

        Returns:
            ``(policy_logits, value)`` mit Shapes ``(B, S*S+1)`` und ``(B,)``.
            Policy sind rohe Logits, Value liegt dank tanh in ``[-1, 1]``.
        """
        out = self.stem(x)
        for block in self.res_blocks:
            out = block(out)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.flatten(start_dim=1)
        policy_logits = self.policy_fc(p)

        # Value
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.flatten(start_dim=1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)

        return policy_logits, value
