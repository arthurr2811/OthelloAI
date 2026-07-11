"""Trainingsschleife: Netz aus Replay-Buffer-Daten lernen.

Loss = Policy-Cross-Entropy + ``value_loss_weight`` * Value-MSE. Die Policy-CE ist
die *weiche* Variante gegen die MCTS-Visit-Verteilung als Ziel:
``-sum_a pi(a) * log softmax(logits)_a``. L2-Regularisierung läuft über den
Weight-Decay des Optimizers.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from az.net import OthelloNet
from az.replay import ReplayBuffer
from config import DEFAULT_TRAIN, TrainConfig


class Trainer:
    """Kapselt Netz, Optimizer und einen Trainingsschritt."""

    def __init__(
        self,
        net: OthelloNet,
        config: TrainConfig = DEFAULT_TRAIN,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        self.net = net
        self.config = config
        self.device = torch.device(device) if device is not None else next(net.parameters()).device
        self.net.to(self.device)
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

    def train_step(
        self, planes: np.ndarray, policy: np.ndarray, value: np.ndarray
    ) -> dict[str, float]:
        """Ein Gradientenschritt auf einem Batch. Rückgabe: Loss-Komponenten."""
        self.net.train()
        x = torch.as_tensor(planes, dtype=torch.float32, device=self.device)
        target_policy = torch.as_tensor(policy, dtype=torch.float32, device=self.device)
        target_value = torch.as_tensor(value, dtype=torch.float32, device=self.device)

        logits, pred_value = self.net(x)
        log_probs = F.log_softmax(logits, dim=1)
        policy_loss = -(target_policy * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(pred_value, target_value)
        total = policy_loss + self.config.value_loss_weight * value_loss

        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()

        return {
            "total": float(total.item()),
            "policy": float(policy_loss.item()),
            "value": float(value_loss.item()),
        }

    def train(
        self,
        buffer: ReplayBuffer,
        n_steps: int,
        rng: np.random.Generator,
        *,
        log_path: str | Path | None = None,
    ) -> list[dict[str, float]]:
        """Zieht ``n_steps`` Batches aus dem Buffer und trainiert darauf.

        Loggt pro Schritt die Loss-Komponenten optional als CSV nach ``log_path``.
        """
        history: list[dict[str, float]] = []
        writer = _CsvLogger(log_path) if log_path is not None else None
        for step in range(n_steps):
            batch = buffer.sample_batch(self.config.batch_size, rng)
            losses = self.train_step(*batch)
            history.append(losses)
            if writer is not None:
                writer.write({"step": step, **losses})
        if writer is not None:
            writer.close()
        return history


class _CsvLogger:
    """Minimaler CSV-Logger für Loss-Kurven (Header aus dem ersten Row)."""

    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer: csv.DictWriter | None = None

    def write(self, row: dict) -> None:
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)

    def close(self) -> None:
        self._file.close()
