"""Tests für Schritt 2.4: Trainingsschleife, Loss, Checkpoints."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from az.checkpoint import load_checkpoint, save_checkpoint
from az.encoding import num_actions
from az.net import OthelloNet
from az.replay import ReplayBuffer, Sample
from az.train import Trainer
from config import TrainConfig


def _net(size=6):
    torch.manual_seed(0)
    return OthelloNet(board_size=size)


def _fixed_batch(size=6, n=16, seed=0):
    """Ein kleiner, fester Datensatz mit *scharfen* Targets (für Overfit-Test).

    Die Policy-Targets sind one-hot, damit der erreichbare Policy-Loss gegen 0
    geht (bei nahezu gleichverteilten Targets läge der Boden bei ihrer Entropie
    ~log(A) und der Overfit-Beweis wäre unmöglich).
    """
    rng = np.random.default_rng(seed)
    samples = []
    a = num_actions(size)
    for _ in range(n):
        planes = rng.standard_normal((3, size, size)).astype(np.float32)
        policy = np.zeros(a, dtype=np.float32)
        policy[rng.integers(a)] = 1.0                       # one-hot
        value = float(rng.choice([-1.0, 0.0, 1.0]))
        samples.append(Sample(planes, policy, value))
    return samples


def test_train_step_returns_finite_losses():
    trainer = Trainer(_net(), TrainConfig(batch_size=8))
    buf = ReplayBuffer(100)
    buf.add(_fixed_batch())
    losses = trainer.train_step(*buf.sample_batch(8, np.random.default_rng(0)))
    assert set(losses) == {"total", "policy", "value"}
    assert all(np.isfinite(v) for v in losses.values())
    assert losses["value"] >= 0.0


def test_overfits_fixed_batch():
    # Korrektheitsbeweis: auf einem festen Batch muss der Loss deutlich sinken.
    size = 6
    trainer = Trainer(_net(size), TrainConfig(lr=1e-3, batch_size=16, weight_decay=0.0))
    buf = ReplayBuffer(100)
    buf.add(_fixed_batch(size, n=16))
    rng = np.random.default_rng(0)
    history = trainer.train(buf, n_steps=300, rng=rng)
    first = np.mean([h["total"] for h in history[:5]])
    last = np.mean([h["total"] for h in history[-5:]])
    assert last < first * 0.3, f"Loss sank nicht genug: {first:.3f} -> {last:.3f}"
    assert last < 0.5, f"Overfit erreichte keinen kleinen Loss: {last:.3f}"


def test_train_writes_csv_log(tmp_path):
    trainer = Trainer(_net(), TrainConfig(batch_size=8))
    buf = ReplayBuffer(100)
    buf.add(_fixed_batch())
    log = tmp_path / "loss.csv"
    trainer.train(buf, n_steps=5, rng=np.random.default_rng(0), log_path=log)
    lines = log.read_text().strip().splitlines()
    assert lines[0].split(",") == ["step", "total", "policy", "value"]
    assert len(lines) == 6                              # Header + 5 Schritte


def test_checkpoint_roundtrip(tmp_path):
    net = _net(size=6)
    x = torch.randn(2, 3, 6, 6)
    net.eval()
    with torch.no_grad():
        p0, v0 = net(x)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(net, path, extra={"iteration": 7})
    loaded, extra = load_checkpoint(path)

    assert extra["iteration"] == 7
    assert loaded.board_size == 6
    with torch.no_grad():
        p1, v1 = loaded(x)
    assert torch.allclose(p0, p1, atol=1e-6)
    assert torch.allclose(v0, v1, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="keine CUDA-GPU")
def test_train_step_on_gpu():
    net = OthelloNet(board_size=6).cuda()
    trainer = Trainer(net, TrainConfig(batch_size=8), device="cuda")
    buf = ReplayBuffer(100)
    buf.add(_fixed_batch())
    losses = trainer.train_step(*buf.sample_batch(8, np.random.default_rng(0)))
    assert np.isfinite(losses["total"])
