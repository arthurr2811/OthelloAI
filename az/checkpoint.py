"""Speichern und Laden von Netz-Checkpoints.

Ein Checkpoint enthält alles, um das Netz ohne Kenntnis der Config zu
rekonstruieren: Brettgröße, Architektur-Parameter und Gewichte. Optional lassen
sich Metadaten (z. B. Iterationsnummer) mitspeichern.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from az.net import OthelloNet
from config import NetConfig


def save_checkpoint(net: OthelloNet, path: str | Path, *, extra: dict | None = None) -> None:
    """Schreibt Netz-Gewichte + Architektur nach ``path`` (.pt)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "board_size": net.board_size,
            "net_config": asdict(net.config),
            "state_dict": net.state_dict(),
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(
    path: str | Path, device: str | torch.device | None = None
) -> tuple[OthelloNet, dict]:
    """Lädt ein Netz aus ``path``. Rückgabe: ``(net, extra)``.

    Das Netz wird auf ``device`` verschoben und in den Eval-Modus gesetzt.
    """
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    net = OthelloNet(ckpt["board_size"], NetConfig(**ckpt["net_config"]))
    net.load_state_dict(ckpt["state_dict"])
    net.to(device or "cpu")
    net.eval()
    return net, ckpt.get("extra", {})
