"""Exportiert einen Netz-Checkpoint nach ONNX fürs Browser-Frontend.

    python scripts/export_onnx.py
    python scripts/export_onnx.py --checkpoint checkpoints/iter_100.pt

Das Frontend (``web/static/``) lädt ``model.onnx`` mit onnxruntime-web und
rechnet die KI komplett im Browser.

Zur Kontrolle werden Stellungen aus Zufallspartien durch PyTorch *und* das
exportierte Modell (onnxruntime) geschickt; bei zu großer Abweichung bricht das
Skript mit Fehler ab.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from az.checkpoint import load_checkpoint  # noqa: E402
from az.encoding import encode_state  # noqa: E402
from config import DEFAULT_CHECKPOINT, project_path  # noqa: E402
from othello.board import GameState  # noqa: E402

DEFAULT_OUTPUT = "web/static/model.onnx"
TOLERANCE = 1e-4


def _random_positions(size: int, n: int, seed: int) -> np.ndarray:
    """``n`` Stellungen aus Zufallspartien, kodiert als ``(n, 3, size, size)``."""
    rng = random.Random(seed)
    planes: list[np.ndarray] = []
    while len(planes) < n:
        state = GameState.initial(size)
        while not state.is_terminal() and len(planes) < n:
            planes.append(encode_state(state))
            state = state.apply(rng.choice(state.legal_moves()))
    return np.stack(planes)


def export(net: torch.nn.Module, size: int, output: Path) -> None:
    """torch.export-basierter Export (faltet BatchNorm in die Convs).

    Batch-Achse dynamisch, damit das Frontend später auch gebündelt auswerten
    kann. ``external_data=False``: Gewichte in *einer* Datei, die der Browser am
    Stück lädt.
    """
    # Dummy mit Batch 2: bei Batch 1 würde torch.export die Achse als Konstante
    # spezialisieren.
    dummy = torch.zeros(2, 3, size, size)
    batch = torch.export.Dim("batch", min=1, max=1024)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net,
        (dummy,),
        str(output),
        input_names=["x"],
        output_names=["policy", "value"],
        dynamic_shapes=({0: batch},),
        opset_version=18,
        dynamo=True,
        external_data=False,
        verbose=False,
    )


def verify(net: torch.nn.Module, size: int, output: Path, n: int) -> float:
    """Max. absolute Abweichung zwischen PyTorch und ONNX über ``n`` Stellungen."""
    import onnxruntime as ort

    x = _random_positions(size, n, seed=0)
    with torch.no_grad():
        t_policy, t_value = net(torch.from_numpy(x))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    o_policy, o_value = session.run(["policy", "value"], {"x": x})
    return max(
        float(np.abs(t_policy.numpy() - o_policy).max()),
        float(np.abs(t_value.numpy() - o_value).max()),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Checkpoint nach ONNX exportieren")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Pfad zum Netz-Checkpoint (.pt); relativ = ab Projektwurzel")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Ziel der .onnx-Datei; relativ = ab Projektwurzel")
    parser.add_argument("--n-check", type=int, default=200,
                        help="Anzahl Stellungen für den PyTorch-vs-ONNX-Vergleich")
    args = parser.parse_args()

    checkpoint = project_path(args.checkpoint)
    output = project_path(args.output)
    if not checkpoint.exists():
        print(f"FEHLER: Checkpoint nicht gefunden: {checkpoint}")
        return 1

    net, _ = load_checkpoint(checkpoint, "cpu")
    size = net.board_size
    export(net, size, output)
    print(f"Exportiert: {output}  ({output.stat().st_size / 1e6:.1f} MB, Brett {size}x{size})")

    diff = verify(net, size, output, args.n_check)
    print(f"Max. Abweichung PyTorch vs. ONNX über {args.n_check} Stellungen: {diff:.2e}")
    if diff > TOLERANCE:
        print(f"FEHLER: Abweichung über Toleranz {TOLERANCE:.0e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
