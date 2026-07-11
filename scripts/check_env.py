"""Prüft die Umgebung: PyTorch-Version + CUDA-Verfügbarkeit + GPU-Name.

Aufruf:
    python scripts/check_env.py
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch ist nicht installiert. Siehe README (cu128-Wheel).")
        return 1

    print(f"PyTorch:      {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA verfügbar: {cuda_ok}")

    if cuda_ok:
        print(f"CUDA-Version:  {torch.version.cuda}")
        print(f"GPU:          {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"Compute-Cap.: sm_{cap[0]}{cap[1]}")
    else:
        print("WARNUNG: Keine GPU erkannt – Training läuft nur auf CPU.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
