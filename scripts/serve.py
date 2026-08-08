"""Startet das lokale Web-Frontend zum Selberspielen gegen das trainierte Netz.

    python scripts/serve.py                       # models/best.pt, http://127.0.0.1:8000
    python scripts/serve.py --checkpoint checkpoints/iter_100.pt
    python scripts/serve.py --host 0.0.0.0 --port 8080

Relative Checkpoint-Pfade gelten ab der Projektwurzel – der Aufrufort ist also
egal. Danach im Browser http://127.0.0.1:8000 öffnen.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_CHECKPOINT, project_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Othello-Web-Frontend starten")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Pfad zum Netz-Checkpoint (.pt); relativ = ab Projektwurzel")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    checkpoint = project_path(args.checkpoint)
    if not checkpoint.exists():
        print(f"FEHLER: Checkpoint nicht gefunden: {checkpoint}")
        print("Tipp: --checkpoint mit einem existierenden Pfad angeben "
              "(relativ zur Projektwurzel oder absolut).")
        return 1

    # Der Server liest den Pfad aus der Umgebung (er wird als Modul importiert).
    os.environ["OTHELLO_CHECKPOINT"] = str(checkpoint)

    import uvicorn  # spät importieren, damit --help ohne uvicorn funktioniert

    print(f"Lade Modell aus {checkpoint} …")
    print(f"Frontend läuft auf http://{args.host}:{args.port}  (Strg+C zum Beenden)")
    uvicorn.run("web.server:app", host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
