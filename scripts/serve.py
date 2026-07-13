"""Startet das lokale Web-Frontend zum Selberspielen gegen das trainierte Netz.

    python scripts/serve.py                       # best.pt, http://127.0.0.1:8000
    python scripts/serve.py --checkpoint checkpoints/iter_100.pt
    python scripts/serve.py --host 0.0.0.0 --port 8080

Aus dem Projekt-Root starten (dort liegt ``checkpoints/``). Danach im Browser
http://127.0.0.1:8000 öffnen.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description="Othello-Web-Frontend starten")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt",
                        help="Pfad zum Netz-Checkpoint (.pt)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"FEHLER: Checkpoint nicht gefunden: {Path(args.checkpoint).resolve()}")
        print("Tipp: aus dem Verzeichnis mit 'checkpoints/' starten oder "
              "--checkpoint mit vollem Pfad angeben.")
        return 1

    # Der Server liest den Pfad aus der Umgebung (er wird als Modul importiert).
    os.environ["OTHELLO_CHECKPOINT"] = str(args.checkpoint)

    import uvicorn  # spät importieren, damit --help ohne uvicorn funktioniert

    print(f"Lade Modell aus {args.checkpoint} …")
    print(f"Frontend läuft auf http://{args.host}:{args.port}  (Strg+C zum Beenden)")
    uvicorn.run("web.server:app", host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
