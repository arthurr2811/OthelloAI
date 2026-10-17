"""Startet das Web-Frontend lokal zum Selberspielen gegen das trainierte Netz.

    python scripts/serve.py
    python scripts/serve.py --host 0.0.0.0 --port 8080

Im Browser http://127.0.0.1:8000 öffnen.

Die KI läuft komplett im Browser (onnxruntime-web + MCTS in JS, siehe
``web/static/engine/``) – dieses Skript liefert nur die statischen Dateien aus,
genau wie später GitHub Pages. Das Netz kommt aus ``web/static/model.onnx``;
nach einem neuen Training erst ``python scripts/export_onnx.py`` ausführen.
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "static"


class _Handler(SimpleHTTPRequestHandler):
    # Explizit setzen: unter Windows kann die Registry .js sonst als text/plain
    # melden, und der Browser verweigert dann die ES-Module.
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".wasm": "application/wasm",
        ".onnx": "application/octet-stream",
    }

    def end_headers(self) -> None:
        # Lokal immer frisch laden, damit Änderungen am Frontend sofort greifen.
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:
        pass   # kein Log pro Request


def main() -> int:
    parser = argparse.ArgumentParser(description="Othello-Web-Frontend lokal starten")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if not (STATIC_DIR / "model.onnx").exists():
        print(f"FEHLER: {STATIC_DIR / 'model.onnx'} fehlt.")
        print("Tipp: python scripts/export_onnx.py ausführen.")
        return 1

    server = ThreadingHTTPServer((args.host, args.port), partial(_Handler, directory=str(STATIC_DIR)))
    print(f"Frontend läuft auf http://{args.host}:{args.port}  (Strg+C zum Beenden)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
