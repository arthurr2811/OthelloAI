"""Erzeugt Test-Fixtures für die JS-Engine aus der Python-Engine.

    python scripts/gen_js_fixtures.py

Schreibt ``tests/js/fixtures/engine.json``: Stellungen aus Zufallspartien mit
den Erwartungswerten der Python-Engine (legale Optionen, Flips und Folgebrett je
Zug, Terminal, Gewinner, Netz-Encoding). ``tests/js/engine.test.mjs`` prüft die
JS-Engine (``web/static/engine/othello.js``) dagegen.

Nur nach Änderungen an den Spielregeln/am Encoding neu erzeugen.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from az.encoding import encode_state  # noqa: E402
from config import project_path  # noqa: E402
from othello.board import PASS, GameState, flips_for_move  # noqa: E402

OUTPUT = "tests/js/fixtures/engine.json"

# (Brettgröße, Anzahl Partien). Kleine Bretter liefern viele Pässe und vorzeitige
# Spielenden – genau die Randfälle, die in 8x8-Zufallspartien selten sind.
GAMES = [(8, 12), (6, 15), (4, 40)]
ENCODING_EVERY = 7   # Encoding nur für jede n-te Stellung (hält die Datei klein)

_SYMBOL = {1: "X", -1: "O", 0: "."}


def _board_str(board) -> str:
    return "".join(_SYMBOL[int(v)] for v in board.flatten())


def _index(move, size: int) -> int:
    """Zug -> JS-Index (Pass = size*size, wie der Pass-Index der Policy)."""
    if move == PASS:
        return size * size
    r, c = move
    return r * size + c


def _position(state: GameState, with_encoding: bool) -> dict:
    size = state.size
    options = state.legal_moves()
    moves = []
    for move in options:
        if move == PASS:
            continue
        flips = flips_for_move(state.board, state.current_player, move)
        moves.append({
            "move": _index(move, size),
            "flips": [r * size + c for r, c in flips],
            "next": _board_str(state.apply(move).board),
        })
    pos = {
        "size": size,
        "board": _board_str(state.board),
        "player": state.current_player,
        "options": [_index(m, size) for m in options],
        "moves": moves,
        "terminal": state.is_terminal(),
        "winner": state.winner(),
    }
    if with_encoding:
        pos["encoding"] = "".join(str(int(v)) for v in encode_state(state).flatten())
    return pos


def main() -> int:
    rng = random.Random(0)
    positions: list[dict] = []
    for size, n_games in GAMES:
        for _ in range(n_games):
            state = GameState.initial(size)
            while True:
                positions.append(_position(state, len(positions) % ENCODING_EVERY == 0))
                if state.is_terminal():
                    break
                state = state.apply(rng.choice(state.legal_moves()))

    out = project_path(OUTPUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"positions": positions}, separators=(",", ":")), encoding="utf-8")

    n_pass = sum(1 for p in positions if p["options"] == [p["size"] ** 2])
    n_term = sum(1 for p in positions if p["terminal"])
    print(f"{len(positions)} Stellungen ({n_pass} Pass, {n_term} terminal) -> {out} "
          f"({out.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
