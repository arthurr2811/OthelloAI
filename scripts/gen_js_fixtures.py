"""Erzeugt Test-Fixtures für die JS-Engine aus der Python-Engine.

    python scripts/gen_js_fixtures.py

Schreibt zwei Dateien nach ``tests/js/fixtures/``:

* ``engine.json``: Stellungen aus Zufallspartien mit den Erwartungswerten der
  Python-Engine (legale Optionen, Flips und Folgebrett je Zug, Terminal, Gewinner,
  Netz-Encoding). Geprüft von ``tests/js/engine.test.mjs`` gegen
  ``web/static/engine/othello.js``.
* ``mcts.json``: Visit-Counts und Werte der Wurzelkinder nach einer Suche von
  ``NeuralMCTS`` (``az/mcts.py``) mit einem deterministischen Dummy-Evaluator
  statt des Netzes. Geprüft von ``tests/js/mcts.test.mjs`` gegen
  ``web/static/engine/mcts.js``, der denselben Dummy nachbaut.

Nur nach Änderungen an Spielregeln, Encoding oder Suche neu erzeugen.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from az.encoding import encode_state, legal_move_mask, move_to_index  # noqa: E402
from az.mcts import NeuralMCTS  # noqa: E402
from config import MCTSConfig, project_path  # noqa: E402
from othello.board import PASS, GameState, flips_for_move  # noqa: E402

FIXTURE_DIR = "tests/js/fixtures"

# (Brettgröße, Anzahl Partien). Kleine Bretter liefern viele Pässe und vorzeitige
# Spielenden – genau die Randfälle, die in 8x8-Zufallspartien selten sind.
GAMES = [(8, 12), (6, 15), (4, 40)]
ENCODING_EVERY = 7   # Encoding nur für jede n-te Stellung (hält die Datei klein)

# MCTS-Fixture: (Brettgröße, Ply-Nummern der Suchstellungen) und Sim-Budgets.
# Auf 4x4 erreicht die Suche terminale Knoten und Pässe im Baum.
MCTS_POSITIONS = [(8, [0, 9, 24, 41, 53]), (6, [4, 13, 22]), (4, [1, 4, 7])]
MCTS_SIMS = [1, 2, 7, 50, 300]

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


def _dummy_eval(state: GameState) -> tuple[np.ndarray, float]:
    """Deterministischer Netz-Ersatz – identisch nachgebaut in tests/js/mcts.test.mjs.

    Logits hängen von Aktionsindex und Steinzahlen ab (damit die Priors je
    Stellung variieren), Value ist die Steindifferenz aus Sicht des Ziehenden.
    Beides exakt in float64 darstellbar.
    """
    n = state.size * state.size
    own = int(np.count_nonzero(state.board == state.current_player))
    opp = int(np.count_nonzero(state.board == -state.current_player))
    logits = np.array([((a * 7 + own * 3 + opp) % 11) / 4 for a in range(n + 1)])
    return logits, (own - opp) / n


class _DummyMCTS(NeuralMCTS):
    """``NeuralMCTS`` mit Dummy statt Netz; Masking/Softmax bleiben die echten."""

    def _evaluate(self, state: GameState) -> tuple[np.ndarray, float]:
        logits, value = _dummy_eval(state)
        return self._masked_softmax(logits, legal_move_mask(state)), value


def _mcts_cases(rng: random.Random) -> list[dict]:
    cases: list[dict] = []
    for size, plies in MCTS_POSITIONS:
        state = GameState.initial(size)
        for ply in range(max(plies) + 1):
            if ply in plies:
                for sims in MCTS_SIMS:
                    mcts = _DummyMCTS(None, MCTSConfig(n_simulations=sims), device="cpu")
                    root = mcts.run(state)
                    best = NeuralMCTS.visit_distribution(root, 0.0).argmax()
                    cases.append({
                        "size": size,
                        "board": _board_str(state.board),
                        "player": state.current_player,
                        "sims": sims,
                        "root_n": root.N,
                        "children": [
                            {"move": move_to_index(m, size), "n": c.N, "w": c.W}
                            for m, c in root.children.items()
                        ],
                        "best": int(best),
                    })
            state = state.apply(rng.choice(state.legal_moves()))
            assert not state.is_terminal(), "Ply-Nummer hinter Partieende"
    return cases


def _write(name: str, data: dict) -> Path:
    out = project_path(FIXTURE_DIR) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    return out


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

    out = _write("engine.json", {"positions": positions})
    n_pass = sum(1 for p in positions if p["options"] == [p["size"] ** 2])
    n_term = sum(1 for p in positions if p["terminal"])
    print(f"{len(positions)} Stellungen ({n_pass} Pass, {n_term} terminal) -> {out} "
          f"({out.stat().st_size / 1e3:.0f} kB)")

    cases = _mcts_cases(random.Random(1))
    out = _write("mcts.json", {"cases": cases})
    print(f"{len(cases)} MCTS-Suchen -> {out} ({out.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
