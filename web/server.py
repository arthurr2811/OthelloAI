"""FastAPI-Inferenz-Backend zum Selberspielen gegen das trainierte Netz.

Lädt einen Checkpoint (Default: ``checkpoints/best.pt``) und stellt eine kleine
JSON-API bereit, die ein Browser-Frontend (``web/static/``) bedient:

    POST /api/new_game   -> neue Partie (Farbe + Schwierigkeit wählbar)
    POST /api/move       -> Menschzug anwenden, KI antwortet, neuer Zustand
    GET  /api/state      -> aktuellen Zustand einer Partie abfragen

Die **Stärke** steuern zwei Regler: ``n_simulations`` (MCTS-Denkbudget pro Zug)
und ``temperature`` (Zug-Zufall), siehe :data:`PRESETS` für die Vorbelegung.
Partien liegen im Prozess-Speicher (dict, per ``game_id``) – reicht fürs lokale
Einzelspiel.

Start:  ``python scripts/serve.py``  (oder ``uvicorn web.server:app``)
"""

from __future__ import annotations

import os
import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from az.checkpoint import load_checkpoint  # noqa: E402
from az.encoding import encode_state  # noqa: E402
from az.mcts import NeuralMCTS  # noqa: E402
from config import MCTSConfig  # noqa: E402
from othello.board import (  # noqa: E402
    BLACK, PASS, WHITE, GameState, disc_counts, flips_for_move,
)


# --- Stärke-Parameter -----------------------------------------------------
# Die KI-Stärke wird über zwei Regler gesteuert (Frontend), die 1:1 an das MCTS
# durchgereicht werden:
#
#   n_simulations : Denkbudget pro Zug. Mehr Simulationen = tiefere Suche =
#                   stärker. 1 ≈ reiner Policy-Zufall, mehrere hundert = ernst.
#   temperature   : Zug-Zufall. 0 = immer der (laut Suche) beste Zug =
#                   maximale Stärke. > 0 sampelt proportional zu den Visit-Counts
#                   (N^(1/T)); höhere Werte flachen die Verteilung ab und lassen
#                   die KI öfter patzen – so wird sie für Menschen schlagbar. Die
#                   Temperatur wirkt über die *ganze* Partie, nicht nur die
#                   Eröffnung.
#
# Presets belegen nur die Regler vor; gespielt wird mit den tatsächlichen
# Slider-Werten.
SIM_MIN, SIM_MAX = 1, 800
TEMP_MIN, TEMP_MAX = 0.0, 3.0

PRESETS: dict[str, dict] = {
    "easy":   {"label": "Einfach", "n_simulations": 10,  "temperature": 1.5},
    "medium": {"label": "Mittel",  "n_simulations": 80,  "temperature": 0.4},
    "hard":   {"label": "Schwer",  "n_simulations": 250, "temperature": 0.0},
}
DEFAULT_PRESET = "medium"

CHECKPOINT_PATH = os.environ.get("OTHELLO_CHECKPOINT", "checkpoints/best.pt")


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# --- KI-Engine (Netz + MCTS je Schwierigkeit, prozessweit geteilt) --------
class AIEngine:
    """Hält das Netz und je Schwierigkeit ein ``NeuralMCTS``.

    Der Zugriff wird per Lock serialisiert – die Endpoints laufen im FastAPI-
    Threadpool, aber Netz/Suche sollen nicht parallel dieselben Objekte nutzen.
    """

    def __init__(self, checkpoint: str | Path) -> None:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint nicht gefunden: {path.resolve()}\n"
                f"Tipp: aus dem Verzeichnis mit 'checkpoints/' starten oder "
                f"OTHELLO_CHECKPOINT setzen."
            )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net, self.extra = load_checkpoint(path, self.device)
        self.board_size = self.net.board_size
        self._lock = threading.Lock()
        # Ein Searcher je Sim-Budget, bei Bedarf angelegt und gecacht (das Anlegen
        # ist billig – es umschließt nur das geteilte Netz).
        self._searchers: dict[int, NeuralMCTS] = {}

    def _searcher(self, n_simulations: int) -> NeuralMCTS:
        s = self._searchers.get(n_simulations)
        if s is None:
            s = NeuralMCTS(
                self.net,
                MCTSConfig(n_simulations=n_simulations),
                device=self.device,
                add_noise=False,
                seed=None,
            )
            self._searchers[n_simulations] = s
        return s

    def select_move(self, state: GameState, n_simulations: int, temperature: float):
        """KI-Zug für ``state`` mit gegebenem Sim-Budget und Temperatur."""
        with self._lock:
            return self._searcher(n_simulations).select_move(state, temperature=temperature)

    def evaluate(self, state: GameState) -> float:
        """Value-Head-Bewertung der Stellung, umgerechnet auf **Schwarz-Sicht**.

        Das Netz bewertet aus Sicht des Spielers am Zug; hier auf eine feste
        Perspektive (Schwarz) normiert, damit die Frontend-Leiste konsistent ist:
        +1 = Schwarz gewinnt sicher, -1 = Weiß gewinnt sicher, 0 = ausgeglichen.
        Ein einzelner Forward-Pass (kein MCTS) – billig genug pro Stellung.
        """
        x = torch.from_numpy(encode_state(state)).unsqueeze(0).to(self.device)
        with self._lock:
            with torch.no_grad():
                _, value = self.net(x)
        v = float(value[0])
        return v if state.current_player == BLACK else -v


# --- Partie-Verwaltung ----------------------------------------------------
@dataclass
class Game:
    state: GameState
    human_color: int
    n_simulations: int
    temperature: float
    last_ai_moves: list[list[int]] = field(default_factory=list)


class NewGameRequest(BaseModel):
    human_color: str = "black"          # "black" (zieht zuerst) oder "white"
    n_simulations: int = PRESETS[DEFAULT_PRESET]["n_simulations"]
    temperature: float = PRESETS[DEFAULT_PRESET]["temperature"]


class MoveRequest(BaseModel):
    game_id: str
    move: list[int] | None = None       # [r, c]; None/leer = Pass


class AiMoveRequest(BaseModel):
    game_id: str


app = FastAPI(title="Othello AI")
engine = AIEngine(CHECKPOINT_PATH)
games: dict[str, Game] = {}

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _make_step(state: GameState, move, is_ai: bool) -> dict:
    """Beschreibt einen realen Zug (kein Pass) fürs Frontend: gesetzter Stein +
    umgedrehte Steine. Muss **vor** dem Anwenden des Zugs gerufen werden, da die
    Flips aus dem Brett *vor* dem Zug berechnet werden."""
    player = state.current_player
    flips = flips_for_move(state.board, player, move)
    return {
        "is_ai": is_ai,
        "player": "black" if player == BLACK else "white",
        "cell": [int(move[0]), int(move[1])],
        "flips": [[int(r), int(c)] for r, c in flips],
    }


def _ai_steps(game: Game) -> list[dict]:
    """Lässt die KI ziehen, solange sie am Zug und das Spiel nicht vorbei ist,
    und gibt die Animationsschritte zurück.

    Mehrere KI-Züge hintereinander sind möglich, wenn der Mensch zwischendurch
    passen muss (dann ist die KI erneut dran, ohne dass der Mensch handelt).
    Pässe erzeugen keinen Animationsschritt (nichts zu sehen).
    """
    steps: list[dict] = []
    while not game.state.is_terminal() and game.state.current_player != game.human_color:
        options = game.state.legal_moves()
        if options == [PASS]:
            game.state = game.state.apply(PASS)
            continue
        move = engine.select_move(game.state, game.n_simulations, game.temperature)
        steps.append(_make_step(game.state, move, is_ai=True))
        game.state = game.state.apply(move)
    game.last_ai_moves = [s["cell"] for s in steps]
    return steps


def _serialize(game: Game) -> dict:
    state = game.state
    counts = disc_counts(state.board)
    terminal = state.is_terminal()
    legal = state.legal_moves()
    human_turn = (not terminal) and state.current_player == game.human_color
    # Legale Menschzüge (nur wenn der Mensch am Zug ist) als [r, c]-Liste.
    # PASS *vor* dem Entpacken filtern – es ist ein String, kein (r, c)-Tupel.
    human_legal = (
        [[int(m[0]), int(m[1])] for m in legal if m != PASS] if human_turn else []
    )
    must_pass = human_turn and legal == [PASS]

    winner = None
    if terminal:
        w = state.winner()
        winner = {BLACK: "black", WHITE: "white", 0: "draw"}[w]

    # KI-Bewertung der Stellung aus Schwarz-Sicht in [-1, 1]. Bei Spielende der
    # echte Ausgang statt der (dann bedeutungslosen) Netz-Schätzung.
    if terminal:
        eval_black = 1.0 if winner == "black" else (-1.0 if winner == "white" else 0.0)
    else:
        eval_black = engine.evaluate(state)

    return {
        "board": state.board.astype(int).tolist(),
        "board_size": state.size,
        "current_player": "black" if state.current_player == BLACK else "white",
        "human_color": "black" if game.human_color == BLACK else "white",
        "n_simulations": game.n_simulations,
        "temperature": game.temperature,
        "human_turn": human_turn,
        "must_pass": must_pass,
        "legal_moves": human_legal,
        "last_ai_moves": game.last_ai_moves,
        "counts": {"black": counts[BLACK], "white": counts[WHITE]},
        "game_over": terminal,
        "winner": winner,
        "eval": round(eval_black, 4),
    }


@app.get("/api/config")
def config() -> dict:
    """Regler-Grenzen und Presets für das Frontend."""
    return {
        "sim_min": SIM_MIN, "sim_max": SIM_MAX,
        "temp_min": TEMP_MIN, "temp_max": TEMP_MAX,
        "default": DEFAULT_PRESET,
        "presets": [{"key": k, **v} for k, v in PRESETS.items()],
    }


@app.post("/api/new_game")
def new_game(req: NewGameRequest) -> dict:
    if req.human_color not in ("black", "white"):
        raise HTTPException(400, f"Ungültige Farbe: {req.human_color}")

    # Regler-Werte robust in die erlaubten Grenzen zwingen.
    n_sims = int(_clamp(req.n_simulations, SIM_MIN, SIM_MAX))
    temperature = round(_clamp(req.temperature, TEMP_MIN, TEMP_MAX), 2)

    human_color = BLACK if req.human_color == "black" else WHITE
    game = Game(
        state=GameState.initial(engine.board_size),
        human_color=human_color,
        n_simulations=n_sims,
        temperature=temperature,
    )
    # Ist die KI Schwarz, zieht sie sofort los, bis der Mensch dran ist.
    steps = _ai_steps(game)

    game_id = uuid.uuid4().hex
    games[game_id] = game
    return {"game_id": game_id, "steps": steps, **_serialize(game)}


@app.post("/api/move")
def move(req: MoveRequest) -> dict:
    game = games.get(req.game_id)
    if game is None:
        raise HTTPException(404, "Partie nicht gefunden (neu starten).")
    if game.state.is_terminal():
        raise HTTPException(400, "Partie ist bereits vorbei.")
    if game.state.current_player != game.human_color:
        raise HTTPException(400, "Die KI ist am Zug, nicht der Mensch.")

    legal = game.state.legal_moves()
    if req.move is None:
        human_move = PASS
    else:
        human_move = (int(req.move[0]), int(req.move[1]))

    if human_move not in legal:
        raise HTTPException(400, f"Illegaler Zug: {req.move}")

    # Nur den Menschzug anwenden – die KI zieht separat über /api/ai_move, damit
    # das Frontend den eigenen Zug sofort zeigen kann, *bevor* die KI nachdenkt.
    steps: list[dict] = []
    if human_move != PASS:
        steps.append(_make_step(game.state, human_move, is_ai=False))
    game.state = game.state.apply(human_move)
    game.last_ai_moves = []
    return {"game_id": req.game_id, "steps": steps, **_serialize(game)}


@app.post("/api/ai_move")
def ai_move(req: AiMoveRequest) -> dict:
    """Lässt die KI antworten (nachdem der Menschzug schon angezeigt wurde)."""
    game = games.get(req.game_id)
    if game is None:
        raise HTTPException(404, "Partie nicht gefunden (neu starten).")

    if not game.state.is_terminal() and game.state.current_player != game.human_color:
        steps = _ai_steps(game)
    else:
        steps = []
        game.last_ai_moves = []
    return {"game_id": req.game_id, "steps": steps, **_serialize(game)}


@app.get("/api/state")
def state(game_id: str) -> dict:
    game = games.get(game_id)
    if game is None:
        raise HTTPException(404, "Partie nicht gefunden.")
    return {"game_id": game_id, **_serialize(game)}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=_STATIC_DIR), name="static")
