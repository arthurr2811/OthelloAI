// Partie-Verwaltung im Browser – ersetzt die Endpunkte des früheren
// FastAPI-Servers (new_game / move / ai_move) und liefert dieselben Antworten,
// damit app.js nur die Aufrufe tauschen muss.
//
// Koordinaten nach außen wie bisher als [r, c]; intern nutzt die Engine
// Feld-Indizes (r*size + c, Pass = size*size).

import {
  BLACK, WHITE, applyToState, discCounts, flipsForMove, initialState, isTerminal,
  legalOptions, passMove, winner,
} from "./othello.js";

// --- Stärke-Parameter -----------------------------------------------------
// Zwei Regler, die 1:1 an die MCTS gehen:
//   n_simulations : Denkbudget pro Zug (1 ≈ reiner Policy-Zufall, mehrere hundert = ernst)
//   temperature   : Zug-Zufall über die ganze Partie; 0 = immer der beste Zug,
//                   höher = die KI patzt öfter und wird für Menschen schlagbar.
// Presets belegen nur die Regler vor; gespielt wird mit den Slider-Werten.
export const CONFIG = {
  sim_min: 1,
  sim_max: 800,
  sim_warn: 250,   // ab hier Hinweis im Frontend: im Browser dauert ein Zug dann lange
  temp_min: 0,
  temp_max: 3,
  default: "medium",
  presets: [
    { key: "easy", label: "Einfach", n_simulations: 10, temperature: 1.5 },
    { key: "medium", label: "Mittel", n_simulations: 80, temperature: 0.4 },
    { key: "hard", label: "Schwer", n_simulations: 250, temperature: 0.0 },
  ],
};

const BOARD_SIZE = 8;   // Das ausgelieferte Netz ist ein 8x8-Netz.

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const colorName = (player) => (player === BLACK ? "black" : "white");

export class Game {
  /**
   * @param engine  EngineClient (selectMove / evaluate)
   */
  constructor(engine, { humanColor, nSimulations, temperature }) {
    if (humanColor !== "black" && humanColor !== "white") {
      throw new Error(`Ungültige Farbe: ${humanColor}`);
    }
    this.engine = engine;
    this.state = initialState(BOARD_SIZE);
    this.humanColor = humanColor === "black" ? BLACK : WHITE;
    // Regler-Werte robust in die erlaubten Grenzen zwingen.
    this.nSimulations = Math.round(clamp(nSimulations, CONFIG.sim_min, CONFIG.sim_max));
    this.temperature = Math.round(clamp(temperature, CONFIG.temp_min, CONFIG.temp_max) * 100) / 100;
    this.lastAiMoves = [];
  }

  /** Neue Partie; ist die KI Schwarz, zieht sie sofort, bis der Mensch dran ist. */
  async start() {
    const steps = await this._aiSteps();
    return { steps, ...(await this.serialize()) };
  }

  /**
   * Menschzug anwenden ([r, c] oder null = Pass). Die KI zieht separat über
   * aiMove(), damit das Frontend den eigenen Zug zeigen kann, bevor sie denkt.
   */
  async move(cell) {
    if (isTerminal(this.state)) throw new Error("Partie ist bereits vorbei.");
    if (this.state.player !== this.humanColor) throw new Error("Die KI ist am Zug, nicht der Mensch.");
    const size = this.state.size;
    const move = cell === null ? passMove(size) : cell[0] * size + cell[1];
    if (!legalOptions(this.state).includes(move)) throw new Error(`Illegaler Zug: ${cell}`);

    const steps = move === passMove(size) ? [] : [this._step(move, false)];
    this.state = applyToState(this.state, move);
    this.lastAiMoves = [];
    return { steps, ...(await this.serialize()) };
  }

  /** Lässt die KI antworten (nachdem der Menschzug schon angezeigt wurde). */
  async aiMove() {
    let steps = [];
    if (!isTerminal(this.state) && this.state.player !== this.humanColor) {
      steps = await this._aiSteps();
    } else {
      this.lastAiMoves = [];
    }
    return { steps, ...(await this.serialize()) };
  }

  /**
   * Lässt die KI ziehen, solange sie dran ist – mehrfach, wenn der Mensch
   * zwischendurch passen muss. Pässe erzeugen keinen Animationsschritt.
   */
  async _aiSteps() {
    const steps = [];
    while (!isTerminal(this.state) && this.state.player !== this.humanColor) {
      const options = legalOptions(this.state);
      const pass = passMove(this.state.size);
      if (options.length === 1 && options[0] === pass) {
        this.state = applyToState(this.state, pass);
        continue;
      }
      const move = await this.engine.selectMove(this.state, this.nSimulations, this.temperature);
      steps.push(this._step(move, true));
      this.state = applyToState(this.state, move);
    }
    this.lastAiMoves = steps.map((s) => s.cell);
    return steps;
  }

  /** Animationsschritt für einen realen Zug – vor dem Anwenden aufrufen, da die
   * Flips aus dem Brett *vor* dem Zug berechnet werden. */
  _step(move, isAi) {
    const { board, size, player } = this.state;
    const rc = (i) => [Math.floor(i / size), i % size];
    return {
      is_ai: isAi,
      player: colorName(player),
      cell: rc(move),
      flips: flipsForMove(board, size, player, move).map(rc),
    };
  }

  /** Zustand in der Form der früheren Server-Antwort. */
  async serialize() {
    const state = this.state;
    const { board, size } = state;
    const terminal = isTerminal(state);
    const humanTurn = !terminal && state.player === this.humanColor;
    const options = legalOptions(state);
    const pass = passMove(size);
    const rc = (i) => [Math.floor(i / size), i % size];

    let win = null;
    if (terminal) win = { [BLACK]: "black", [WHITE]: "white", 0: "draw" }[winner(board)];

    // KI-Bewertung aus Schwarz-Sicht in [-1, 1]; bei Spielende der echte Ausgang.
    let evalBlack;
    if (terminal) evalBlack = win === "black" ? 1 : win === "white" ? -1 : 0;
    else evalBlack = await this.engine.evaluate(state);

    return {
      board: Array.from({ length: size }, (_, r) => Array.from(board.subarray(r * size, (r + 1) * size))),
      board_size: size,
      current_player: colorName(state.player),
      human_color: colorName(this.humanColor),
      n_simulations: this.nSimulations,
      temperature: this.temperature,
      human_turn: humanTurn,
      must_pass: humanTurn && options.length === 1 && options[0] === pass,
      legal_moves: humanTurn ? options.filter((m) => m !== pass).map(rc) : [],
      last_ai_moves: this.lastAiMoves,
      counts: discCounts(board),
      game_over: terminal,
      winner: win,
      eval: Math.round(evalBlack * 1e4) / 1e4,
    };
  }
}
