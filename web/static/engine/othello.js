// Reversi-Spiel-Engine fürs Browser-Frontend – 1:1-Port von othello/board.py
// und encode_state aus az/encoding.py.
//
// Konvention wie in Python:
//    1 = Schwarz (zieht zuerst), -1 = Weiß, 0 = leer.
// Das Brett ist ein flaches Int8Array der Länge size*size, Feld (r, c) liegt an
// Index r*size + c. Ein Zug ist dieser Index; der Pass-Zug ist size*size (wie der
// Pass-Index der Policy).
//
// Ein Zustand ist { board, size, player }. Alle Funktionen sind pure: Züge
// erzeugen ein neues Brett, das alte bleibt unverändert.
//
// Die Semantik (inkl. Reihenfolge von Zügen und Flips) ist per Fixtures gegen die
// Python-Engine abgesichert: tests/js/engine.test.mjs.

export const BLACK = 1;
export const WHITE = -1;
export const EMPTY = 0;

// Die 8 Richtungen (dr, dc), gleiche Reihenfolge wie DIRECTIONS in Python – sie
// bestimmt die Reihenfolge der Flips (und damit der Umklapp-Animation).
const DIRECTIONS = [
  [-1, -1], [-1, 0], [-1, 1],
  [0, -1],           [0, 1],
  [1, -1],  [1, 0],  [1, 1],
];

export const passMove = (size) => size * size;

export function initialState(size = 8) {
  if (size < 2 || size % 2 !== 0) {
    throw new Error(`board_size muss gerade und >= 2 sein, war ${size}`);
  }
  const board = new Int8Array(size * size);
  const lo = size / 2 - 1;
  const hi = size / 2;
  // Diagonal gleichfarbig: Weiß auf (lo,lo)/(hi,hi), Schwarz auf (lo,hi)/(hi,lo).
  board[lo * size + lo] = WHITE;
  board[hi * size + hi] = WHITE;
  board[lo * size + hi] = BLACK;
  board[hi * size + lo] = BLACK;
  return { board, size, player: BLACK };
}

/** Indizes der Steine, die ein Zug auf `move` für `player` umdreht (leer = illegal). */
export function flipsForMove(board, size, player, move) {
  if (board[move] !== EMPTY) return [];
  const r = Math.floor(move / size);
  const c = move % size;
  const opp = -player;
  const flips = [];
  for (const [dr, dc] of DIRECTIONS) {
    // Ununterbrochene Kette gegnerischer Steine, abgeschlossen von einem eigenen.
    let rr = r + dr;
    let cc = c + dc;
    const start = flips.length;
    while (rr >= 0 && rr < size && cc >= 0 && cc < size && board[rr * size + cc] === opp) {
      flips.push(rr * size + cc);
      rr += dr;
      cc += dc;
    }
    const closed = rr >= 0 && rr < size && cc >= 0 && cc < size && board[rr * size + cc] === player;
    if (!closed) flips.length = start;   // Kette nicht eingeklammert: verwerfen
  }
  return flips;
}

/** Alle legalen Felder für `player` (ohne Pass), zeilenweise sortiert. */
export function legalMoves(board, size, player) {
  const moves = [];
  for (let i = 0; i < size * size; i++) {
    if (board[i] === EMPTY && flipsForMove(board, size, player, i).length) moves.push(i);
  }
  return moves;
}

export function hasLegalMove(board, size, player) {
  for (let i = 0; i < size * size; i++) {
    if (board[i] === EMPTY && flipsForMove(board, size, player, i).length) return true;
  }
  return false;
}

/** Neues Brett nach `move`; wirft bei illegalem Zug. */
export function applyMove(board, size, player, move) {
  const flips = flipsForMove(board, size, player, move);
  if (!flips.length) throw new Error(`Illegaler Zug ${move} für Spieler ${player}`);
  const next = board.slice();
  next[move] = player;
  for (const f of flips) next[f] = player;
  return next;
}

/**
 * Optionen für den Spieler am Zug (wie GameState.legal_moves):
 * reale Felder, sonst [PASS] wenn nur der Gegner ziehen kann, sonst [] (Spielende).
 */
export function legalOptions(state) {
  const { board, size, player } = state;
  const moves = legalMoves(board, size, player);
  if (moves.length) return moves;
  if (hasLegalMove(board, size, -player)) return [passMove(size)];
  return [];
}

/** Neuer Zustand nach `move` (Feld-Index oder Pass). */
export function applyToState(state, move) {
  const { board, size, player } = state;
  if (move === passMove(size)) {
    if (hasLegalMove(board, size, player)) throw new Error("PASS unzulässig: es gibt legale Züge");
    return { board, size, player: -player };
  }
  return { board: applyMove(board, size, player, move), size, player: -player };
}

/** Spiel vorbei, wenn kein Spieler mehr ziehen kann. */
export function isTerminal(state) {
  const { board, size } = state;
  return !hasLegalMove(board, size, BLACK) && !hasLegalMove(board, size, WHITE);
}

export function discCounts(board) {
  let black = 0;
  let white = 0;
  for (const v of board) {
    if (v === BLACK) black++;
    else if (v === WHITE) white++;
  }
  return { black, white };
}

/** Gewinner nach Steinmehrheit: BLACK, WHITE oder EMPTY (Unentschieden). */
export function winner(board) {
  const { black, white } = discCounts(board);
  if (black > white) return BLACK;
  if (white > black) return WHITE;
  return EMPTY;
}

/**
 * Netz-Eingabe (3, size, size) als flaches Float32Array, aus Sicht des Spielers
 * am Zug: Ebene 0 eigene Steine, 1 gegnerische, 2 konstant 1 wenn Schwarz zieht.
 */
export function encodeState(state) {
  const { board, size, player } = state;
  const n = size * size;
  const x = new Float32Array(3 * n);
  const color = player === BLACK ? 1 : 0;
  for (let i = 0; i < n; i++) {
    if (board[i] === player) x[i] = 1;
    else if (board[i] === -player) x[n + i] = 1;
    x[2 * n + i] = color;
  }
  return x;
}
