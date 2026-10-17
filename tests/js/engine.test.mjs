// Parität JS-Engine (web/static/engine/othello.js) vs. Python-Engine.
//
//   node --test tests/js
//
// Die Erwartungswerte stammen aus der Python-Engine (scripts/gen_js_fixtures.py).

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
  applyMove, applyToState, encodeState, flipsForMove, initialState, isTerminal,
  legalOptions, passMove, winner,
} from "../../web/static/engine/othello.js";

const { positions } = JSON.parse(
  readFileSync(new URL("./fixtures/engine.json", import.meta.url), "utf8"),
);

const VALUE = { X: 1, O: -1, ".": 0 };
const parseBoard = (s) => Int8Array.from(s, (ch) => VALUE[ch]);
const stateOf = (p) => ({ board: parseBoard(p.board), size: p.size, player: p.player });
const where = (p) => `\n${p.size}x${p.size}, Spieler ${p.player}, Brett ${p.board}`;

test("Fixtures decken Pässe und Spielenden ab", () => {
  assert.ok(positions.some((p) => p.options.length === 1 && p.options[0] === passMove(p.size)));
  assert.ok(positions.some((p) => p.terminal && p.board.includes(".")));   // vorzeitiges Ende
});

test("Startstellung", () => {
  for (const p of positions.filter((q) => q.board === positions[0].board)) {
    const s = initialState(p.size);
    assert.deepEqual(Array.from(s.board), Array.from(parseBoard(p.board)));
    assert.equal(s.player, 1);
  }
});

test("legale Optionen (inkl. Pass / Spielende)", () => {
  for (const p of positions) {
    assert.deepEqual(legalOptions(stateOf(p)), p.options, where(p));
  }
});

test("Flips und Folgebrett je Zug", () => {
  for (const p of positions) {
    const board = parseBoard(p.board);
    for (const m of p.moves) {
      assert.deepEqual(flipsForMove(board, p.size, p.player, m.move), m.flips, where(p));
      assert.deepEqual(Array.from(applyMove(board, p.size, p.player, m.move)),
        Array.from(parseBoard(m.next)), where(p));
    }
    assert.equal(p.board, Array.from(board, (v) => ".XO".at(v)).join(""), "Brett mutiert");
  }
});

test("Zustandsübergang wechselt den Spieler, Pass lässt das Brett stehen", () => {
  for (const p of positions) {
    const s = stateOf(p);
    for (const move of p.options) {
      const next = applyToState(s, move);
      assert.equal(next.player, -p.player);
      if (move === passMove(p.size)) assert.deepEqual(next.board, s.board);
    }
  }
});

test("illegale Züge werfen", () => {
  const s = initialState(8);
  assert.throws(() => applyMove(s.board, 8, s.player, 0));
  assert.throws(() => applyToState(s, passMove(8)));   // Pass trotz legaler Züge
});

test("Terminal und Gewinner", () => {
  for (const p of positions) {
    const s = stateOf(p);
    assert.equal(isTerminal(s), p.terminal, where(p));
    assert.equal(winner(s.board), p.winner, where(p));
  }
});

test("Netz-Encoding", () => {
  const withEnc = positions.filter((p) => p.encoding);
  assert.ok(withEnc.length > 100);
  for (const p of withEnc) {
    assert.equal(Array.from(encodeState(stateOf(p))).join(""), p.encoding, where(p));
  }
});
