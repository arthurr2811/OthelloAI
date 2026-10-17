// Parität JS-MCTS (web/static/engine/mcts.js) vs. NeuralMCTS (az/mcts.py).
//
//   node --test "tests/js/*.test.mjs"
//
// Beide Seiten suchen mit demselben deterministischen Dummy-Evaluator statt des
// Netzes (Python: _dummy_eval in scripts/gen_js_fixtures.py). Erwartet werden
// exakt gleiche Visit-Counts je Wurzelkind – das prüft PUCT-Auswahl,
// Tie-Breaking, Masking/Softmax, Backprop-Vorzeichen, Terminal- und Pass-Knoten.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { NeuralMCTS, maskedSoftmax, visitDistribution } from "../../web/static/engine/mcts.js";
import { initialState, legalOptions } from "../../web/static/engine/othello.js";

const load = (name) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), "utf8"));
const { cases } = load("mcts.json");
const { positions } = load("engine.json");   // Quelle für erzwungene Züge/Pässe

const VALUE = { X: 1, O: -1, ".": 0 };
const stateOf = (c) => ({
  board: Int8Array.from(c.board, (ch) => VALUE[ch]),
  size: c.size,
  player: c.player,
});

// Nachbau von _dummy_eval (Python) – muss Zeichen für Zeichen dasselbe rechnen.
async function dummyEval(state) {
  const n = state.size * state.size;
  let own = 0;
  let opp = 0;
  for (const v of state.board) {
    if (v === state.player) own++;
    else if (v === -state.player) opp++;
  }
  const logits = Array.from({ length: n + 1 }, (_, a) => ((a * 7 + own * 3 + opp) % 11) / 4);
  return { logits, value: (own - opp) / n };
}

test("Visit-Counts und Werte der Wurzelkinder wie in Python", async () => {
  assert.ok(cases.length > 0);
  for (const c of cases) {
    const mcts = new NeuralMCTS(dummyEval, { nSimulations: c.sims });
    const root = await mcts.run(stateOf(c));
    const where = `\n${c.size}x${c.size}, ${c.sims} Sims, Brett ${c.board}`;
    assert.equal(root.N, c.root_n, where);
    assert.deepEqual(root.moves, c.children.map((ch) => ch.move), where);
    root.children.forEach((child, i) => {
      assert.equal(child.N, c.children[i].n, `Zug ${root.moves[i]}${where}`);
      assert.ok(Math.abs(child.W - c.children[i].w) < 1e-9, `W Zug ${root.moves[i]}${where}`);
    });
    assert.equal(visitDistribution(root, 0).indexOf(1), c.best, where);
  }
});

test("selectMove bei T=0 nimmt den meistbesuchten Zug", async () => {
  for (const c of cases.filter((x) => x.sims > 1)) {
    const mcts = new NeuralMCTS(dummyEval, { nSimulations: c.sims });
    assert.equal(await mcts.selectMove(stateOf(c), 0), c.best);
  }
});

test("erzwungener Zug/Pass ohne Suche", async () => {
  let calls = 0;
  const counting = async (s) => { calls++; return dummyEval(s); };
  const forced = positions.map(stateOf).filter((s) => legalOptions(s).length === 1);
  assert.ok(forced.some((s) => legalOptions(s)[0] === s.size * s.size));   // inkl. Pass
  const mcts = new NeuralMCTS(counting, { nSimulations: 50 });
  for (const s of forced) assert.equal(await mcts.selectMove(s, 0), legalOptions(s)[0]);
  assert.equal(calls, 0);
});

test("Sampling bei T>0 folgt N^(1/T) und wählt nur legale Züge", async () => {
  const state = initialState(8);
  // Zufallsquelle 0 -> erster Zug mit positiver Masse; knapp unter 1 -> letzter.
  const lo = new NeuralMCTS(dummyEval, { nSimulations: 30, random: () => 0 });
  const hi = new NeuralMCTS(dummyEval, { nSimulations: 30, random: () => 1 - 1e-12 });
  const legal = legalOptions(state);
  const root = await lo.run(state);
  const pi = visitDistribution(root, 1);
  assert.ok(Math.abs(pi.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  const support = legal.filter((m) => pi[m] > 0);
  assert.equal(await lo.selectMove(state, 1), support[0]);
  assert.equal(await hi.selectMove(state, 1), support.at(-1));
});

test("maskedSoftmax: nur legale Aktionen, Summe 1", () => {
  const p = maskedSoftmax([5, 1, 2, 1000, 3], [1, 2, 4]);
  assert.equal(p[0], 0);
  assert.equal(p[3], 0);
  assert.ok(Math.abs(p[1] + p[2] + p[4] - 1) < 1e-12);
  assert.ok(p[4] > p[2] && p[2] > p[1]);
});
