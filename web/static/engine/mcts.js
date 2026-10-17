// PUCT-MCTS fürs Browser-Frontend – 1:1-Port von NeuralMCTS aus az/mcts.py
// (Spiel-Modus: ohne Dirichlet-Rauschen).
//
// Das Netz ist als Evaluator injiziert:  async (state) => { logits, value }
//   logits : Policy-Logits über alle Aktionen (Länge size*size + 1, Pass zuletzt)
//   value  : Bewertung in [-1, 1] aus Sicht des Spielers am Zug
// So läuft dieselbe Suche mit onnxruntime-web (Worker) und mit einem
// deterministischen Test-Evaluator (tests/js/mcts.test.mjs prüft die Visit-Counts
// gegen die Python-Suche).

import { applyToState, isTerminal, legalOptions, winner, EMPTY } from "./othello.js";

class Node {
  constructor(state, parent, mover, prior) {
    this.state = state;
    this.parent = parent;
    // Spieler, der den Zug in diesen Knoten hinein machte. null = Wurzel.
    this.mover = mover;
    this.prior = prior;          // P(a) aus der Policy des Elternteils
    this.moves = [];             // Züge der Kinder, parallel zu children
    this.children = [];
    this.N = 0;
    this.W = 0;
    this.isExpanded = false;
  }

  /** Mittlerer Wert aus Sicht des `mover` (0, solange unbesucht). */
  get q() {
    return this.N > 0 ? this.W / this.N : 0;
  }
}

/** Trägt `value` (Sicht: Spieler am Zug im Blatt) den Pfad hinauf, mit
 * Vorzeichenwechsel pro Ebene (W ist aus Sicht des `mover`). */
function backprop(leaf, value) {
  for (let node = leaf; node !== null; node = node.parent) {
    node.N += 1;
    if (node.mover !== null) node.W += -value;
    value = -value;
  }
}

/** Softmax nur über die legalen Aktionen `options`; Rest exakt 0. */
export function maskedSoftmax(logits, options) {
  const priors = new Float64Array(logits.length);
  let max = -Infinity;
  for (const a of options) max = Math.max(max, logits[a]);
  let total = 0;
  for (const a of options) {
    priors[a] = Math.exp(logits[a] - max);
    total += priors[a];
  }
  for (const a of options) {
    // Fallback (Summe 0, z. B. NaN-Logits): gleichverteilt über legal.
    priors[a] = total > 0 ? priors[a] / total : 1 / options.length;
  }
  return priors;
}

/** Echtes Partie-Ergebnis aus Sicht des Spielers am Zug in `state`. */
function terminalValue(state) {
  const w = winner(state.board);
  if (w === EMPTY) return 0;
  return w === state.player ? 1 : -1;
}

/** Normalisierte Visit-Counts über alle Aktionen, mit Temperatur (N^(1/T)). */
export function visitDistribution(root, temperature) {
  const size = root.state.size;
  const counts = new Float64Array(size * size + 1);
  root.children.forEach((child, i) => { counts[root.moves[i]] = child.N; });
  if (counts.every((n) => n === 0)) {   // Wurzel ohne Sims: gleichverteilt legal
    for (const m of root.moves) counts[m] = 1;
  }
  const pi = new Float64Array(counts.length);
  if (temperature === 0) {
    let best = 0;
    for (let a = 1; a < counts.length; a++) if (counts[a] > counts[best]) best = a;
    pi[best] = 1;
    return pi;
  }
  let total = 0;
  for (let a = 0; a < counts.length; a++) {
    pi[a] = counts[a] ** (1 / temperature);
    total += pi[a];
  }
  for (let a = 0; a < pi.length; a++) pi[a] /= total;
  return pi;
}

export class NeuralMCTS {
  /**
   * @param evaluate  async (state) => { logits, value }
   * @param nSimulations  Simulationsbudget pro Suche
   * @param cPuct  Explorations-Gewicht (wie MCTSConfig.c_puct)
   * @param random  Zufallsquelle in [0, 1) fürs Sampling bei T > 0
   */
  constructor(evaluate, { nSimulations, cPuct = 1.5, random = Math.random }) {
    this.evaluate = evaluate;
    this.nSimulations = nSimulations;
    this.cPuct = cPuct;
    this.random = random;
  }

  /** Führt das Simulationsbudget aus und gibt den Wurzelknoten zurück. */
  async run(state) {
    const root = new Node(state, null, null, 1);
    backprop(root, await this.expandAndEvaluate(root));   // zählt als 1. Simulation
    for (let i = 0; i < this.nSimulations - 1; i++) {
      const leaf = this.selectLeaf(root);
      const value = isTerminal(leaf.state)
        ? terminalValue(leaf.state)
        : await this.expandAndEvaluate(leaf);
      backprop(leaf, value);
    }
    return root;
  }

  /** Wählt einen Zug: bei T > 0 gesampelt, bei T = 0 der meistbesuchte. */
  async selectMove(state, temperature) {
    const options = legalOptions(state);
    if (options.length === 1) return options[0];   // erzwungener Zug/Pass
    const root = await this.run(state);
    const pi = visitDistribution(root, temperature);
    if (temperature === 0) return pi.indexOf(1);
    let u = this.random();
    for (let a = 0; a < pi.length; a++) {
      u -= pi[a];
      if (u < 0) return a;
    }
    // Rundungsrest: letzter Zug mit positiver Wahrscheinlichkeit.
    for (let a = pi.length - 1; a >= 0; a--) if (pi[a] > 0) return a;
  }

  /** PUCT-Abstieg bis zu einem nicht expandierten oder terminalen Knoten. */
  selectLeaf(root) {
    let node = root;
    while (node.isExpanded && node.children.length) node = this.selectChild(node);
    return node;
  }

  /** PUCT: Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a)); bei Gleichstand der erste. */
  selectChild(node) {
    const sqrtParent = Math.sqrt(node.N);
    let bestScore = -Infinity;
    let best = null;
    for (const child of node.children) {
      const score = child.q + (this.cPuct * child.prior * sqrtParent) / (1 + child.N);
      if (score > bestScore) {
        bestScore = score;
        best = child;
      }
    }
    return best;
  }

  /** Bewertet `node` mit dem Netz und legt alle legalen Kinder an.
   * Rückgabe: Value aus Sicht des Spielers am Zug in `node.state`. */
  async expandAndEvaluate(node) {
    const { logits, value } = await this.evaluate(node.state);
    const options = legalOptions(node.state);
    const priors = maskedSoftmax(logits, options);
    const mover = node.state.player;
    for (const move of options) {
      node.moves.push(move);
      node.children.push(new Node(applyToState(node.state, move), node, mover, priors[move]));
    }
    node.isExpanded = true;
    return value;
  }
}
