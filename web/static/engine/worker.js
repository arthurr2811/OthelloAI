// Web Worker: Netz + MCTS laufen hier, damit die Oberfläche während der Suche
// flüssig bleibt. Wird von engine/client.js als Module-Worker gestartet.
//
// Protokoll (jede Anfrage trägt eine `id`, die Antwort dieselbe):
//   { type: "init" }                                  -> { ready: true }
//   { type: "selectMove", state, sims, temperature } -> { move }
//   { type: "evaluate", state }                       -> { value }  (Schwarz-Sicht)
// Fehler kommen als { id, error } zurück.

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs";

import { NeuralMCTS } from "./mcts.js";
import { createEvaluator, evaluateBlack } from "./net.js";

// Die .wasm-Dateien vom selben CDN-Pfad wie das Skript laden.
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

let evaluatorPromise = null;

function evaluator() {
  if (!evaluatorPromise) {
    evaluatorPromise = createEvaluator(ort, new URL("../model.onnx", import.meta.url).href);
  }
  return evaluatorPromise;
}

const handlers = {
  async init() {
    await evaluator();
    return { ready: true };
  },

  async selectMove({ state, sims, temperature }) {
    const mcts = new NeuralMCTS(await evaluator(), { nSimulations: sims });
    return { move: await mcts.selectMove(state, temperature) };
  },

  async evaluate({ state }) {
    return { value: await evaluateBlack(await evaluator(), state) };
  },
};

// Anfragen strikt nacheinander abarbeiten (wie der Lock im alten Server): die
// ORT-Session soll nicht von zwei Suchen gleichzeitig benutzt werden.
let queue = Promise.resolve();

self.onmessage = ({ data }) => {
  const { id, type } = data;
  queue = queue.then(async () => {
    try {
      self.postMessage({ id, ...(await handlers[type](data)) });
    } catch (e) {
      self.postMessage({ id, error: String((e && e.message) || e) });
    }
  });
};
