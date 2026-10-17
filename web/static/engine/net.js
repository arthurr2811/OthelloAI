// Netz-Auswertung mit onnxruntime-web (model.onnx aus scripts/export_onnx.py).
//
// `ort` wird hereingereicht statt importiert: Der Worker lädt es vom CDN, der
// Node-Test aus npm – die Auswertung selbst ist dieselbe.

import { BLACK, encodeState } from "./othello.js";

/**
 * Lädt das Modell und gibt den Evaluator für die MCTS zurück:
 *   async (state) => { logits: Float32Array(size*size+1), value }
 * mit value aus Sicht des Spielers am Zug (wie das Netz).
 */
export async function createEvaluator(ort, model, options = {}) {
  const session = await ort.InferenceSession.create(model, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
    ...options,
  });

  return async function evaluate(state) {
    const x = new ort.Tensor("float32", encodeState(state), [1, 3, state.size, state.size]);
    const out = await session.run({ x });
    return { logits: out.policy.data, value: out.value.data[0] };
  };
}

/** Value-Head-Bewertung aus **Schwarz-Sicht** (+1 = Schwarz gewinnt sicher),
 * wie AIEngine.evaluate im alten Server – für die Bewertungsleiste. */
export async function evaluateBlack(evaluate, state) {
  const { value } = await evaluate(state);
  return state.player === BLACK ? value : -value;
}
