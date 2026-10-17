// Hauptthread-Seite des KI-Workers (engine/worker.js): Promise-API über
// postMessage, plus Abbruch einer laufenden Suche.

export class AbortedError extends Error {
  constructor() {
    super("abgebrochen");
    this.name = "AbortedError";
  }
}

export class EngineClient {
  constructor() {
    this._start();
  }

  /** Erfüllt, sobald das Modell im Worker geladen ist. */
  get ready() {
    return this._ready;
  }

  /** KI-Zug (Feld-Index oder Pass) für `state`. */
  selectMove(state, sims, temperature) {
    return this._call("selectMove", { state, sims, temperature }).then((r) => r.move);
  }

  /** Netz-Bewertung der Stellung aus Schwarz-Sicht in [-1, 1]. */
  evaluate(state) {
    return this._call("evaluate", { state }).then((r) => r.value);
  }

  /**
   * Bricht laufende Anfragen ab (sie werfen AbortedError). Eine Suche lässt sich
   * im Worker nicht unterbrechen, deshalb wird er neu gestartet – das Modell
   * kommt dann aus dem Browser-Cache.
   */
  abort() {
    if (this._pending.size === 0) return;
    this._worker.terminate();
    this._rejectAll(new AbortedError());
    this._start();
  }

  _start() {
    this._worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    this._pending = new Map();
    this._nextId = 0;
    this._worker.onmessage = ({ data }) => {
      const p = this._pending.get(data.id);
      if (!p) return;
      this._pending.delete(data.id);
      if (data.error) p.reject(new Error(data.error));
      else p.resolve(data);
    };
    // Z. B. CDN nicht erreichbar: das Worker-Skript lädt gar nicht erst.
    this._worker.onerror = (e) => {
      e.preventDefault();
      this._rejectAll(new Error(e.message || "KI-Worker konnte nicht gestartet werden"));
    };
    this._ready = this._call("init");
    this._ready.catch(() => {});   // Fehler meldet, wer auf `ready` wartet
  }

  _call(type, payload = {}) {
    const id = this._nextId++;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this._worker.postMessage({ id, type, ...payload });
    });
  }

  _rejectAll(err) {
    for (const p of this._pending.values()) p.reject(err);
    this._pending.clear();
  }
}
