"use strict";

const el = {
  board: document.getElementById("board"),
  presets: document.getElementById("presets"),
  sims: document.getElementById("sims"),
  simsVal: document.getElementById("sims-val"),
  temp: document.getElementById("temp"),
  tempVal: document.getElementById("temp-val"),
  speed: document.getElementById("speed"),
  speedVal: document.getElementById("speed-val"),
  color: document.getElementById("color"),
  newGame: document.getElementById("new-game"),
  soundToggle: document.getElementById("sound-toggle"),
  status: document.getElementById("status"),
  countBlack: document.getElementById("count-black"),
  countWhite: document.getElementById("count-white"),
  passBtn: document.getElementById("pass-btn"),
  thinking: document.getElementById("thinking"),
};

// Animations-Basiszeiten (ms) bei Tempo 1.0×. Alle werden durch `speed`
// geteilt: höheres Tempo = kürzere Delays.
const INTRO_DELAY = 200;   // Abstand zwischen den 4 Startsteinen
const PLACE_DUR = 150;     // Pop-in eines gesetzten Steins
const FLIP_STAGGER = 80;   // Versatz zwischen zwei umklappenden Steinen
const FLIP_DUR = 220;      // Dauer einer Umklapp-Animation

// Feste Pause (NICHT vom Tempo-Regler skaliert)
const AI_REVEAL_PAUSE = 550;

let speed = 1;             // Animationstempo (Slider); teilt alle Basiszeiten
const scaled = (ms) => ms / speed;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// CSS-Animationsdauern (Pop/Flip) ans Tempo koppeln.
function applyAnimVars() {
  el.board.style.setProperty("--pop-dur", `${scaled(PLACE_DUR)}ms`);
  el.board.style.setProperty("--flip-dur", `${scaled(FLIP_DUR)}ms`);
}

let gameId = null;
let current = null;        // letzter (autoritativer) Serverzustand
let busy = false;          // blockiert Eingaben während Animation/KI-Zug
let running = false;       // läuft gerade eine Partie? (steuert Sperre + Button)
let gen = 0;               // Generationszähler: invalidiert laufende Abläufe bei Abbruch
let presets = [];          // [{key,label,n_simulations,temperature}]
let size = 8;
let cells = [];            // cells[r][c] -> DOM-Element
let boardModel = [];       // lokales Brett, das die Animation mitführt
let humanLegal = new Set();

/* ---------- Sound (Web Audio, keine Dateien nötig) ---------- */

let soundOn = true;
let audioCtx = null;

function ac() {
  if (!audioCtx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return null;
    audioCtx = new AC();
  }
  if (audioCtx.state === "suspended") audioCtx.resume();
  return audioCtx;
}

// Ein kurzer Ton mit weichem An-/Abschwellen.
function tone(freq, dur, { type = "sine", gain = 0.15, delay = 0 } = {}) {
  if (!soundOn) return;
  const ctx = ac();
  if (!ctx) return;
  const t0 = ctx.currentTime + delay;
  const osc = ctx.createOscillator();
  const g = ctx.createGain();
  osc.type = type;
  osc.frequency.value = freq;
  g.gain.setValueAtTime(0.0001, t0);
  g.gain.linearRampToValueAtTime(gain, t0 + 0.008);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  osc.connect(g);
  g.connect(ctx.destination);
  osc.start(t0);
  osc.stop(t0 + dur + 0.03);
}

const sfx = {
  place: () => tone(300, 0.12, { type: "triangle", gain: 0.22 }),
  // Flips leicht ansteigend, aber leise – auch bei vielen Steinen nicht nervig.
  flip: (i) => tone(520 + Math.min(i, 6) * 45, 0.07, { type: "sine", gain: 0.07 }),
  start: () => [523, 659, 784].forEach((f, i) => tone(f, 0.18, { gain: 0.13, delay: i * 0.08 })),
  win: () => [523, 659, 784, 1047].forEach((f, i) => tone(f, 0.28, { gain: 0.16, delay: i * 0.12 })),
  lose: () => [440, 349, 262].forEach((f, i) => tone(f, 0.32, { type: "sine", gain: 0.16, delay: i * 0.14 })),
};

function toggleSound() {
  soundOn = !soundOn;
  el.soundToggle.textContent = soundOn ? "🔊" : "🔇";
  el.soundToggle.classList.toggle("muted", !soundOn);
  if (soundOn) { ac(); sfx.place(); }   // kurzer Bestätigungston + AudioContext wecken
}

async function api(path, opts) {
  const res = await fetch(path, {
    method: opts && opts.body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return res.json();
}

/* ---------- Einstellungen / Presets ---------- */

async function loadConfig() {
  const cfg = await api("/api/config");
  presets = cfg.presets;
  el.sims.min = cfg.sim_min;
  el.sims.max = cfg.sim_max;
  el.temp.min = cfg.temp_min;
  el.temp.max = cfg.temp_max;

  el.presets.innerHTML = "";
  for (const p of presets) {
    const btn = document.createElement("button");
    btn.className = "preset-btn";
    btn.textContent = p.label;
    btn.dataset.key = p.key;
    btn.title = `${p.n_simulations} Simulationen, Temperatur ${p.temperature}`;
    btn.addEventListener("click", () => applyPreset(p));
    el.presets.appendChild(btn);
  }
  const def = presets.find((p) => p.key === cfg.default) || presets[0];
  if (def) applyPreset(def);
}

function applyPreset(p) {
  el.sims.value = p.n_simulations;
  el.temp.value = p.temperature;
  syncSliderLabels();
  markActivePreset();
}

function syncSliderLabels() {
  el.simsVal.textContent = el.sims.value;
  el.tempVal.textContent = Number(el.temp.value).toFixed(1);
}

// Stärke-Einstellungen (Sims/Temperatur/Presets/Farbe) gelten pro Partie und
// werden beim Start eingefroren – daher während einer laufenden Partie sperren.
// Ausgenommen: Tempo-Regler und Sound, die live wirken.
function updateSettingsLock() {
  el.sims.disabled = running;
  el.temp.disabled = running;
  el.color.disabled = running;
  for (const btn of el.presets.children) btn.disabled = running;
}

// Der Hauptbutton ist im Spiel „Partie abbrechen", sonst „Neue Partie".
function refreshButton() {
  if (running) {
    el.newGame.textContent = "Partie abbrechen";
    el.newGame.classList.add("danger");
  } else {
    el.newGame.textContent = "Neue Partie";
    el.newGame.classList.remove("danger");
  }
}

function markActivePreset() {
  const s = Number(el.sims.value);
  const t = Number(el.temp.value);
  for (const btn of el.presets.children) {
    const p = presets.find((x) => x.key === btn.dataset.key);
    const match = p && p.n_simulations === s && Math.abs(p.temperature - t) < 1e-6;
    btn.classList.toggle("active", !!match);
  }
}

/* ---------- Brett-Aufbau & Steine ---------- */

function buildGrid(n) {
  size = n;
  el.board.style.gridTemplateColumns = `repeat(${n}, 1fr)`;
  el.board.innerHTML = "";
  cells = [];
  boardModel = [];
  for (let r = 0; r < n; r++) {
    const row = [];
    const mrow = [];
    for (let c = 0; c < n; c++) {
      const cell = document.createElement("div");
      cell.className = "cell";
      cell.addEventListener("click", () => onCellClick(r, c));
      el.board.appendChild(cell);
      row.push(cell);
      mrow.push(0);
    }
    cells.push(row);
    boardModel.push(mrow);
  }
}

function placeStone(r, c, value, animate) {
  const cell = cells[r][c];
  const hint = cell.querySelector(".hint");
  if (hint) hint.remove();
  let s = cell.querySelector(".stone");
  if (!s) {
    s = document.createElement("div");
    s.className = "stone";
    cell.appendChild(s);
  }
  s.classList.remove("black", "white");
  s.classList.add(value === 1 ? "black" : "white");
  if (animate) {
    s.classList.remove("pop");
    void s.offsetWidth;   // Reflow erzwingen, damit die Animation neu startet
    s.classList.add("pop");
  }
  boardModel[r][c] = value;
}

function flipStone(r, c, value) {
  const s = cells[r][c].querySelector(".stone");
  if (!s) { placeStone(r, c, value, false); return; }
  s.classList.remove("flip");
  void s.offsetWidth;
  s.classList.add("flip");
  // Farbe genau in der Mitte des Klappens tauschen (dann steht der Stein hochkant).
  setTimeout(() => {
    s.classList.remove("black", "white");
    s.classList.add(value === 1 ? "black" : "white");
  }, scaled(FLIP_DUR) / 2);
  boardModel[r][c] = value;
}

function removeStone(r, c) {
  const s = cells[r][c].querySelector(".stone");
  if (s) s.remove();
  boardModel[r][c] = 0;
}

function updateCounts() {
  let b = 0, w = 0;
  for (let r = 0; r < size; r++)
    for (let c = 0; c < size; c++) {
      if (boardModel[r][c] === 1) b++;
      else if (boardModel[r][c] === -1) w++;
    }
  el.countBlack.textContent = b;
  el.countWhite.textContent = w;
}

/* ---------- Animation der Züge ---------- */

function startPosition(n) {
  // Reihenfolge im Uhrzeigersinn, damit das Erscheinen hübsch aussieht.
  const lo = n / 2 - 1, hi = n / 2;
  return [
    { r: lo, c: lo, v: -1 },
    { r: lo, c: hi, v: 1 },
    { r: hi, c: hi, v: -1 },
    { r: hi, c: lo, v: 1 },
  ];
}

async function animateIntro() {
  for (let r = 0; r < size; r++)
    for (let c = 0; c < size; c++) removeStone(r, c);
  updateCounts();
  await sleep(scaled(150));
  sfx.start();
  for (const s of startPosition(size)) {
    placeStone(s.r, s.c, s.v, true);
    updateCounts();
    await sleep(scaled(INTRO_DELAY));
  }
}

async function animateStep(step) {
  const [r, c] = step.cell;
  const val = step.player === "black" ? 1 : -1;
  placeStone(r, c, val, true);
  sfx.place();
  updateCounts();
  await sleep(scaled(PLACE_DUR));
  let i = 0;
  for (const [fr, fc] of step.flips) {
    flipStone(fr, fc, val);
    sfx.flip(i++);
    updateCounts();
    await sleep(scaled(FLIP_STAGGER));
  }
  if (step.flips.length) await sleep(scaled(FLIP_DUR));
}

function clearLastAI() {
  for (let r = 0; r < size; r++)
    for (let c = 0; c < size; c++) cells[r][c].classList.remove("last-ai");
}

async function animateSteps(steps) {
  clearLastAI();   // alte KI-Markierung entfernen, bevor der neue Zug beginnt
  for (const step of steps) {
    if (step.is_ai) {
      // Zielfeld zuerst markieren (gelbes Kästchen), feste Pause, dann Stein.
      clearLastAI();
      cells[step.cell[0]][step.cell[1]].classList.add("last-ai");
      await sleep(AI_REVEAL_PAUSE);   // bewusst NICHT tempo-skaliert
    }
    await animateStep(step);
  }
}

/* ---------- Abschluss eines Halbzugs ---------- */

function clearHints() {
  for (let r = 0; r < size; r++)
    for (let c = 0; c < size; c++) {
      const cell = cells[r][c];
      cell.classList.remove("playable");
      const h = cell.querySelector(".hint");
      if (h) h.remove();
    }
}

function syncBoard() {
  // Lokales Modell an den autoritativen Serverzustand angleichen (korrigiert
  // etwaige Drift).
  for (let r = 0; r < size; r++)
    for (let c = 0; c < size; c++) {
      const v = current.board[r][c];
      if (v !== boardModel[r][c]) {
        if (v === 0) removeStone(r, c);
        else placeStone(r, c, v, false);
      }
    }
}

function finishTurn() {
  syncBoard();
  el.countBlack.textContent = current.counts.black;
  el.countWhite.textContent = current.counts.white;

  // KI-Zug(e) hervorheben.
  for (let r = 0; r < size; r++)
    for (let c = 0; c < size; c++) cells[r][c].classList.remove("last-ai");
  for (const [r, c] of current.last_ai_moves) cells[r][c].classList.add("last-ai");

  // Legale Menschzüge markieren.
  clearHints();
  humanLegal = new Set(current.legal_moves.map(([r, c]) => r * size + c));
  if (current.human_turn) {
    for (const [r, c] of current.legal_moves) {
      const cell = cells[r][c];
      cell.classList.add("playable");
      const hint = document.createElement("div");
      hint.className = "hint";
      cell.appendChild(hint);
    }
  }

  if (current.game_over) running = false;
  el.passBtn.classList.toggle("hidden", !current.must_pass);
  updateSettingsLock();   // bei Partieende wieder freigeben
  refreshButton();
  updateStatus();
}

function updateStatus() {
  el.status.className = "status";
  if (current.game_over) {
    const you = current.human_color;
    if (current.winner === "draw") {
      el.status.textContent = "Unentschieden.";
      el.status.classList.add("draw");
    } else if (current.winner === you) {
      el.status.textContent = "Du gewinnst! 🎉";
      el.status.classList.add("win");
      sfx.win();
    } else {
      el.status.textContent = "Die KI gewinnt.";
      el.status.classList.add("lose");
      sfx.lose();
    }
    return;
  }
  if (current.must_pass) el.status.textContent = "Kein legaler Zug – du musst passen.";
  else if (current.human_turn) el.status.textContent = "Du bist am Zug.";
  else el.status.textContent = "KI ist am Zug…";
}

function setThinking(on) {
  el.thinking.classList.toggle("hidden", !on);
}

/* ---------- Aktionen ---------- */

async function newGame() {
  const myGen = ++gen;
  busy = true;
  running = true;
  refreshButton();
  updateSettingsLock();        // Einstellungen für die Dauer der Partie sperren
  el.status.className = "status";
  el.status.textContent = "Neue Partie…";
  el.passBtn.classList.add("hidden");
  try {
    setThinking(true);
    const res = await api("/api/new_game", {
      body: JSON.stringify({
        human_color: el.color.value,
        n_simulations: Number(el.sims.value),
        temperature: Number(el.temp.value),
      }),
    });
    if (myGen !== gen) return;   // während des Ladens abgebrochen
    setThinking(false);
    current = res;
    gameId = res.game_id;
    buildGrid(res.board_size);
    await animateIntro();
    if (myGen !== gen) return;
    await animateSteps(res.steps || []);
    if (myGen !== gen) return;
    finishTurn();
  } catch (e) {
    if (myGen === gen) el.status.textContent = "Fehler: " + e.message;
  } finally {
    if (myGen === gen) { setThinking(false); busy = false; }
  }
}

// Laufende Partie abbrechen (ohne neue zu starten): entsperrt die Einstellungen.
function abortGame() {
  gen++;                        // invalidiert laufende Fetches/Animationen
  running = false;
  busy = false;
  gameId = null;
  current = null;
  humanLegal = new Set();
  setThinking(false);
  if (cells.length) { clearHints(); clearLastAI(); }
  el.passBtn.classList.add("hidden");
  updateSettingsLock();
  refreshButton();
  el.status.className = "status";
  el.status.textContent = "Partie abgebrochen – Einstellungen anpassen und neu starten.";
}

function onPrimary() {
  if (running) abortGame();
  else newGame();
}

async function sendMove(move) {
  if (busy || !gameId) return;
  const myGen = gen;
  busy = true;
  clearHints();               // Board wirkt sofort "committed"
  try {
    // 1) Menschzug anwenden und *sofort* zeigen (Server denkt hier noch nicht).
    let res = await api("/api/move", {
      body: JSON.stringify({ game_id: gameId, move: move }),
    });
    if (myGen !== gen) return;   // Partie zwischenzeitlich abgebrochen
    current = res;
    await animateSteps(res.steps || []);
    if (myGen !== gen) return;

    // 2) Erst danach die KI denken lassen und ihre Antwort animieren.
    if (!current.game_over && !current.human_turn) {
      updateStatus();          // "KI ist am Zug…"
      setThinking(true);
      res = await api("/api/ai_move", {
        body: JSON.stringify({ game_id: gameId }),
      });
      if (myGen !== gen) return;
      setThinking(false);
      current = res;
      await animateSteps(res.steps || []);
      if (myGen !== gen) return;
    }
    finishTurn();
  } catch (e) {
    if (myGen !== gen) return;
    el.status.textContent = "Fehler: " + e.message;
    if (current) finishTurn();   // Hints wiederherstellen
  } finally {
    if (myGen === gen) { setThinking(false); busy = false; }
  }
}

function onCellClick(r, c) {
  if (busy || !current || !current.human_turn) return;
  if (!humanLegal.has(r * size + c)) return;
  sendMove([r, c]);
}

/* ---------- Verdrahtung ---------- */

function syncSpeed() {
  speed = Number(el.speed.value);
  el.speedVal.textContent = speed.toFixed(2).replace(/0$/, "") + "×";
  applyAnimVars();
}

el.newGame.addEventListener("click", onPrimary);
el.passBtn.addEventListener("click", () => sendMove(null));
el.sims.addEventListener("input", () => { syncSliderLabels(); markActivePreset(); });
el.temp.addEventListener("input", () => { syncSliderLabels(); markActivePreset(); });
el.speed.addEventListener("input", syncSpeed);
el.soundToggle.addEventListener("click", toggleSound);

syncSpeed();
refreshButton();
loadConfig().catch((e) => {
  el.status.textContent = "Fehler beim Laden: " + e.message;
});
