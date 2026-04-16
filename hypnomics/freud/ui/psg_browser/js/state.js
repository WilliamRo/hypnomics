// =============================================================================
// state.js — Stage map, settings persistence, state variables, themes, layout
// =============================================================================

// Built per-file from label_names: { labelInt: { row, color, name } }
let stageMap = {};

function buildStageMap(labelNames) {
  stageMap = {};

  // Try to use label_names from the file
  let names = [];
  if (labelNames && labelNames.length > 0) {
    for (let i = 0; i < labelNames.length; i++) {
      names.push((typeof labelNames[i] === 'string' ? labelNames[i] : String(labelNames[i])).trim());
    }
  }

  // If no label_names, detect convention from unique labels
  if (names.length === 0 && annotations && annotations.labels) {
    const unique = new Set(annotations.labels);
    const maxLabel = Math.max(...unique);
    const fb = maxLabel >= 5 ? FALLBACK_MAPS.rk : FALLBACK_MAPS.aasm;
    for (const [k, v] of Object.entries(fb)) {
      names[parseInt(k)] = v;
    }
  }

  for (let i = 0; i < names.length; i++) {
    if (!names[i]) continue;
    const row = matchStageRow(names[i]);
    if (row !== undefined) {
      stageMap[i] = { row, color: STAGE_COLOR_BY_ROW[row], name: STAGE_SHORT_BY_ROW[row] };
    }
  }
}

function stageRow(labelInt) {
  return stageMap[labelInt]?.row;
}
function stageColor(labelInt) {
  return stageMap[labelInt]?.color;
}
function stageName(labelInt) {
  return stageMap[labelInt]?.name || '?';
}

function updateHypnoYAxis() {
  const el = document.getElementById('hypnoYAxis');
  el.innerHTML = '';
  el.style.position = 'relative';
  const h = hypnoHeight;
  const pad = 4;
  const plotH = h - pad * 2;
  for (let r = 0; r < STAGE_ROWS; r++) {
    const span = document.createElement('span');
    span.style.color = STAGE_COLOR_BY_ROW[r] || 'var(--text-dim)';
    span.textContent = STAGE_SHORT_BY_ROW[r] || '';
    span.style.position = 'absolute';
    span.style.right = '0';
    const y = pad + (r / (STAGE_ROWS - 1)) * plotH;
    span.style.top = y + 'px';
    span.style.transform = 'translateY(-50%)';
    el.appendChild(span);
  }
}

// (0) Settings persistence
const STORAGE_KEY = 'morpheus_settings';
function loadSettings() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; } catch(e) { return {}; }
}
function saveSettings(patch) {
  const s = { ...loadSettings(), ...patch };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
}

// (0.1) IndexedDB for PSG file cache
const IDB_NAME = 'morpheus_db', IDB_STORE = 'files';
function openIDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(IDB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(IDB_STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
async function saveFileToIDB(arrayBuffer, key) {
  const db = await openIDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readwrite');
    tx.objectStore(IDB_STORE).put(arrayBuffer, key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}
async function loadFileFromIDB(key) {
  const db = await openIDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readonly');
    const req = tx.objectStore(IDB_STORE).get(key);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

// Per-file settings (recent files, max 20)
const MAX_RECENT = 100;
function getFileSettings(fileName) {
  const s = loadSettings();
  return (s.recentFiles || {})[fileName] || {};
}
function saveFileSettings(fileName, patch) {
  const s = loadSettings();
  const recent = s.recentFiles || {};
  // Move to end (most recent) by re-inserting
  const existing = recent[fileName] || {};
  delete recent[fileName];
  recent[fileName] = { ...existing, ...patch };
  // Evict oldest if over limit
  const keys = Object.keys(recent);
  while (keys.length > MAX_RECENT) {
    delete recent[keys.shift()];
  }
  saveSettings({ recentFiles: recent, lastFileName: fileName });
}
let _saveTimer = null;
function saveCurrentFileState() {
  if (!lastFileName) return;
  if (_saveTimer) clearTimeout(_saveTimer);
  _saveTimer = setTimeout(() => {
    saveFileSettings(lastFileName, {
      markIn, markOut, epoch: currentEpoch, fixedYmax: { ...fixedYmax },
      pinnedChannels: { ...pinnedChannels }, isolatedChannels: { ...isolatedChannels },
      autoScaleGlobal,
    });
  }, 500);
}

// (1) State
let psgFile = null;
let channels = [];
let activeChannels = [];
let annotations = null;
let currentEpoch = 0;
let viewStartSec = 0; // precise start time for fast view
let totalEpochs = 0;
const savedSettings = loadSettings();
let gain = 20; // kept for backward compat in drawTraces
let autoScaleGlobal = false; // default OFF
let globalYmax = {}; // { chName: p99_value } — cached per file
let duration = 0;

// Mark In/Out
let markIn = null;   // epoch index or null
let markOut = null;  // epoch index or null
let ctxMenuEpoch = 0; // epoch at right-click position

// Fixed ymax per group: { groupName: value } or null for std-based
let fixedYmax = {};  // e.g., { EEG_EOG: 75.0 }

// Per-channel pin/isolate state
let pinnedChannels = {};   // { chName: ymaxValue }
let isolatedChannels = {}; // { chName: true }

// AASM bandpass filter state (session-only, recomputed on each file load)
let filterEnabled = false;
let filteredData = {};          // { chName: Float32Array }
let filteredDataMeta = {};      // { chName: { p99 } }
let filterMemoryBytes = 0;

function clearFilteredData() {
  filteredData = {};
  filteredDataMeta = {};
  filterMemoryBytes = 0;
}

// Data cache: avoids re-reading HDF5 on every redraw
const epochCache = {}; // { "chName:epoch": { data, mean, std } }
function clearEpochCache() {
  for (const k in epochCache) delete epochCache[k];
  _slowCacheKey = ''; // invalidate slow render cache
}

// Undo/Redo
const undoStack = [];
const redoStack = [];
function pushUndo() {
  undoStack.push({ markIn, markOut });
  redoStack.length = 0; // clear redo on new action
}
function undo() {
  if (undoStack.length === 0) return;
  redoStack.push({ markIn, markOut });
  const state = undoStack.pop();
  markIn = state.markIn;
  markOut = state.markOut;
  currentEpoch = Math.max(visibleStart(), Math.min(visibleEnd(), currentEpoch));
  viewStartSec = currentEpoch * 30;
  drawHypnogram(); drawWaveforms(); updateEpochInfo();
}
function redo() {
  if (redoStack.length === 0) return;
  undoStack.push({ markIn, markOut });
  const state = redoStack.pop();
  markIn = state.markIn;
  markOut = state.markOut;
  currentEpoch = Math.max(visibleStart(), Math.min(visibleEnd(), currentEpoch));
  viewStartSec = currentEpoch * 30;
  drawHypnogram(); drawWaveforms(); updateEpochInfo();
}
let h5Ready = false;

// (2) DOM refs
const dropzone = document.getElementById('dropzone');
const viewer = document.getElementById('viewer');
const loading = document.getElementById('loading');
const fileInput = document.getElementById('fileInput');
const openBtn = document.getElementById('openBtn');
const loadTrajBtn = document.getElementById('loadTrajBtn');
const trajInput = document.getElementById('trajInput');
let trajFile = null;
let lastTrajFileName = null;
// Trajectory "signals" derived from .traj.h5 — parallel to `channels`.
// Each entry: { name (unique id `traj::<ch>::<tr>s::<pk>`), displayName,
//               ch, tr, pk, sfreq (=1/tr), length (= n_epochs), unit: '' }
let trajSignals = [];
// Names of traj signals for which the log₁₀ transform is currently active.
// When present, readTrajData returns log10(max(v, 1e-12)) instead of raw v,
// and computeTrajGlobalYmax / drawTraces operate in log space.
let trajLogEnabled = new Set();

// Hypnogram background overlay: a single whole-night traj rendered in gray
// behind the staircase, scaled so that the p_margin / (100-p_margin)
// percentiles align with the N3 and W rows respectively.
let hypnoTrajName = null;               // which traj is overlaid (null = none)
let hypnoTrajRawData = null;            // Float32Array (full night, log-applied if the sig has log mode)
let hypnoTrajTr = 0;                    // time resolution in seconds
let hypnoTrajPLow = 0;                  // percentile at margin%
let hypnoTrajPHigh = 1;                 // percentile at (100 - margin)%
// User preferences (persist via saveSettings)
let hypnoTrajPctMargin = savedSettings.hypnoTrajPctMargin ?? 1;  // 1–5
let hypnoTrajInvert = savedSettings.hypnoTrajInvert ?? false;    // flip W↔N3 mapping

// Show/hide hypnogram staircase (toggle in hypnogram label bar)
let showHypnogram = savedSettings.showHypnogram ?? true;

// Persist traj-related UI state (which probes are active, which have log
// mode on, and which h5 tree nodes are expanded) to the current PSG's
// per-file settings. Called from the checkbox / log-toggle / tree-expand
// handlers so loading a PSG via recent history restores everything.
function saveTrajUIState() {
  if (!lastFileName) return;
  try {
    saveFileSettings(lastFileName, {
      activeTrajChannels: activeChannels.filter(
        n => typeof n === 'string' && n.startsWith('traj::')),
      trajLogEnabled: Array.from(trajLogEnabled),
      expandedH5Paths: Array.from(expandedH5Paths),
    });
  } catch(_) {}
}

// Transient toast notification. Fades in, holds for `ms`, then fades out.
// Non-blocking; multiple calls stack vertically.
function showToast(msg, ms = 2200) {
  const el = document.createElement('div');
  el.textContent = msg;
  el.style.cssText =
    "position:fixed;top:62px;left:50%;transform:translateX(-50%);" +
    "background:var(--bg-panel);border:1px solid var(--accent);" +
    "color:var(--text);font-family:'JetBrains Mono',monospace;font-size:12px;" +
    "padding:8px 16px;border-radius:4px;z-index:9999;pointer-events:none;" +
    "box-shadow:0 4px 14px rgba(0,0,0,0.35);opacity:0;" +
    "transition:opacity 0.3s ease";
  document.body.appendChild(el);
  // Stack vertically if there are already visible toasts
  const existing = document.querySelectorAll('[data-morpheus-toast]');
  el.dataset.morpheusToast = '1';
  el.style.top = (62 + existing.length * 36) + 'px';
  requestAnimationFrame(() => { el.style.opacity = '1'; });
  setTimeout(() => {
    el.style.opacity = '0';
    setTimeout(() => { try { el.remove(); } catch(_) {} }, 400);
  }, ms);
}
// H5 tree panel expansion memory. Keys are arbitrary path strings
// ("psg:/annotations/stage Ground-Truth", "traj/EEG Fpz-Cz/6s", etc.)
// identifying nodes that should be drawn open on the next panel rebuild.
// Cleared when a new .psg.h5 is loaded (paths are file-specific).
let expandedH5Paths = new Set();
const fileMeta = document.getElementById('fileMeta');
const hypnogramCanvas = document.getElementById('hypnogramCanvas');
const waveformCanvas = document.getElementById('waveformCanvas');
const slowCanvas = document.getElementById('slowCanvas');
const waveformDivider = document.getElementById('waveformDivider');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
const epochInfo = document.getElementById('epochInfo');
// channelToggles removed — now using channel panel
const hypnoTime = document.getElementById('hypnoTime'); // may be null (removed)

// (2.1) Color themes
const THEMES = {
  Midnight: {
    '--bg-deep': '#0a0e1a', '--bg-panel': '#0d1220', '--bg-surface': '#111827',
    '--bg-waveform': '#0a0e1a',
    '--border': '#1e293b', '--text': '#e2e8f0', '--text-dim': '#64748b', '--accent': '#00d4ff',
  },
  Clinical: {
    '--bg-deep': '#ffffff', '--bg-panel': '#ffffff', '--bg-surface': '#f5f5f5',
    '--bg-waveform': '#fefce8',
    '--border': '#cccccc', '--text': '#1a1a1a', '--text-dim': '#666666', '--accent': '#0066cc',
  },
};

let darkMode = savedSettings.darkMode === true; // default light

function applyTheme(dark) {
  const vars = dark ? THEMES.Midnight : THEMES.Clinical;
  const root = document.documentElement;
  for (const [k, v] of Object.entries(vars)) root.style.setProperty(k, v);
  document.getElementById('themeToggle').textContent = dark ? '\uD83C\uDF1A' : '\uD83C\uDF1D';
  document.getElementById('themeModeLabel').textContent = dark ? 'Dark Mode' : 'Light Mode';
  document.querySelector('link[rel="icon"]').href =
    "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>" +
    (dark ? '\uD83C\uDF18' : '\uD83C\uDF16') + "</text></svg>";
  darkMode = dark;
  saveSettings({ darkMode: dark });
}

// Window durations
const FAST_WINDOW_OPTIONS = [5, 10, 15, 30];
let fastWindowSec = savedSettings.fastWindowSec ?? 30;
let slowWindowSec = savedSettings.slowWindowSec ?? 300;

// Font size adjust: +/- 1px on all text elements
let fontDelta = savedSettings.fontDelta ?? 0;

const FONT_DELTA_MIN = -4, FONT_DELTA_MAX = 8;

function applyFontDelta() {
  try {
    document.querySelectorAll('*').forEach(el => {
      if (el.tagName === 'CANVAS' || el.tagName === 'SVG' || el.tagName === 'SCRIPT' || el.tagName === 'STYLE') return;
      if (!el.dataset.origFs) {
        el.dataset.origFs = parseInt(getComputedStyle(el).fontSize) || 0;
      }
      const orig = parseInt(el.dataset.origFs);
      if (orig > 0) {
        el.style.fontSize = Math.max(6, orig + fontDelta) + 'px';
      }
    });
  } catch(e) { console.warn('applyFontDelta error:', e); }
}

function adjustFontDelta(delta) {
  fontDelta = Math.max(FONT_DELTA_MIN, Math.min(FONT_DELTA_MAX, fontDelta + delta));
  applyFontDelta();
  saveSettings({ fontDelta });
}

// Resize constants and state
const HYPNO_MIN = 72, HYPNO_MAX = 128;
let hypnoHeight = savedSettings.hypnoHeight ?? 72;

const LABEL_MIN = 72, LABEL_MAX = 144;
let labelWidth = savedSettings.labelWidth ?? 72;

function resizeCanvases() {
  // Label panel width
  document.documentElement.style.setProperty('--label-width', labelWidth + 'px');

  // Hypnogram — set CSS var so body, y-axis, and canvas all share the same height
  document.documentElement.style.setProperty('--hypno-height', hypnoHeight + 'px');
  const hBody = document.querySelector('.hypnogram-body');
  const hypnoCanvasW = hBody.clientWidth - 28;
  hypnogramCanvas.width = hypnoCanvasW;
  hypnogramCanvas.height = hypnoHeight;

  // Time axis (same width as hypnogram canvas)
  const taCanvas = document.getElementById('timeAxisCanvas');
  taCanvas.width = hypnoCanvasW;
  taCanvas.height = 14;

  // Waveform — both canvases share the wrap's width
  const wrap = document.getElementById('waveformWrap');
  const wrapW = wrap.clientWidth;
  // Heights determined by flex — need to read after layout
  waveformCanvas.width = wrapW;
  waveformCanvas.height = waveformCanvas.clientHeight || wrap.clientHeight;
  slowCanvas.width = wrapW;
  slowCanvas.height = slowCanvas.clientHeight || 0;
}

// DOM refs for auto-scale (gain slider removed)
const autoScaleBtn = document.getElementById('autoScaleBtn');

// Navigation helpers
function visibleStart() { return markIn ?? 0; }
function visibleEnd() { return markOut ?? (totalEpochs - 1); }

// Label state
let prevLabelKey = '';
let currentLabelData = []; // live reference for double-click handlers

// Last file name
let lastFileName = '';
