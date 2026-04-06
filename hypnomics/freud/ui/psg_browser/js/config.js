// =============================================================================
// config.js — Signal type detection, stage mapping, and formatting utilities
// =============================================================================

// Signal type detection and ordering (Mucha palette)
// ymaxGroup: channels sharing the same group get the same ymax
const SIGNAL_TYPES = [
  { type: 'EOG', order: 0, ymaxGroup: 'EEG_EOG', speed: 'fast', pattern: /eog|e1|e2|^E1|^E2/i,
    light: '#1b4965', dark: '#5ba3d9' },
  { type: 'EEG', order: 1, ymaxGroup: 'EEG_EOG', speed: 'fast', pattern: /eeg|fpz|fz|cz|pz|oz|f3|f4|c3|c4|o1|o2|a1|a2|^F\d|^C\d|^O\d|^P\d/i,
    light: '#2b2b2b', dark: '#d4cfc8' },
  { type: 'EMG', order: 2, ymaxGroup: 'EMG', speed: 'fast', pattern: /emg|chin|subment/i,
    light: '#c68b3f', dark: '#daa654' },
  { type: 'ECG', order: 3, ymaxGroup: 'ECG', speed: 'fast', pattern: /ecg|ekg|heart/i,
    light: '#8b3a3a', dark: '#c47070' },
  { type: 'Resp', order: 4, ymaxGroup: 'Resp', speed: 'slow', pattern: /nasal|therm|thor|abdo|flow|press|resp|snore|sum/i,
    light: '#3e7a5e', dark: '#6db88e' },
  { type: 'SpO2', order: 5, ymaxGroup: 'SpO2', speed: 'slow', pattern: /spo2|sao2|oxygen/i,
    light: '#1b4965', dark: '#5ba3d9' },
  { type: 'Limb', order: 6, ymaxGroup: 'Limb', speed: 'slow', pattern: /leg|limb|plm/i,
    light: '#6b4c6e', dark: '#a07ca3' },
  { type: 'Other', order: 7, ymaxGroup: 'Other', speed: 'slow', pattern: /./,
    light: '#666666', dark: '#999999' },
];

function getSignalType(chName) {
  for (const st of SIGNAL_TYPES) {
    if (st.pattern.test(chName)) return st;
  }
  return SIGNAL_TYPES[SIGNAL_TYPES.length - 1];
}

function getYmaxGroup(chName) {
  return getSignalType(chName).ymaxGroup;
}

function getSignalSpeed(chName) {
  return getSignalType(chName).speed;
}

function getSignalColor(chName) {
  const st = getSignalType(chName);
  return darkMode ? st.dark : st.light;
}

function getSignalOrder(chName) {
  return getSignalType(chName).order;
}

// Legacy fallback
const TRACE_COLORS = [
  '#00ff88', '#ff6b9d', '#ffd700', '#a78bfa',
  '#fb923c', '#34d399', '#f87171'
];

// Dynamic stage mapping — built from label_names in h5 file
// Canonical AASM rows: W=0, R=1, N1=2, N2=3, N3=4
// Map stage name -> AASM row. Matched by patterns, not exact strings.
const STAGE_NAME_PATTERNS = [
  { row: 0, patterns: [/\bw(ake)?\b/i, /\bwakefulness\b/i] },
  { row: 1, patterns: [/\br(em)?\b/i] },
  { row: 2, patterns: [/\bn1\b/i, /\bs(tage)?\s*1\b/i] },
  { row: 3, patterns: [/\bn2\b/i, /\bs(tage)?\s*2\b/i] },
  { row: 4, patterns: [/\bn3\b/i, /\bs(tage)?\s*3\b/i, /\bs(tage)?\s*4\b/i, /\bsws\b/i, /\bn4\b/i, /\bs4\b/i] },
];

function matchStageRow(name) {
  for (const { row, patterns } of STAGE_NAME_PATTERNS) {
    for (const p of patterns) {
      if (p.test(name)) return row;
    }
  }
  return undefined; // unknown -> gap
}
const STAGE_COLOR_BY_ROW = {
  0: '#34d399', // W — green
  1: '#f472b6', // REM — pink
  2: '#7dd3fc', // N1
  3: '#3b82f6', // N2
  4: '#1e3a5f', // N3
};
const STAGE_SHORT_BY_ROW = { 0: 'W', 1: 'R', 2: 'N1', 3: 'N2', 4: 'N3' };
const STAGE_ROWS = 5;

// Fallback mappings for common conventions when label_names is absent
const FALLBACK_MAPS = {
  // R&K / SC convention: 0=W, 1=N1, 2=N2, 3=N3, 4=S4, 5=REM, 6=MT, 7=?
  rk: { 0: 'w', 1: 'n1', 2: 'n2', 3: 'n3', 4: 's4', 5: 'rem', 6: 'mt', 7: 'unknown' },
  // AASM convention: 0=W, 1=N1, 2=N2, 3=N3, 4=REM, 5=Unknown
  aasm: { 0: 'wake', 1: 'n1', 2: 'n2', 3: 'n3', 4: 'rem', 5: 'unknown' },
};

function formatTime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return `${h}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}

function formatYScale(val) {
  const abs = Math.abs(val);
  if (abs < 0.001) return (val * 1e6).toFixed(0) + ' nV';
  if (abs < 1) return (val * 1e3).toFixed(0) + ' \u00b5V';
  if (abs < 1000) return val.toFixed(0) + ' \u00b5V';
  if (abs < 1e6) return (val / 1e3).toFixed(1) + ' mV';
  return val.toFixed(0);
}

function niceRound(val) {
  if (!val || !isFinite(val) || val <= 0) return 1;
  const mag = Math.pow(10, Math.floor(Math.log10(val)));
  const norm = val / mag;
  if (norm <= 1) return mag;
  if (norm <= 2) return 2 * mag;
  if (norm <= 5) return 5 * mag;
  return 10 * mag;
}
