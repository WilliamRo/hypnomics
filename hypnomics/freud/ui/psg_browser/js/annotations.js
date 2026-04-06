// =============================================================================
// annotations.js — Stage annotation management
// =============================================================================

// AASM stage shortcuts: w=Wake, 1=N1, 2=N2, 3=N3, r=REM, 0=Unknown
const ANNO_KEY_MAP = {
  'w': 'Wake', 'W': 'Wake',
  '1': 'N1',
  '2': 'N2',
  '3': 'N3',
  'r': 'REM', 'R': 'REM',
  '0': 'Unknown',
};
const ANNO_LABEL_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM', 'Unknown'];

// Stage mode: when active, keyboard shortcuts annotate epochs
let stageMode = false;

function isStageEditable() {
  return activeAnnoKey && isLocalAnnotation(activeAnnoKey);
}

function updateStageModeBtn() {
  const btn = document.getElementById('stageModeBtn');
  if (!psgFile || !isStageEditable()) {
    btn.style.display = 'none';
    stageMode = false;
    return;
  }
  btn.style.display = '';
  btn.classList.toggle('active', stageMode);
  btn.textContent = stageMode ? '✏ Staging...' : '✏ Stage';
}

function toggleStageMode() {
  if (!isStageEditable()) return;
  stageMode = !stageMode;
  updateStageModeBtn();
}

// Current active annotation key (e.g., "stage Ground-Truth" or "stage MyAnno")
let activeAnnoKey = '';
// Local annotations: { fileName: { annoKey: { labels: Int32Array, modified: bool } } }
// Stored in IDB alongside file data

// All available annotation keys for the current file
let annoKeys = [];

// Get the local annotation store key
function localAnnoIDBKey(fileName, annoKey) {
  return `anno:${fileName}:${annoKey}`;
}

// Build the annotation selector dropdown
function buildAnnoSelect() {
  const sel = document.getElementById('annoSelect');
  sel.innerHTML = '';
  sel.style.display = psgFile ? '' : 'none';
  if (!psgFile) return;

  annoKeys.forEach(key => {
    const opt = document.createElement('option');
    opt.value = key;
    const shortName = key.replace('stage ', '');
    const isLocal = isLocalAnnotation(key);
    opt.textContent = shortName + (isLocal ? ' *' : '');
    sel.appendChild(opt);
  });

  // "New" option
  const newOpt = document.createElement('option');
  newOpt.value = '__new__';
  newOpt.textContent = '+ New annotation...';
  sel.appendChild(newOpt);

  sel.value = activeAnnoKey;
  updateStageModeBtn();
}

// Check if an annotation is local (not from the h5 file)
function isLocalAnnotation(key) {
  if (!psgFile) return false;
  try {
    const annoGroup = psgFile.get('annotations');
    if (!annoGroup) return true;
    const keys = annoGroup.keys();
    return !keys.includes(key);
  } catch(e) { return true; }
}

// Get labels for the active annotation
function getActiveLabels() {
  return annotations ? annotations.labels : null;
}

// Switch to a different annotation
async function switchAnnotation(key) {
  stageMode = false; // reset on any switch
  if (key === '__new__') {
    const name = prompt('New annotation name:');
    if (!name || !name.trim()) {
      document.getElementById('annoSelect').value = activeAnnoKey;
      return;
    }
    const annoKey = 'stage ' + name.trim();
    if (annoKeys.includes(annoKey)) {
      alert('Annotation "' + name.trim() + '" already exists.');
      document.getElementById('annoSelect').value = activeAnnoKey;
      return;
    }

    // Create empty annotation (all Unknown)
    const nEpochs = totalEpochs;
    const labels = new Int32Array(nEpochs);
    labels.fill(ANNO_LABEL_NAMES.indexOf('Unknown')); // 5 = Unknown
    const intervals = new Float64Array(nEpochs * 2);
    for (let i = 0; i < nEpochs; i++) {
      intervals[i * 2] = i * 30;
      intervals[i * 2 + 1] = (i + 1) * 30;
    }

    // Store locally
    const annoData = { labels: Array.from(labels), intervals: Array.from(intervals), labelNames: ANNO_LABEL_NAMES, modified: true };
    await saveLocalAnno(lastFileName, annoKey, annoData);

    annoKeys.push(annoKey);
    activeAnnoKey = annoKey;
    annotations = {
      intervals: intervals,
      labels: labels,
      labelNames: ANNO_LABEL_NAMES,
      key: annoKey,
    };
    buildStageMap(ANNO_LABEL_NAMES);
    updateHypnoYAxis();
    buildAnnoSelect();
    drawHypnogram();
    drawWaveforms();
    updateEpochInfo();
    return;
  }

  activeAnnoKey = key;

  // Try loading from local first, then from h5
  const local = await loadLocalAnno(lastFileName, key);
  if (local) {
    annotations = {
      intervals: local.intervals instanceof Float64Array ? local.intervals : new Float64Array(local.intervals),
      labels: local.labels instanceof Int32Array ? local.labels : new Int32Array(local.labels),
      labelNames: local.labelNames || ANNO_LABEL_NAMES,
      key: key,
    };
    buildStageMap(annotations.labelNames);
  } else {
    // Read from h5
    try {
      const grp = psgFile.get(`annotations/${key}`);
      const intervals = psgFile.get(`annotations/${key}/intervals`).value;
      const labels = psgFile.get(`annotations/${key}/labels`).value;
      let labelNames = [];
      try { labelNames = grp.attrs['label_names'].value; } catch(e) {}
      annotations = { intervals, labels, labelNames, key };
      buildStageMap(labelNames);
    } catch(e) {
      console.error('Failed to load annotation:', key, e);
      return;
    }
  }

  updateHypnoYAxis();
  buildAnnoSelect();
  drawHypnogram();
  drawWaveforms();
  updateEpochInfo();
}

// Annotate the current epoch with a stage
function annotateCurrentEpoch(stageNameStr) {
  if (!annotations || !activeAnnoKey) return;
  if (!isLocalAnnotation(activeAnnoKey)) {
    alert('Cannot modify "' + activeAnnoKey.replace('stage ', '') + '" (read-only from h5 file).\nCreate a new annotation to edit.');
    return;
  }

  const labelIdx = ANNO_LABEL_NAMES.indexOf(stageNameStr);
  if (labelIdx < 0) return;

  // Find the epoch index for current epoch
  const t = currentEpoch * 30;
  for (let i = 0; i < annotations.labels.length; i++) {
    const t0 = annotations.intervals[i * 2];
    const t1 = annotations.intervals[i * 2 + 1];
    if (t >= t0 && t < t1) {
      annotations.labels[i] = labelIdx;
      break;
    }
  }

  // Save locally
  saveLocalAnno(lastFileName, activeAnnoKey, {
    labels: Array.from(annotations.labels),
    intervals: Array.from(annotations.intervals),
    labelNames: ANNO_LABEL_NAMES,
    modified: true,
  });

  // Auto-advance to next epoch
  navigate(1);
  drawHypnogram();
  drawWaveforms();
  updateEpochInfo();
}

// Save annotation to IDB
async function saveLocalAnno(fileName, annoKey, data) {
  const db = await openIDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readwrite');
    tx.objectStore(IDB_STORE).put(data, localAnnoIDBKey(fileName, annoKey));
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

// Load annotation from IDB
async function loadLocalAnno(fileName, annoKey) {
  const db = await openIDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readonly');
    const req = tx.objectStore(IDB_STORE).get(localAnnoIDBKey(fileName, annoKey));
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

// List local annotation keys for a file
async function listLocalAnnoKeys(fileName) {
  const db = await openIDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readonly');
    const req = tx.objectStore(IDB_STORE).getAllKeys();
    req.onsuccess = () => {
      const prefix = `anno:${fileName}:`;
      const keys = req.result
        .filter(k => typeof k === 'string' && k.startsWith(prefix))
        .map(k => k.slice(prefix.length));
      resolve(keys);
    };
    req.onerror = () => reject(req.error);
  });
}

// Export annotation as .anno.json download
// (JSON is reliable from file:// protocol; can be converted to h5 via Python later)
async function exportAnnotation(annoKey) {
  const local = await loadLocalAnno(lastFileName, annoKey);
  if (!local) {
    alert('Only local annotations can be exported.');
    return;
  }

  const shortName = annoKey.replace('stage ', '');
  const exportData = {
    anno_key: annoKey,
    source_file: lastFileName,
    label_names: local.labelNames,
    labels: Array.from(local.labels),
    intervals: [],
  };
  // Convert flat intervals to [N, 2] pairs
  for (let i = 0; i < local.labels.length; i++) {
    exportData.intervals.push([local.intervals[i * 2], local.intervals[i * 2 + 1]]);
  }

  const json = JSON.stringify(exportData, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = lastFileName.replace('.psg.h5', '') + '.' + shortName + '.anno.json';
  a.click();
  URL.revokeObjectURL(url);
}
