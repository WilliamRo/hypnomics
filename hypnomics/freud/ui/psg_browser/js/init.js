// =============================================================================
// init.js — Session restore, file loading, and initialization
// =============================================================================

// (4) File loading
async function restoreLastSession(fileName) {
  const s = loadSettings();
  const name = fileName || s.lastFileName;
  if (!name) { alert('No previous session found.'); return; }
  const buf = await loadFileFromIDB(name);
  if (!buf) { alert('Cached file not found.'); return; }
  const fs = getFileSettings(name);
  lastFileName = name;
  await loadFile(buf, {
    fileName: name,
    restore: true,
    markIn: fs.markIn,
    markOut: fs.markOut,
    epoch: fs.epoch,
    fixedYmax: fs.fixedYmax,
    pinnedChannels: fs.pinnedChannels,
    isolatedChannels: fs.isolatedChannels,
    autoScaleGlobal: fs.autoScaleGlobal,
  });
}

async function loadFile(arrayBuffer, opts = {}) {
  loading.classList.add('active');
  clearEpochCache();

  try {
    if (!h5Ready) await initH5();

    const FS = h5wasm.FS;
    const File = h5wasm.File;
    const fname = '_psg_temp.h5';

    // Close previous file before loading a new one
    if (psgFile) {
      try { psgFile.close(); } catch(e) {}
      try { FS.unlink(fname); } catch(e) {}
    }

    // A new PSG invalidates any previously loaded .traj.h5 and its signals
    if (trajFile) {
      try { trajFile.close(); } catch(e) {}
      try { FS.unlink('_traj_temp.h5'); } catch(e) {}
      trajFile = null;
      lastTrajFileName = null;
    }
    trajSignals = [];
    trajLogEnabled = new Set();
    // Also drop any stale traj:: entries from saved activeChannels
    if (Array.isArray(activeChannels)) {
      activeChannels = activeChannels.filter(n => !(typeof n === 'string' && n.startsWith('traj::')));
    }
    // Paths in expandedH5Paths refer to the previous file — discard.
    expandedH5Paths = new Set();
    // Drop any hypnogram background overlay from a prior file
    hypnoTrajName = null;
    hypnoTrajRawData = null;

    FS.writeFile(fname, new Uint8Array(arrayBuffer));
    psgFile = new File(fname, 'r');

    // (4.1) Read channels
    channels = [];
    const sigGroup = psgFile.get('signals');
    if (!sigGroup || typeof sigGroup.keys !== 'function') {
      throw new Error(
        "File has no top-level 'signals/' group. " +
        "Are you sure this is a .psg.h5? (A .traj.h5 file only contains 'traj/'.)");
    }
    const chNames = sigGroup.keys();
    for (const name of chNames) {
      const ds = psgFile.get(`signals/${name}`);
      const sfreq = ds.attrs['sfreq'].value;
      const unit = ds.attrs['unit']?.value || '';
      const edf_unit = ds.attrs['edf_unit']?.value || '';
      channels.push({ name, sfreq, length: ds.shape[0], unit, edf_unit });
    }

    // (4.2) Restore saved channels or default to highest sfreq
    const maxSfreq = Math.max(...channels.map(c => c.sfreq));
    const savedCh = savedSettings.activeChannels;
    const availableNames = channels.map(c => c.name);
    if (savedCh && savedCh.some(n => availableNames.includes(n))) {
      activeChannels = savedCh.filter(n => availableNames.includes(n));
    } else {
      activeChannels = channels
        .filter(c => c.sfreq === maxSfreq)
        .map(c => c.name);
    }

    // (4.3) Compute duration and epochs
    const mainCh = channels.find(c => c.sfreq === maxSfreq);
    duration = mainCh.length / mainCh.sfreq;
    totalEpochs = Math.floor(duration / 30);

    // (4.4) Read annotations — collect all stage keys
    annotations = null;
    annoKeys = [];
    try {
      const annoGroup = psgFile.get('annotations');
      if (annoGroup) {
        const keys = annoGroup.keys();
        for (const key of keys) {
          if (key.startsWith('stage')) annoKeys.push(key);
        }
      }
    } catch(e) { console.warn('No annotations found:', e); }

    // Add local annotation keys
    try {
      const localKeys = await listLocalAnnoKeys(opts.fileName || lastFileName);
      localKeys.forEach(k => { if (!annoKeys.includes(k)) annoKeys.push(k); });
    } catch(e) {}

    // Load first stage annotation (prefer Ground-Truth)
    activeAnnoKey = annoKeys.find(k => k.includes('Ground-Truth')) || annoKeys[0] || '';
    if (activeAnnoKey) {
      await switchAnnotation(activeAnnoKey);
    }
    buildAnnoSelect();

    // (4.5) Read metadata
    const label = psgFile.attrs['label']?.value || 'Unknown';

    // (4.6) Update UI
    fileMeta.innerHTML = `
      <span>${label}</span> |
      ${formatTime(duration)} |
      ${maxSfreq} Hz |
      ${channels.length} ch
    `;

    // (4.7) Build channel toggles
    buildChannelToggles();

    // (4.8) Show viewer
    dropzone.style.display = 'none';
    viewer.classList.add('active');
    // Reveal the "Load .traj.h5" button now that a PSG is loaded
    loadTrajBtn.style.display = '';

    // (4.9) Restore session or navigate to first annotated epoch
    if (opts.fileName) lastFileName = opts.fileName;
    if (opts.restore) {
      markIn = opts.markIn ?? null;
      markOut = opts.markOut ?? null;
      fixedYmax = opts.fixedYmax ? { ...opts.fixedYmax } : {};
      pinnedChannels = opts.pinnedChannels ? { ...opts.pinnedChannels } : {};
      isolatedChannels = opts.isolatedChannels ? { ...opts.isolatedChannels } : {};
      // Restore global ymax cache and auto-scale state
      const cacheKey = 'morpheus_globalYmax_' + lastFileName;
      const cached = localStorage.getItem(cacheKey);
      globalYmax = cached ? JSON.parse(cached) : {};
      autoScaleGlobal = Object.keys(globalYmax).length > 0 && (opts.autoScaleGlobal ?? false);
      updateAutoScaleBtn();
      filterEnabled = false;
      clearFilteredData();
      updateFilterBtn();
      currentEpoch = Math.max(visibleStart(), Math.min(visibleEnd(), opts.epoch ?? 0));
      viewStartSec = currentEpoch * 30;
    } else {
      markIn = null;
      markOut = null;
      fixedYmax = {};
      pinnedChannels = {};
      isolatedChannels = {};
      globalYmax = {};
      autoScaleGlobal = false;
      updateAutoScaleBtn();
      filterEnabled = false;
      clearFilteredData();
      updateFilterBtn();
      if (annotations) {
        const firstAnnoTime = annotations.intervals[0];
        currentEpoch = Math.floor(firstAnnoTime / 30);
      } else {
        currentEpoch = 0;
      }
      viewStartSec = currentEpoch * 30;
    }
    undoStack.length = 0;
    redoStack.length = 0;

    resizeCanvases();
    drawHypnogram();
    drawWaveforms();
    updateEpochInfo();

    // (4.10) Save session to IDB (fire and forget)
    if (!opts.restore && lastFileName) {
      saveFileToIDB(arrayBuffer, lastFileName).catch(e => console.warn('IDB save failed:', e));
      saveFileSettings(lastFileName, { markIn, markOut, epoch: currentEpoch, fixedYmax: { ...fixedYmax } });
    }

    // (4.10.05) Restore H5 tree expansion state for this file so reopening
    // preserves the PSG tree's open nodes (traj nodes are restored after
    // the traj file itself loads, but live in the same Set).
    try {
      const _fs = lastFileName ? getFileSettings(lastFileName) : {};
      if (Array.isArray(_fs.expandedH5Paths)) {
        expandedH5Paths = new Set(_fs.expandedH5Paths);
      }
    } catch(_) {}

    // (4.10.1) Auto-load the .traj.h5 associated with this PSG, if any was
    // persisted via a previous session. Done for all open paths — restore,
    // drag-drop, and the Open .psg.h5 dialog — so reopening a PSG always
    // restores its companion traj without a second file dialog.
    try {
      const fs = lastFileName ? getFileSettings(lastFileName) : {};
      if (fs && fs.trajFileName) {
        const trajBuf = await loadFileFromIDB(fs.trajFileName);
        if (trajBuf) {
          await loadTrajFile(trajBuf, { fileName: fs.trajFileName });
        } else {
          console.warn('Associated traj file not found in IDB:', fs.trajFileName);
        }
      }
    } catch(e) { console.warn('Auto traj load failed:', e); }

  } catch(e) {
    console.error('Failed to load file:', e);
    alert('Failed to load file: ' + e.message);
  } finally {
    loading.classList.remove('active');
  }
}

// (4.11) Load a companion .traj.h5 file. Requires a PSG already loaded.
// The file is kept open so the H5 tree panel can render its `traj/` group
// (with checkboxes that activate each probe as a slow signal).
async function loadTrajFile(arrayBuffer, opts = {}) {
  if (!psgFile) { alert('Load a .psg.h5 first, then load .traj.h5.'); return; }
  if (!h5Ready) await initH5();

  const FS = h5wasm.FS;
  const File = h5wasm.File;
  const fname = '_traj_temp.h5';

  // Close any previously loaded traj and clear its active signals
  if (trajFile) {
    try { trajFile.close(); } catch(e) {}
    try { FS.unlink(fname); } catch(e) {}
    trajFile = null;
  }
  trajSignals = [];
  trajLogEnabled = new Set();
  activeChannels = activeChannels.filter(n => !(typeof n === 'string' && n.startsWith('traj::')));
  // Drop any hypnogram overlay from the previous traj — its name won't
  // be in the new trajSignals.
  hypnoTrajName = null;
  hypnoTrajRawData = null;
  // Evict any cached traj slices from a prior session
  try { clearEpochCache(); } catch(_) {}

  try {
    FS.writeFile(fname, new Uint8Array(arrayBuffer));
    trajFile = new File(fname, 'r');

    const rejectTraj = (msg) => {
      try { trajFile.close(); } catch(_) {}
      try { FS.unlink(fname); } catch(_) {}
      trajFile = null;
      trajSignals = [];
      alert(msg);
    };

    // (a) Must contain a top-level `traj/` group
    let trajGroup;
    try {
      trajGroup = trajFile.get('traj');
      if (!trajGroup || typeof trajGroup.keys !== 'function') throw new Error('no traj/ group');
    } catch(e) {
      rejectTraj('File does not contain a top-level `traj/` group.');
      return;
    }

    // (b) Label match — traj must belong to the currently loaded PSG.
    //     Empty label on either side is treated as unknown → skip check.
    let psgLabel = '', trajLabel = '';
    try { psgLabel = psgFile.attrs['label']?.value || ''; } catch(_) {}
    try { trajLabel = trajFile.attrs['label']?.value || ''; } catch(_) {}
    if (psgLabel && trajLabel && psgLabel !== trajLabel) {
      rejectTraj(`Label mismatch:\n  PSG:  ${psgLabel}\n  Traj: ${trajLabel}\n\nThe .traj.h5 does not belong to the loaded .psg.h5.`);
      return;
    }

    // (c) Walk traj/{channel}/{tr}s/{probe} and build trajSignals[]
    const chNames = trajGroup.keys();
    for (const ch of chNames) {
      const chGrp = trajFile.get(`traj/${ch}`);
      if (!chGrp || typeof chGrp.keys !== 'function') continue;
      for (const trKey of chGrp.keys()) {
        // trKey looks like '30s' / '10s' / '6s'
        const m = /^(\d+)s$/.exec(trKey);
        if (!m) continue;
        const tr = parseInt(m[1], 10);
        const trGrp = trajFile.get(`traj/${ch}/${trKey}`);
        if (!trGrp || typeof trGrp.keys !== 'function') continue;
        for (const pk of trGrp.keys()) {
          const ds = trajFile.get(`traj/${ch}/${trKey}/${pk}`);
          if (!ds || !ds.shape) continue;
          const length = ds.shape[0] | 0;
          trajSignals.push({
            name: `traj::${ch}::${trKey}::${pk}`,
            displayName: `${pk}@${ch}[${trKey}]`,
            ch, tr, pk,
            sfreq: 1 / tr,
            length,
            unit: '',
          });
        }
      }
    }

    lastTrajFileName = opts.fileName || '.traj.h5';
    console.log(`traj file loaded: ${lastTrajFileName}  (${trajSignals.length} probe signals)`);

    // Restore any previously-remembered active probes and log-mode flags
    // BEFORE the tree panel refresh so checkboxes render the correct state.
    // We filter against the names that actually exist in the new trajSignals
    // to avoid leaving stale entries if the traj file has changed.
    try {
      const fsRestore = lastFileName ? getFileSettings(lastFileName) : {};
      const validNames = new Set(trajSignals.map(s => s.name));

      if (Array.isArray(fsRestore.trajLogEnabled)) {
        // User has an explicit saved state — restore it verbatim.
        for (const n of fsRestore.trajLogEnabled) {
          if (validNames.has(n)) trajLogEnabled.add(n);
        }
      } else {
        // First-time load for this file — apply the default: every probe
        // whose key starts with "PR" (band power ratios like PR-DELTA_TOTAL)
        // gets log₁₀ mode on by default. User can override per-probe and
        // their choice is then remembered.
        for (const sig of trajSignals) {
          if (typeof sig.pk === 'string' && sig.pk.startsWith('PR')) {
            trajLogEnabled.add(sig.name);
          }
        }
        // Persist so next load goes through the "restore verbatim" branch
        try { saveTrajUIState(); } catch(_) {}
      }

      let addedAny = false;
      if (Array.isArray(fsRestore.activeTrajChannels)) {
        for (const n of fsRestore.activeTrajChannels) {
          if (validNames.has(n) && !activeChannels.includes(n)) {
            activeChannels.push(n);
            addedAny = true;
          }
        }
      }
      // If we actually activated traj signals, redraw the waveforms so the
      // restored slow traces show up immediately.
      if (addedAny) {
        try { refreshChannelPanel(); } catch(_) {}
        try { drawWaveforms(); } catch(_) {}
      }
    } catch(e) { console.warn('traj state restore failed:', e); }

    // Persist: cache the raw bytes in IDB and record the association on
    // the current PSG's file settings. Next time this PSG is loaded (via
    // recent history or explicit reopen), we'll auto-restore this traj.
    try {
      saveFileToIDB(arrayBuffer, lastTrajFileName).catch(e => console.warn('traj IDB save failed:', e));
    } catch(e) { console.warn('traj IDB save failed (sync):', e); }
    if (lastFileName) {
      try { saveFileSettings(lastFileName, { trajFileName: lastTrajFileName }); } catch(e) { console.warn(e); }
    }

    // If the H5 tree panel is open, refresh it so the traj section + checkboxes appear
    const panel = document.getElementById('sidePanel');
    if (panel && panel.classList.contains('active')) {
      try { cmdShowH5(); } catch(e) { console.warn(e); }
    }

    // Transient success hint
    try { showToast(`Loaded .traj.h5 — ${lastTrajFileName} (${trajSignals.length} probes)`); } catch(_) {}
  } catch(e) {
    console.error('Failed to load .traj.h5:', e);
    alert('Failed to load .traj.h5: ' + e.message);
  }
}

// (11) Init
if (fontDelta !== 0) applyFontDelta();
initH5().then(() => {
  console.log('h5wasm ready');
  // Build recent files list
  const s = loadSettings();
  const recent = s.recentFiles || {};
  const names = Object.keys(recent);
  if (names.length > 0) {
    const list = document.getElementById('recentList');
    list.style.display = '';
    // Show most recent first (last key in object = most recently opened)
    const sorted = [...names].reverse();
    sorted.forEach(name => {
      const item = document.createElement('div');
      item.className = 'recent-item';
      const nameSpan = document.createElement('span');
      nameSpan.className = 'ri-name';
      nameSpan.textContent = name;
      const meta = document.createElement('span');
      meta.className = 'ri-meta';
      const fs = recent[name];
      const ep = fs.epoch != null ? `ep ${fs.epoch + 1}` : '';
      const marks = (fs.markIn != null || fs.markOut != null) ? ' \u2702' : '';
      const fixed = fs.fixedYmax && Object.keys(fs.fixedYmax).length > 0 ? ' \uD83D\uDCCC' : '';
      meta.textContent = [ep, marks, fixed].filter(Boolean).join('');
      item.appendChild(nameSpan);
      item.appendChild(meta);
      item.onclick = () => restoreLastSession(name);
      list.appendChild(item);
    });
  }
}).catch(e => {
  console.error('Failed to load h5wasm:', e);
});
