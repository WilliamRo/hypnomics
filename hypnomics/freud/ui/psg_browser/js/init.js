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

    FS.writeFile(fname, new Uint8Array(arrayBuffer));
    psgFile = new File(fname, 'r');

    // (4.1) Read channels
    channels = [];
    const sigGroup = psgFile.get('signals');
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

  } catch(e) {
    console.error('Failed to load file:', e);
    alert('Failed to load file: ' + e.message);
  } finally {
    loading.classList.remove('active');
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
    // Show most recent first (lastFileName on top, then others)
    const sorted = [...names].sort((a, b) => {
      if (a === s.lastFileName) return -1;
      if (b === s.lastFileName) return 1;
      return 0;
    });
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
