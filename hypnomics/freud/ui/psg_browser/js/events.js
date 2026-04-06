// =============================================================================
// events.js — Epoch throttle, navigation, and ALL event listeners
// =============================================================================

// Epoch navigation throttle: 24fps cap for epoch-changing operations
const EPOCH_FRAME_MS = 1000 / 24;
let lastEpochRender = 0;
let epochRenderPending = false;

function renderEpoch() {
  drawHypnogram();
  drawWaveforms();
  updateEpochInfo();
  lastEpochRender = Date.now();
  epochRenderPending = false;
}

function scheduleEpochRender() {
  if (epochRenderPending) return;
  const elapsed = Date.now() - lastEpochRender;
  if (elapsed >= EPOCH_FRAME_MS) {
    renderEpoch();
  } else {
    epochRenderPending = true;
    setTimeout(renderEpoch, EPOCH_FRAME_MS - elapsed);
  }
}

function navigate(delta) {
  currentEpoch = Math.max(visibleStart(), Math.min(visibleEnd(), currentEpoch + delta));
  scheduleEpochRender();
  saveCurrentFileState();
}

function navigateToTime(seconds) {
  currentEpoch = Math.max(visibleStart(), Math.min(visibleEnd(), Math.floor(seconds / 30)));
  scheduleEpochRender();
}

function updateEpochInfo() {
  const t = currentEpoch * 30;
  let stageStr = '';
  if (annotations) {
    const labels = annotations.labels;
    const intervals = annotations.intervals;
    for (let i = 0; i < labels.length; i++) {
      const t0 = intervals[i * 2];
      const t1 = intervals[i * 2 + 1];
      if (t >= t0 && t < t1) {
        stageStr = ` | ${stageName(labels[i])}`;
        break;
      }
    }
  }
  epochInfo.innerHTML = `Epoch <span>${currentEpoch + 1}</span> / ${totalEpochs} | ${formatTime(t)}${stageStr}`;
  hypnoTime.textContent = formatTime(t) + ' - ' + formatTime(t + 30);
}

// --- Apply theme on load ---
applyTheme(darkMode);

// --- Config button toggle ---
document.getElementById('configBtn').onclick = () => {
  document.getElementById('configPanel').classList.toggle('active');
};

// --- Theme toggle ---
document.getElementById('themeToggle').onclick = () => {
  applyTheme(!darkMode);
  buildChannelToggles();
  drawHypnogram();
  drawWaveforms();
};

// --- Slow window selector ---
const slowSelect = document.getElementById('slowWindowSelect');
slowSelect.value = slowWindowSec;
slowSelect.onchange = () => {
  slowWindowSec = parseInt(slowSelect.value);
  saveSettings({ slowWindowSec });
  clearEpochCache();
  buildChannelToggles();
  drawWaveforms();
};

// --- Click outside to close panels ---
document.addEventListener('click', (e) => {
  const configPanel = document.getElementById('configPanel');
  const configBtn = document.getElementById('configBtn');
  if (configPanel.classList.contains('active') &&
      !configPanel.contains(e.target) && !configBtn.contains(e.target)) {
    configPanel.classList.remove('active');
  }
  const helpPanel = document.getElementById('helpPanel');
  if (helpPanel.classList.contains('active') && !helpPanel.contains(e.target)) {
    helpPanel.classList.remove('active');
  }
});

// --- File open / drop ---
openBtn.onclick = () => fileInput.click();
fileInput.onchange = async (e) => {
  const file = e.target.files[0];
  if (file) loadFile(await file.arrayBuffer(), { fileName: file.name });
};

// Drag and drop
dropzone.ondragover = (e) => { e.preventDefault(); dropzone.classList.add('active'); };
dropzone.ondragleave = () => dropzone.classList.remove('active');
dropzone.ondrop = async (e) => {
  e.preventDefault();
  dropzone.classList.remove('active');
  const file = e.dataTransfer.files[0];
  if (file) loadFile(await file.arrayBuffer(), { fileName: file.name });
};

// Also allow drop on the whole body
document.body.ondragover = (e) => e.preventDefault();
document.body.ondrop = async (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file) loadFile(await file.arrayBuffer(), { fileName: file.name });
};

// --- Hypnogram resize ---
const hypnogramResize = document.getElementById('hypnogramResize');
let resizing = false, resizeStartY = 0, resizeStartH = 0;
hypnogramResize.onmousedown = (e) => {
  resizing = true;
  resizeStartY = e.clientY;
  resizeStartH = hypnoHeight;
  e.preventDefault();
};
document.addEventListener('mousemove', (e) => {
  if (!resizing) return;
  const delta = e.clientY - resizeStartY;
  hypnoHeight = Math.max(HYPNO_MIN, Math.min(HYPNO_MAX, resizeStartH + delta));
  resizeCanvases();
  drawHypnogram();
  drawWaveforms();
});
document.addEventListener('mouseup', () => {
  if (resizing) {
    resizing = false;
    saveSettings({ hypnoHeight });
  }
  if (labelResizing) {
    labelResizing = false;
    saveSettings({ labelWidth });
  }
});

// --- Label panel resize ---
const labelResizeEl = document.getElementById('labelResize');
let labelResizing = false, labelResizeStartX = 0, labelResizeStartW = 0;
labelResizeEl.onmousedown = (e) => {
  labelResizing = true;
  labelResizeStartX = e.clientX;
  labelResizeStartW = labelWidth;
  e.preventDefault();
};
document.addEventListener('mousemove', (e) => {
  if (!labelResizing) return;
  const delta = e.clientX - labelResizeStartX;
  labelWidth = Math.max(LABEL_MIN, Math.min(LABEL_MAX, labelResizeStartW + delta));
  document.documentElement.style.setProperty('--label-width', labelWidth + 'px');
  const wrap = document.getElementById('waveformWrap');
  const wrapW = wrap.clientWidth;
  waveformCanvas.width = wrapW;
  slowCanvas.width = wrapW;
  drawWaveforms();
});

// --- Hypnogram right-click context menu ---
const ctxMenu = document.getElementById('ctxMenu');
hypnogramCanvas.oncontextmenu = (e) => {
  e.preventDefault();
  const rect = hypnogramCanvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const tStart = visibleStart() * 30;
  const tEnd = (visibleEnd() + 1) * 30;
  const t = tStart + (x / rect.width) * (tEnd - tStart);
  ctxMenuEpoch = Math.floor(t / 30);
  ctxMenu.style.left = e.clientX + 'px';
  ctxMenu.style.top = e.clientY + 'px';
  ctxMenu.classList.add('active');
};

function applyMarks() {
  currentEpoch = Math.max(visibleStart(), Math.min(visibleEnd(), currentEpoch));
  ctxMenu.classList.remove('active');
  drawHypnogram();
  drawWaveforms();
  updateEpochInfo();
  saveCurrentFileState();
}
document.getElementById('ctxMarkIn').onclick = () => {
  pushUndo();
  markIn = ctxMenuEpoch;
  if (markOut !== null && markOut < markIn) markOut = null;
  applyMarks();
};
document.getElementById('ctxMarkOut').onclick = () => {
  pushUndo();
  markOut = ctxMenuEpoch;
  if (markIn !== null && markIn > markOut) markIn = null;
  applyMarks();
};
document.getElementById('ctxClearMarks').onclick = () => {
  pushUndo();
  markIn = null;
  markOut = null;
  applyMarks();
};

// Close context menu on click outside
document.addEventListener('click', (e) => {
  if (!ctxMenu.contains(e.target)) ctxMenu.classList.remove('active');
});

// --- Navigation buttons ---
prevBtn.onclick = () => navigate(-1);
nextBtn.onclick = () => navigate(1);

// --- Keyboard handler ---
document.onkeydown = (e) => {
  // Command palette
  if (e.key === ':' && !cmdBarOpen && document.activeElement === document.body) {
    e.preventDefault(); openCmdBar(); return;
  }
  if (cmdBarOpen) return; // let cmd input handle its own keys

  // Esc closes panels
  if (e.key === 'Escape') {
    const sp = document.getElementById('sidePanel');
    const cp = document.getElementById('channelPanel');
    if (sp.classList.contains('active')) { sp.classList.remove('active'); return; }
    if (cp.classList.contains('active')) { cp.classList.remove('active'); return; }
  }

  // T toggles h5 tree panel
  if (e.key === 't' || e.key === 'T') {
    const sp = document.getElementById('sidePanel');
    if (sp.classList.contains('active')) { sp.classList.remove('active'); }
    else { cmdShowH5(); }
    return;
  }

  // C toggles channel panel
  if (e.key === 'c' || e.key === 'C') {
    document.getElementById('channelPanel').classList.toggle('active');
    return;
  }

  // Font size: +/-
  if (e.key === '=' || e.key === '+') { adjustFontDelta(1); return; }
  if (e.key === '-') { adjustFontDelta(-1); return; }

  // Undo/Redo
  if (e.key === 'z' && e.ctrlKey && !e.shiftKey) { e.preventDefault(); undo(); return; }
  if (e.key === 'Z' && e.ctrlKey && e.shiftKey) { e.preventDefault(); redo(); return; }

  if (e.key === 'ArrowLeft') navigate(-1);
  if (e.key === 'ArrowRight') navigate(1);
  if (e.key === 'ArrowUp') navigate(-10);
  if (e.key === 'ArrowDown') navigate(10);
  if (e.key === '?') {
    const panel = document.getElementById('helpPanel');
    panel.classList.toggle('active');
    if (panel.classList.contains('active')) {
      let used = 0;
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        used += k.length + localStorage.getItem(k).length;
      }
      const total = 5 * 1024 * 1024; // 5 MB typical limit
      document.getElementById('helpStorage').textContent =
        `localStorage: ${(used / 1024).toFixed(1)} KB / ${(total / 1024 / 1024).toFixed(0)} MB`;
    }
  }
};

// --- Hypnogram click/drag ---
let hypnoDragging = false;
function hypnoJump(e) {
  const rect = hypnogramCanvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const tStart = visibleStart() * 30;
  const tEnd = (visibleEnd() + 1) * 30;
  const t = tStart + (x / rect.width) * (tEnd - tStart);
  navigateToTime(t);
}
hypnogramCanvas.onmousedown = (e) => {
  hypnoDragging = true;
  hypnoJump(e);
};
document.addEventListener('mousemove', (e) => {
  if (hypnoDragging) hypnoJump(e);
});
document.addEventListener('mouseup', () => {
  hypnoDragging = false;
});

// --- Wheel scroll on signal panel ---
document.getElementById('waveformWrap').addEventListener('wheel', (e) => {
  e.preventDefault();
  const delta = e.deltaY > 0 ? 1 : -1;
  navigate(delta);
}, { passive: false });

// --- Gain slider ---
gainSlider.oninput = () => {
  gain = parseInt(gainSlider.value);
  gainValue.textContent = gain;
  saveSettings({ gain });
  drawWaveforms();
};

// --- Window resize ---
window.onresize = () => {
  if (psgFile) {
    resizeCanvases();
    drawHypnogram();
    drawWaveforms();
  }
};

// --- Side panel resize/close ---
const SIDE_MIN = 240, SIDE_MAX = 600;
let sidePanelWidth = savedSettings.sidePanelWidth ?? 340;
document.documentElement.style.setProperty('--side-panel-width', sidePanelWidth + 'px');

let sideResizing = false, sideResizeStartX = 0, sideResizeStartW = 0;
document.getElementById('sidePanelResize').onmousedown = (e) => {
  sideResizing = true;
  sideResizeStartX = e.clientX;
  sideResizeStartW = sidePanelWidth;
  e.preventDefault();
};
document.addEventListener('mousemove', (e) => {
  if (!sideResizing) return;
  const delta = sideResizeStartX - e.clientX; // dragging left = wider
  sidePanelWidth = Math.max(SIDE_MIN, Math.min(SIDE_MAX, sideResizeStartW + delta));
  document.documentElement.style.setProperty('--side-panel-width', sidePanelWidth + 'px');
});
document.addEventListener('mouseup', () => {
  if (sideResizing) {
    sideResizing = false;
    saveSettings({ sidePanelWidth });
  }
});

document.getElementById('sidePanelClose').onclick = () => {
  document.getElementById('sidePanel').classList.remove('active');
};

// --- Channel panel resize ---
const CH_PANEL_MIN = 160, CH_PANEL_MAX = 400;
let chPanelWidth = savedSettings.chPanelWidth ?? 200;
document.documentElement.style.setProperty('--ch-panel-width', chPanelWidth + 'px');

let chPanelResizing = false, chResizeStartX = 0, chResizeStartW = 0;
document.getElementById('chPanelResize').onmousedown = (e) => {
  chPanelResizing = true;
  chResizeStartX = e.clientX;
  chResizeStartW = chPanelWidth;
  e.preventDefault();
};
document.addEventListener('mousemove', (e) => {
  if (!chPanelResizing) return;
  const delta = e.clientX - chResizeStartX;
  chPanelWidth = Math.max(CH_PANEL_MIN, Math.min(CH_PANEL_MAX, chResizeStartW + delta));
  document.documentElement.style.setProperty('--ch-panel-width', chPanelWidth + 'px');
});
document.addEventListener('mouseup', () => {
  if (chPanelResizing) {
    chPanelResizing = false;
    saveSettings({ chPanelWidth });
  }
});

// --- Command bar input listeners ---
document.getElementById('cmdInput').addEventListener('input', (e) => {
  cmdActiveIdx = 0;
  updateCmdDropdown(e.target.value);
});

document.getElementById('cmdInput').addEventListener('keydown', (e) => {
  const q = e.target.value.toLowerCase().trim();
  const filtered = q ? COMMANDS.filter(c => c.name.includes(q)) : COMMANDS;
  if (e.key === 'ArrowDown') { e.preventDefault(); cmdActiveIdx = Math.min(cmdActiveIdx + 1, filtered.length - 1); updateCmdDropdown(q); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); cmdActiveIdx = Math.max(cmdActiveIdx - 1, 0); updateCmdDropdown(q); }
  else if (e.key === 'Tab') {
    e.preventDefault();
    if (filtered.length > 0) {
      e.target.value = filtered[cmdActiveIdx].name;
      updateCmdDropdown(filtered[cmdActiveIdx].name);
    }
  }
  else if (e.key === 'Enter') { e.preventDefault(); executeCmdByIndex(filtered); }
  else if (e.key === 'Escape') { e.preventDefault(); closeCmdBar(); }
});

// Click overlay to close
document.getElementById('cmdOverlay').addEventListener('click', (e) => {
  if (e.target === document.getElementById('cmdOverlay')) closeCmdBar();
});
