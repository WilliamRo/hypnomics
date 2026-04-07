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

function snapViewStart(t) {
  // Snap to nearest multiple of fastWindowSec
  const snapped = Math.floor(t / fastWindowSec) * fastWindowSec;
  const minT = visibleStart() * 30;
  const maxT = (visibleEnd() + 1) * 30 - fastWindowSec;
  viewStartSec = Math.max(minT, Math.min(maxT, snapped));
  currentEpoch = Math.floor(viewStartSec / 30);
}

function navigate(delta) {
  snapViewStart(viewStartSec + delta * fastWindowSec);
  scheduleEpochRender();
  saveCurrentFileState();
}

function navigateToTime(seconds) {
  snapViewStart(seconds);
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
        const sName = stageName(labels[i]);
        const sColor = stageColor(labels[i]) || '';
        stageStr = ` | <span style="color:${sColor};font-weight:600">${sName}</span>`;
        break;
      }
    }
  }
  epochInfo.innerHTML = `Epoch <span>${currentEpoch + 1}</span> / ${totalEpochs} | ${formatTime(viewStartSec)} [${fastWindowSec}s]${stageStr}`;
  hypnoTime.textContent = formatTime(viewStartSec) + ' - ' + formatTime(viewStartSec + fastWindowSec);
}

// --- Apply theme on load ---
applyTheme(darkMode);

// --- Annotation selector ---
document.getElementById('annoSelect').onchange = (e) => {
  switchAnnotation(e.target.value);
  syncCustomAnnoSelect();
};

// Custom anno dropdown toggle
document.getElementById('annoSelectTrigger').onclick = (e) => {
  e.stopPropagation();
  document.getElementById('annoSelectWrap').classList.toggle('open');
};
document.addEventListener('click', (e) => {
  const wrap = document.getElementById('annoSelectWrap');
  if (wrap && !wrap.contains(e.target)) wrap.classList.remove('open');
});

// --- Stage mode toggle ---
document.getElementById('stageModeBtn').onclick = () => toggleStageMode();

// --- Config button toggle ---
document.getElementById('configBtn').onclick = () => {
  document.getElementById('configPanel').classList.toggle('active');
};

// --- Theme toggle ---
document.getElementById('themeToggle').onclick = () => {
  applyTheme(!darkMode);
  prevLabelKey = ''; // force label rebuild with new colors
  buildChannelToggles();
  drawHypnogram();
  drawWaveforms();
};

// --- Fast window selector ---
const fastSelect = document.getElementById('fastWindowSelect');
fastSelect.value = fastWindowSec;

function applyFastWindow(val) {
  fastWindowSec = val;
  fastSelect.value = val;
  document.getElementById('fastSelectLabel').textContent =
    fastSelect.options[fastSelect.selectedIndex]?.textContent || val + 's';
  snapViewStart(viewStartSec);
  saveSettings({ fastWindowSec });
  clearEpochCache();
  buildChannelToggles();
  drawHypnogram();
  drawWaveforms();
  updateEpochInfo();
}

document.getElementById('fastSelectTrigger').onclick = (e) => {
  e.stopPropagation();
  document.getElementById('fastSelectWrap').classList.toggle('open');
};
document.querySelectorAll('#fastSelectDropdown .custom-select-option').forEach(opt => {
  opt.onclick = () => {
    applyFastWindow(parseInt(opt.dataset.val));
    document.getElementById('fastSelectWrap').classList.remove('open');
  };
});
document.getElementById('fastSelectLabel').textContent =
  fastSelect.options[fastSelect.selectedIndex]?.textContent || fastWindowSec + 's';

// --- Slow window selector ---
const slowSelect = document.getElementById('slowWindowSelect');
slowSelect.value = slowWindowSec;

function applySlowWindow(val) {
  slowWindowSec = val;
  slowSelect.value = val;
  document.getElementById('slowSelectLabel').textContent =
    slowSelect.options[slowSelect.selectedIndex]?.textContent || val + 's';
  saveSettings({ slowWindowSec });
  clearEpochCache();
  buildChannelToggles();
  drawWaveforms();
}

document.getElementById('slowSelectTrigger').onclick = (e) => {
  e.stopPropagation();
  document.getElementById('slowSelectWrap').classList.toggle('open');
};
document.querySelectorAll('#slowSelectDropdown .custom-select-option').forEach(opt => {
  opt.onclick = () => {
    applySlowWindow(parseInt(opt.dataset.val));
    document.getElementById('slowSelectWrap').classList.remove('open');
  };
});
document.getElementById('slowSelectLabel').textContent =
  slowSelect.options[slowSelect.selectedIndex]?.textContent || slowWindowSec + 's';

// --- Click outside to close panels/custom selects ---
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
  // Close all custom selects
  document.querySelectorAll('.custom-select.open').forEach(el => {
    if (!el.contains(e.target)) el.classList.remove('open');
  });
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

  // Annotation shortcuts: w/1/2/3/r/0 (only in stage mode)
  if (stageMode && ANNO_KEY_MAP[e.key] && !e.ctrlKey && !e.altKey) {
    annotateCurrentEpoch(ANNO_KEY_MAP[e.key]);
    return;
  }

  // Font size: +/-
  if (e.key === '=' || e.key === '+') { adjustFontDelta(1); return; }
  if (e.key === '-') { adjustFontDelta(-1); return; }

  // Undo/Redo (note: Ctrl+S saves annotation — handled here)
  if (e.key === 'z' && e.ctrlKey && !e.shiftKey) { e.preventDefault(); undo(); return; }

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
  if (e.shiftKey) {
    // Shift+wheel: cycle fast window duration
    const dir = e.deltaY > 0 ? -1 : 1;
    const idx = FAST_WINDOW_OPTIONS.indexOf(fastWindowSec);
    const newIdx = Math.max(0, Math.min(FAST_WINDOW_OPTIONS.length - 1, idx + dir));
    if (FAST_WINDOW_OPTIONS[newIdx] !== fastWindowSec) {
      applyFastWindow(FAST_WINDOW_OPTIONS[newIdx]);
    }
  } else {
    navigate(e.deltaY > 0 ? 1 : -1);
  }
}, { passive: false });

// --- Auto-scale toggle ---
const autoScaleCheck = document.getElementById('autoScaleCheck');

function updateAutoScaleBtn() {
  autoScaleCheck.checked = autoScaleGlobal;
}
updateAutoScaleBtn();

// Track active computation so toggling off mid-compute cancels it
let _scaleCancel = null;

autoScaleCheck.onchange = async () => {
  // If computing, cancel it
  if (_scaleCancel) { _scaleCancel(); _scaleCancel = null; }

  autoScaleGlobal = autoScaleCheck.checked;
  updateAutoScaleBtn();

  if (autoScaleGlobal && Object.keys(globalYmax).length === 0) {
    // Need to compute — check localStorage first
    const cacheKey = 'morpheus_globalYmax_' + lastFileName;
    const cached = localStorage.getItem(cacheKey);
    if (cached) {
      globalYmax = JSON.parse(cached);
    } else {
      // Compute on main thread using requestAnimationFrame for responsive UI
      globalYmax = {};
      const n = channels.length;
      let ci = 0;
      let cancelled = false;

      loading.innerHTML = `
        <div style="text-align:center">
          <div id="scaleProgress">Computing scale...</div>
          <button class="btn" id="scaleCancelBtn" style="margin-top:12px;padding:4px 14px">Cancel</button>
        </div>`;
      loading.classList.add('active');
      const cancelFn = () => { cancelled = true; };
      document.getElementById('scaleCancelBtn').onclick = cancelFn;
      _scaleCancel = cancelFn;

      function computeNext() {
        if (cancelled) {
          loading.innerHTML = 'Loading PSG file...';
          loading.classList.remove('active');
          globalYmax = {};
          autoScaleGlobal = false;
          updateAutoScaleBtn();
          return;
        }
        if (ci >= n) {
          // All done
          _scaleCancel = null;
          localStorage.setItem(cacheKey, JSON.stringify(globalYmax));
          loading.innerHTML = 'Loading PSG file...';
          loading.classList.remove('active');
          clearEpochCache();
          drawWaveforms();
          return;
        }

        const ch = channels[ci];
        const el = document.getElementById('scaleProgress');
        if (el) el.textContent = `Computing: ${ch.name} (${ci + 1}/${n})`;

        // Schedule actual computation on next frame so progress text renders
        requestAnimationFrame(() => {
          if (cancelled) { computeNext(); return; }

          console.time('scale:' + ch.name);
          const ds = psgFile.get(`signals/${ch.name}`);
          const data = readDataset(ds);

          // Subsample for speed
          const step = Math.max(1, Math.floor(data.length / 100000));
          const nSub = Math.ceil(data.length / step);
          let sum = 0;
          for (let i = 0; i < data.length; i += step) sum += data[i];
          const mean = sum / nSub;

          const absVals = new Float32Array(nSub);
          let j = 0;
          for (let i = 0; i < data.length; i += step) absVals[j++] = Math.abs(data[i] - mean);
          absVals.sort();

          const idx = Math.min(Math.floor(nSub * 0.99), nSub - 1);
          globalYmax[ch.name] = absVals[idx] || 1;
          console.timeEnd('scale:' + ch.name);

          ci++;
          // Use setTimeout(0) to yield back to event loop between channels
          setTimeout(computeNext, 0);
        });
      }

      computeNext();
      return; // drawWaveforms called in computeNext when done
    }
  }

  clearEpochCache();
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


// --- Label context menu ---
const labelCtxMenu = document.getElementById('labelCtxMenu');
let labelCtxTarget = null; // current labelData item

function showLabelCtxMenu(x, y, labelItem) {
  labelCtxTarget = labelItem;

  const isPinned = pinnedChannels[labelItem.name] != null;
  const isIsolated = isolatedChannels[labelItem.name] === true;

  // Update toggle switches
  document.getElementById('labelCtxPinCheck').checked = isPinned;
  document.getElementById('labelCtxIsolateCheck').checked = isIsolated;

  // Hide group actions for isolated channels
  document.getElementById('labelCtxPinGroup').style.display = isIsolated ? 'none' : '';
  document.getElementById('labelCtxUnpinGroup').style.display = isIsolated ? 'none' : '';

  // Position: open upward if near bottom of viewport
  labelCtxMenu.classList.add('active');
  const menuH = labelCtxMenu.offsetHeight;
  const viewH = window.innerHeight;
  const openUp = y + menuH > viewH - 20;

  labelCtxMenu.style.left = x + 'px';
  labelCtxMenu.style.top = openUp ? (y - menuH) + 'px' : y + 'px';
}

// Pin scale toggle (via checkbox inside toggle switch)
document.getElementById('labelCtxPinCheck').onchange = (e) => {
  e.stopPropagation();
  if (!labelCtxTarget) return;
  const name = labelCtxTarget.name;
  if (e.target.checked) {
    pinnedChannels[name] = labelCtxTarget.yHalfRange;
  } else {
    delete pinnedChannels[name];
  }
  saveCurrentFileState();
  drawWaveforms();
};
// Also toggle when clicking the row (not just the switch)
document.getElementById('labelCtxPin').onclick = (e) => {
  if (e.target.closest('.toggle-switch')) return; // let checkbox handle it
  const cb = document.getElementById('labelCtxPinCheck');
  cb.checked = !cb.checked;
  cb.dispatchEvent(new Event('change'));
};

// Isolate toggle
document.getElementById('labelCtxIsolateCheck').onchange = (e) => {
  e.stopPropagation();
  if (!labelCtxTarget) return;
  const name = labelCtxTarget.name;
  if (e.target.checked) {
    isolatedChannels[name] = true;
  } else {
    delete isolatedChannels[name];
  }
  // Update group button visibility
  document.getElementById('labelCtxPinGroup').style.display = e.target.checked ? 'none' : '';
  document.getElementById('labelCtxUnpinGroup').style.display = e.target.checked ? 'none' : '';
  saveCurrentFileState();
  drawWaveforms();
};
document.getElementById('labelCtxIsolate').onclick = (e) => {
  if (e.target.closest('.toggle-switch')) return;
  const cb = document.getElementById('labelCtxIsolateCheck');
  cb.checked = !cb.checked;
  cb.dispatchEvent(new Event('change'));
};

// Pin group: pin all non-isolated channels in this group to the same ymax
document.getElementById('labelCtxPinGroup').onclick = () => {
  if (!labelCtxTarget) return;
  const group = labelCtxTarget.group;
  const ymax = labelCtxTarget.yHalfRange;
  // Pin every non-isolated channel in this group (overwrites existing pins)
  currentLabelData.forEach(d => {
    if (d.group === group && !isolatedChannels[d.name]) {
      pinnedChannels[d.name] = ymax;
    }
  });
  labelCtxMenu.classList.remove('active');
  saveCurrentFileState();
  drawWaveforms();
};

// Unpin group: unpin all non-isolated channels in this group
document.getElementById('labelCtxUnpinGroup').onclick = () => {
  if (!labelCtxTarget) return;
  const group = labelCtxTarget.group;
  currentLabelData.forEach(d => {
    if (d.group === group && !isolatedChannels[d.name]) {
      delete pinnedChannels[d.name];
    }
  });
  labelCtxMenu.classList.remove('active');
  saveCurrentFileState();
  drawWaveforms();
};

// Close label context menu on click outside
document.addEventListener('click', (e) => {
  if (!labelCtxMenu.contains(e.target)) labelCtxMenu.classList.remove('active');
});
