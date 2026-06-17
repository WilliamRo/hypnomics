// =============================================================================
// panels.js — Channel toggles, command palette, H5 tree panel
// =============================================================================

// (5) Channel toggles
function buildChannelToggles() {
  const body = document.getElementById('channelPanelBody');
  body.innerHTML = '';

  // Group channels by speed then signal type
  const sorted = [...channels].sort((a, b) => getSignalOrder(a.name) - getSignalOrder(b.name));
  const fastChs = sorted.filter(ch => getSignalSpeed(ch.name) === 'fast');
  const slowChs = sorted.filter(ch => getSignalSpeed(ch.name) === 'slow');

  const sections = [];
  if (fastChs.length > 0) sections.push({ label: `FAST (${fastWindowSec}s)`, chs: fastChs });
  if (slowChs.length > 0) sections.push({ label: `SLOW (${slowWindowSec >= 60 ? (slowWindowSec/60)+'min' : slowWindowSec+'s'})`, chs: slowChs });

  sections.forEach((section, si) => {
    if (si > 0) {
      const sep = document.createElement('div');
      sep.style.cssText = 'border-bottom:1px solid var(--border);margin:6px 0';
      body.appendChild(sep);
    }

    // Section header
    const secHeader = document.createElement('div');
    secHeader.className = 'ch-group-label';
    secHeader.style.cssText = 'font-weight:700;color:var(--accent);padding-top:8px';
    secHeader.textContent = section.label;
    body.appendChild(secHeader);

    // Group by signal type within this section
    const groups = {};
    section.chs.forEach(ch => {
      const type = getSignalType(ch.name).type;
      if (!groups[type]) groups[type] = [];
      groups[type].push(ch);
    });

    for (const [type, chs] of Object.entries(groups)) {
      const groupLabel = document.createElement('div');
      groupLabel.className = 'ch-group-label';
      const actions = document.createElement('span');
      actions.className = 'ch-group-actions';
      const allBtn = document.createElement('span');
      allBtn.textContent = 'all';
      allBtn.onclick = () => { chs.forEach(c => { if (!activeChannels.includes(c.name)) activeChannels.push(c.name); }); refreshChannelPanel(); onChannelsChanged(); };
      const noneBtn = document.createElement('span');
      noneBtn.textContent = 'none';
      noneBtn.onclick = () => { chs.forEach(c => { const idx = activeChannels.indexOf(c.name); if (idx >= 0) activeChannels.splice(idx, 1); }); refreshChannelPanel(); onChannelsChanged(); };
      actions.appendChild(allBtn);
      actions.appendChild(noneBtn);
      groupLabel.textContent = type;
      groupLabel.appendChild(actions);
      body.appendChild(groupLabel);

      chs.forEach(ch => {
        const color = getSignalColor(ch.name);
        const item = document.createElement('div');
        item.className = 'ch-item' + (activeChannels.includes(ch.name) ? ' active' : '');
        item.dataset.ch = ch.name;

        const dot = document.createElement('div');
        dot.className = 'ch-dot';
        dot.style.background = activeChannels.includes(ch.name) ? color : 'transparent';
        dot.style.borderColor = activeChannels.includes(ch.name) ? color : '';

        const label = document.createElement('span');
        label.textContent = ch.name;

        const info = document.createElement('span');
        info.style.cssText = 'margin-left:auto;font-size:9px;color:var(--text-dim)';
        info.textContent = ch.sfreq + ' Hz';

        item.appendChild(dot);
        item.appendChild(label);
        item.appendChild(info);

        item.onclick = () => {
          const idx = activeChannels.indexOf(ch.name);
          if (idx >= 0) activeChannels.splice(idx, 1);
          else activeChannels.push(ch.name);
          refreshChannelPanel();
          onChannelsChanged();
        };

        body.appendChild(item);
      });
    }
  });
  if (fontDelta !== 0) applyFontDelta();
}

function refreshChannelPanel() {
  const items = document.querySelectorAll('#channelPanelBody .ch-item');
  items.forEach(item => {
    const name = item.dataset.ch;
    const isActive = activeChannels.includes(name);
    const color = getSignalColor(name);
    item.classList.toggle('active', isActive);
    const dot = item.querySelector('.ch-dot');
    dot.style.background = isActive ? color : 'transparent';
    dot.style.borderColor = isActive ? color : '';
  });
}

function onChannelsChanged() {
  prevLabelKey = ''; // force label rebuild
  saveSettings({ activeChannels });
  saveCurrentFileState();
  drawWaveforms();
  _renderAnalysisPlugin();
}

// (11) Analysis panel
const _analysisPlugins = [];  // registered plugins that use the analysis panel
let _activeAnalysisPlugin = null;
let analysisPanelWidth = savedSettings.analysisPanelWidth ?? 300;
document.documentElement.style.setProperty('--analysis-width', analysisPanelWidth + 'px');

function registerPlugin(plugin) {
  // plugin: { name, description, category?, channels?, execute, render?, onNavigate? }
  // render(canvas, info): draw into analysis panel
  // onNavigate(): called on epoch change if panel is open with this plugin active
  COMMANDS.push({
    name: plugin.name,
    hint: (plugin.category ? `[${plugin.category}] ` : '') + plugin.description,
    fn: () => {
      if (!psgFile) { alert('No file loaded.'); return; }
      if (plugin.render) {
        openAnalysisPanel(plugin);
      } else {
        plugin.execute();
      }
    },
  });
  if (plugin.render) _analysisPlugins.push(plugin);
}

function openAnalysisPanel(plugin) {
  const panel = document.getElementById('analysisPanel');
  panel.classList.add('active');
  _activeAnalysisPlugin = plugin;
  _buildAnalysisTabs();
  _buildAnalysisControls();
  resizeCanvases();
  drawHypnogram(); drawWaveforms();
  // Render after layout settles
  requestAnimationFrame(() => _renderAnalysisPlugin());
}

function toggleAnalysisPlugin(name) {
  if (!psgFile) return;
  const panel = document.getElementById('analysisPanel');
  // If already open with this plugin, close
  if (panel.classList.contains('active') && _activeAnalysisPlugin && _activeAnalysisPlugin.name === name) {
    closeAnalysisPanel();
    return;
  }
  const plugin = _analysisPlugins.find(p => p.name === name);
  if (plugin) openAnalysisPanel(plugin);
}

function closeAnalysisPanel() {
  const panel = document.getElementById('analysisPanel');
  panel.classList.remove('active');
  _activeAnalysisPlugin = null;
  resizeCanvases();
  drawHypnogram(); drawWaveforms();
}

function _buildAnalysisTabs() {
  const tabs = document.getElementById('analysisTabs');
  tabs.innerHTML = '';
  _analysisPlugins.forEach(p => {
    const tab = document.createElement('div');
    tab.className = 'analysis-tab' + (p === _activeAnalysisPlugin ? ' active' : '');
    tab.textContent = p.name;
    tab.onclick = () => {
      _activeAnalysisPlugin = p;
      _buildAnalysisTabs();
      _buildAnalysisControls();
      _renderAnalysisPlugin();
    };
    tabs.appendChild(tab);
  });
}

function _buildAnalysisControls() {
  const container = document.getElementById('analysisControls');
  container.innerHTML = '';
  container.classList.remove('active');
  if (_activeAnalysisPlugin && _activeAnalysisPlugin.buildControls) {
    _activeAnalysisPlugin.buildControls(container);
    container.classList.add('active');
  }
}

function _renderAnalysisPlugin() {
  if (!_activeAnalysisPlugin || !_activeAnalysisPlugin.render) return;
  const canvas = document.getElementById('analysisCanvas');
  const info = document.getElementById('analysisInfo');
  // Size canvas to container
  const body = document.getElementById('analysisBody');
  canvas.width = body.clientWidth;
  canvas.height = body.clientHeight - info.offsetHeight;
  info.innerHTML = '';
  _activeAnalysisPlugin.render(canvas, info);
}

// Called by navigation to update active plugin
function notifyAnalysisNavigate() {
  if (!_activeAnalysisPlugin) return;
  if (_activeAnalysisPlugin.onNavigate) {
    _activeAnalysisPlugin.onNavigate();
  } else if (_activeAnalysisPlugin.render) {
    _renderAnalysisPlugin();
  }
}

// (12) Command palette
const COMMANDS = [];

let cmdBarOpen = false;
let cmdActiveIdx = 0;

function openCmdBar() {
  cmdBarOpen = true;
  const overlay = document.getElementById('cmdOverlay');
  const input = document.getElementById('cmdInput');
  overlay.classList.add('active');
  input.value = '';
  cmdActiveIdx = 0;
  updateCmdDropdown('');
  input.focus();
}

function closeCmdBar() {
  cmdBarOpen = false;
  document.getElementById('cmdOverlay').classList.remove('active');
  document.getElementById('cmdInput').blur();
  document.getElementById('cmdShadow').textContent = '';
}

function highlightMatch(name, query) {
  if (!query) return name;
  const idx = name.toLowerCase().indexOf(query.toLowerCase());
  if (idx < 0) return name;
  return name.slice(0, idx) + `<span class="cmd-match">${name.slice(idx, idx + query.length)}</span>` + name.slice(idx + query.length);
}

function updateCmdDropdown(query) {
  const dd = document.getElementById('cmdDropdown');
  const shadow = document.getElementById('cmdShadow');
  const empty = document.getElementById('cmdEmpty');
  dd.innerHTML = '';
  const q = query.toLowerCase().trim();
  const filtered = q ? COMMANDS.filter(c => c.name.includes(q) || (c.hint && c.hint.toLowerCase().includes(q))) : COMMANDS;
  cmdActiveIdx = Math.min(cmdActiveIdx, Math.max(0, filtered.length - 1));

  // Shadow hint — only show when query is a prefix
  if (filtered.length > 0 && q && filtered[cmdActiveIdx].name.startsWith(q)) {
    shadow.textContent = filtered[cmdActiveIdx].name;
  } else {
    shadow.textContent = '';
  }

  // Empty state
  if (empty) empty.style.display = filtered.length === 0 ? '' : 'none';

  filtered.forEach((cmd, i) => {
    const el = document.createElement('div');
    el.className = 'cmd-item' + (i === cmdActiveIdx ? ' active' : '');
    // Extract category from hint "[Category] description"
    let cat = '', desc = cmd.hint || '';
    const m = desc.match(/^\[(.+?)\]\s*(.*)/);
    if (m) { cat = m[1]; desc = m[2]; }
    el.innerHTML = `<div class="cmd-item-row">`
      + `<span class="cmd-name">${highlightMatch(cmd.name, q)}</span>`
      + (cat ? `<span class="cmd-cat">${cat}</span>` : '')
      + `</div>`
      + (desc ? `<div class="cmd-desc">${desc}</div>` : '');
    el.onclick = () => { closeCmdBar(); cmd.fn(); };
    dd.appendChild(el);
  });
}

function executeCmdByIndex(filtered) {
  if (filtered.length > 0 && cmdActiveIdx < filtered.length) {
    closeCmdBar();
    filtered[cmdActiveIdx].fn();
  }
}

// (12.1) showh5 command
async function cmdShowH5() {
  if (!psgFile) { alert('No file loaded.'); return; }
  const panel = document.getElementById('sidePanel');
  const body = document.getElementById('sidePanelBody');
  document.getElementById('sidePanelTitle').textContent = 'H5 Structure \u2014 ' + (lastFileName || 'unknown');
  body.innerHTML = '';
  _eventNavControls = [];
  body.appendChild(buildH5Tree(psgFile, '/'));

  // Append traj/ group from the companion .traj.h5 (if loaded), with a
  // checkbox on each leaf probe dataset. Checking adds the probe to the
  // active slow-signal display; unchecking removes it.
  if (trajFile && trajSignals.length > 0) {
    // Default OPEN. An inverted sentinel is stored in expandedH5Paths when
    // the user explicitly collapses, so the collapse state persists but
    // first-time users see the probes without an extra click.
    const TRAJ_ROOT_COLLAPSED_KEY = 'traj:root:collapsed';
    const trajRootCollapsed = expandedH5Paths.has(TRAJ_ROOT_COLLAPSED_KEY);

    const trajHeader = document.createElement('div');
    trajHeader.className = trajRootCollapsed ? 'h5-group' : 'h5-group open';
    trajHeader.textContent = 'traj/ (from ' + (lastTrajFileName || '.traj.h5') + ')';
    // Same color used for traj traces and traj labels — resolved via the
    // Traj SIGNAL_TYPES entry so dark/light themes are both covered.
    trajHeader.style.color = getSignalColor('traj::');
    body.appendChild(trajHeader);

    const trajBody = document.createElement('div');
    trajBody.className = 'h5-node';
    trajBody.style.display = trajRootCollapsed ? 'none' : '';
    trajBody.appendChild(buildTrajTreeWithCheckboxes());
    body.appendChild(trajBody);

    trajHeader.onclick = (e) => {
      e.stopPropagation();
      const nowOpen = trajHeader.classList.toggle('open');
      if (nowOpen) expandedH5Paths.delete(TRAJ_ROOT_COLLAPSED_KEY);
      else expandedH5Paths.add(TRAJ_ROOT_COLLAPSED_KEY);
      trajBody.style.display = nowOpen ? '' : 'none';
      try { saveTrajUIState(); } catch(_) {}
    };
  }

  // Append local annotations under annotations/
  try {
    const localKeys = await listLocalAnnoKeys(lastFileName);
    if (localKeys.length > 0) {
      // Default OPEN — inverted sentinel: key present = collapsed
      const LOCAL_ANNO_COLLAPSED_KEY = 'local-anno:root:collapsed';
      const localRootCollapsed = expandedH5Paths.has(LOCAL_ANNO_COLLAPSED_KEY);

      const localHeader = document.createElement('div');
      localHeader.className = localRootCollapsed ? 'h5-group' : 'h5-group open';
      localHeader.textContent = 'annotations/ (local)';
      localHeader.style.color = '#fb923c';
      body.appendChild(localHeader);

      const localNode = document.createElement('div');
      localNode.className = 'h5-node';
      localNode.style.display = localRootCollapsed ? 'none' : '';

      localHeader.onclick = (e) => {
        e.stopPropagation();
        const nowOpen = localHeader.classList.toggle('open');
        if (nowOpen) expandedH5Paths.delete(LOCAL_ANNO_COLLAPSED_KEY);
        else expandedH5Paths.add(LOCAL_ANNO_COLLAPSED_KEY);
        localNode.style.display = nowOpen ? '' : 'none';
      };

      for (const key of localKeys) {
        const anno = await loadLocalAnno(lastFileName, key);
        if (!anno) continue;

        const annoCollapseKey = `local-anno:${key}:collapsed`;
        const annoCollapsed = expandedH5Paths.has(annoCollapseKey);

        const grpEl = document.createElement('div');
        grpEl.className = annoCollapsed ? 'h5-group' : 'h5-group open';
        grpEl.textContent = key + '/';
        grpEl.style.color = '#fb923c';
        localNode.appendChild(grpEl);

        const details = document.createElement('div');
        details.className = 'h5-node';
        details.style.display = annoCollapsed ? 'none' : '';

        grpEl.onclick = (e) => {
          e.stopPropagation();
          const open = grpEl.classList.toggle('open');
          if (open) expandedH5Paths.delete(annoCollapseKey);
          else expandedH5Paths.add(annoCollapseKey);
          details.style.display = open ? '' : 'none';
        };

        const labelsEl = document.createElement('div');
        labelsEl.className = 'h5-dataset';
        labelsEl.innerHTML = `labels<span class="h5-meta">[${anno.labels.length}] int32</span>`;
        details.appendChild(labelsEl);

        const intEl = document.createElement('div');
        intEl.className = 'h5-dataset';
        intEl.innerHTML = `intervals<span class="h5-meta">[${anno.labels.length}\u00d72] float64</span>`;
        details.appendChild(intEl);

        if (anno.labelNames) {
          const attrEl = document.createElement('div');
          attrEl.className = 'h5-attr';
          attrEl.textContent = 'label_names: ' + anno.labelNames.join(', ');
          details.appendChild(attrEl);
        }

        const modEl = document.createElement('div');
        modEl.className = 'h5-attr';
        modEl.style.color = '#fb923c';
        modEl.textContent = 'modified: true (stored in IDB)';
        details.appendChild(modEl);

        // Export + Delete buttons
        const btnRow = document.createElement('div');
        btnRow.style.cssText = 'margin:4px 0 4px 16px;display:flex;gap:4px';

        const expBtn = document.createElement('button');
        expBtn.className = 'btn';
        expBtn.style.cssText = 'font-size:9px;padding:2px 8px';
        expBtn.textContent = '\u2B07 Export .anno.json';
        expBtn.onclick = (e) => { e.stopPropagation(); exportAnnotation(key); };
        btnRow.appendChild(expBtn);

        const delBtn = document.createElement('button');
        delBtn.className = 'btn';
        delBtn.style.cssText = 'font-size:9px;padding:2px 8px;color:#ef4444';
        delBtn.textContent = '\u2716 Delete';
        delBtn.onclick = async (e) => {
          e.stopPropagation();
          const shortName = key.replace('stage ', '');
          if (!confirm('Delete local annotation "' + shortName + '"?\nThis cannot be undone.')) return;
          await deleteLocalAnno(lastFileName, key);
          const idx = annoKeys.indexOf(key);
          if (idx >= 0) annoKeys.splice(idx, 1);
          if (activeAnnoKey === key) {
            if (annoKeys.length > 0) {
              await switchAnnotation(annoKeys[0]);
            } else {
              activeAnnoKey = '';
              annotations = null;
              buildAnnoSelect();
              drawHypnogram();
              drawWaveforms();
              updateEpochInfo();
            }
          } else {
            buildAnnoSelect();
          }
          cmdShowH5();
        };
        btnRow.appendChild(delBtn);

        details.appendChild(btnRow);
        localNode.appendChild(details);
      }
      body.appendChild(localNode);
    }
  } catch(e) { console.warn('Failed to list local annotations:', e); }

  panel.classList.add('active');
  if (fontDelta !== 0) applyFontDelta();
  try { updateEventNav(); } catch(_) {}
}

// Event-row controls registry (prev/next buttons), so their enabled state can
// be refreshed live as the current epoch changes. Reset on each tree rebuild.
let _eventNavControls = [];
// Modal color-picker state (a top-center popup with a click-outside mask).
let eventColorModalOpen = false;
let _eventColorModalEls = null;

// Decorate an event-annotation group header (rendered in-place under
// annotations/). Layout: [checkbox] event xxx/  [◀][▶][swatch]. The nav arrows
// jump to the prev/next epoch containing this event (disabled when none in that
// direction); the swatch opens a modal color picker. Controls show only when
// the event is checked. Toggling/recoloring persists + repaints the fast panel.
function decorateEventGroupHeader(groupEl, key) {
  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.checked = activeEventKeys.has(key);
  cb.style.cssText = 'vertical-align:middle;margin:0 5px 0 0;width:11px;height:11px;cursor:pointer';
  cb.onclick = (e) => e.stopPropagation();

  // Controls group (arrows + swatch), shown only when checked.
  const ctrl = document.createElement('span');
  ctrl.style.cssText = 'display:none;margin-left:8px;align-items:center;gap:4px;vertical-align:middle';

  const prevBtn = navButton('◀', 'Jump to previous epoch with this event');
  const nextBtn = navButton('▶', 'Jump to next epoch with this event');
  prevBtn.onclick = (e) => {
    e.stopPropagation();
    const t = eventNavEpoch(key, -1);
    if (t === null) return;
    navigateToTime(t * 30); saveCurrentFileState();
  };
  nextBtn.onclick = (e) => {
    e.stopPropagation();
    const t = eventNavEpoch(key, +1);
    if (t === null) return;
    navigateToTime(t * 30); saveCurrentFileState();
  };

  const sw = document.createElement('span');
  sw.title = 'Click to choose highlight color';
  sw.style.cssText =
    'width:12px;height:12px;border-radius:2px;cursor:pointer;' +
    'border:1px solid rgba(128,128,128,0.6);vertical-align:middle';
  sw.onclick = (e) => { e.stopPropagation(); openEventColorModal(key, sw); };

  ctrl.appendChild(prevBtn);
  ctrl.appendChild(nextBtn);
  ctrl.appendChild(sw);

  const syncControls = () => {
    if (cb.checked) {
      ctrl.style.display = 'inline-flex';
      sw.style.background = eventColors[key] || '#888888';
    } else {
      ctrl.style.display = 'none';
    }
  };

  cb.onchange = () => {
    if (cb.checked) {
      activeEventKeys.add(key);
      if (!eventColors[key]) eventColors[key] = nextEventColor();
    } else {
      activeEventKeys.delete(key);
    }
    syncControls();
    try { saveEventUIState(); } catch(_) {}
    try { drawWaveforms(); } catch(_) {}
    try { updateEventNav(); } catch(_) {}
  };

  groupEl.insertBefore(cb, groupEl.firstChild);
  groupEl.appendChild(ctrl);
  _eventNavControls.push({ key, prevBtn, nextBtn });
  syncControls();
}

function navButton(glyph, title) {
  const b = document.createElement('button');
  b.textContent = glyph;
  b.title = title;
  b.style.cssText =
    'font-family:inherit;font-size:10px;line-height:1;padding:1px 4px;' +
    'background:var(--bg-surface);color:var(--text);border:1px solid var(--border);' +
    'border-radius:3px;cursor:pointer;vertical-align:middle';
  b.onmousedown = (e) => e.stopPropagation();
  return b;
}

function setNavBtnEnabled(btn, on) {
  btn.disabled = !on;
  btn.style.opacity = on ? '1' : '0.3';
  btn.style.cursor = on ? 'pointer' : 'default';
}

// Refresh prev/next enabled state for every event row against the current
// epoch. Stale (detached) controls from a prior rebuild are pruned.
function updateEventNav() {
  _eventNavControls = _eventNavControls.filter(c => document.contains(c.prevBtn));
  for (const c of _eventNavControls) {
    setNavBtnEnabled(c.prevBtn, eventNavEpoch(c.key, -1) !== null);
    setNavBtnEnabled(c.nextBtn, eventNavEpoch(c.key, +1) !== null);
  }
}

// Modal color picker: a top-center popup over a full-screen mask. Clicking the
// mask (or Escape) closes it; while open it blocks all other interaction
// (mouse via the mask, keys via the document.onkeydown guard).
function openEventColorModal(key, swatchEl) {
  if (eventColorModalOpen) return;
  eventColorModalOpen = true;

  const mask = document.createElement('div');
  mask.style.cssText = 'position:fixed;inset:0;z-index:10000;background:rgba(0,0,0,0.45)';
  mask.onclick = (e) => { e.stopPropagation(); closeEventColorModal(); };
  mask.addEventListener('wheel', (e) => { e.preventDefault(); e.stopPropagation(); }, { passive: false });

  const panel = document.createElement('div');
  panel.style.cssText =
    'position:fixed;top:64px;left:50%;transform:translateX(-50%);z-index:10001;' +
    'background:var(--bg-panel);border:1px solid var(--accent);border-radius:6px;' +
    "padding:14px 16px;box-shadow:0 10px 30px rgba(0,0,0,0.5);min-width:200px;" +
    "font-family:'JetBrains Mono',monospace";
  panel.onclick = (e) => e.stopPropagation();

  const title = document.createElement('div');
  title.textContent = 'Highlight color — ' + key.replace(/^event /, '');
  title.style.cssText = 'color:var(--text);font-size:12px;margin-bottom:10px';
  panel.appendChild(title);

  const apply = (c, andClose) => {
    eventColors[key] = c;
    if (swatchEl) swatchEl.style.background = c;
    try { saveEventUIState(); } catch(_) {}
    try { drawWaveforms(); } catch(_) {}
    if (andClose) closeEventColorModal();
  };

  const grid = document.createElement('div');
  grid.style.cssText = 'display:grid;grid-template-columns:repeat(5,22px);gap:8px;margin-bottom:12px';
  for (const c of EVENT_COLOR_PALETTE) {
    const b = document.createElement('div');
    b.title = c;
    const sel = (eventColors[key] || '').toLowerCase() === c.toLowerCase();
    b.style.cssText =
      'width:22px;height:22px;border-radius:3px;cursor:pointer;background:' + c +
      ';border:2px solid ' + (sel ? 'var(--text)' : 'transparent');
    b.onclick = (e) => { e.stopPropagation(); apply(c, true); };
    grid.appendChild(b);
  }
  panel.appendChild(grid);

  const row = document.createElement('div');
  row.style.cssText = 'display:flex;align-items:center;gap:8px';
  const lbl = document.createElement('span');
  lbl.textContent = 'Custom';
  lbl.style.cssText = 'color:var(--text-dim);font-size:11px';
  const inp = document.createElement('input');
  inp.type = 'color';
  inp.value = toHex6(eventColors[key] || '#888888');
  inp.style.cssText = 'width:40px;height:26px;padding:0;border:1px solid var(--border);background:none;cursor:pointer';
  inp.oninput = () => apply(inp.value, false);
  row.appendChild(lbl);
  row.appendChild(inp);
  panel.appendChild(row);

  document.body.appendChild(mask);
  document.body.appendChild(panel);
  _eventColorModalEls = { mask, panel };
}

function closeEventColorModal() {
  if (!eventColorModalOpen) return;
  eventColorModalOpen = false;
  try { _eventColorModalEls.mask.remove(); } catch(_) {}
  try { _eventColorModalEls.panel.remove(); } catch(_) {}
  _eventColorModalEls = null;
}

// Normalize an arbitrary color string to a 7-char '#rrggbb' for <input type=color>.
function toHex6(c) {
  return (typeof c === 'string' && /^#[0-9a-fA-F]{6}$/.test(c)) ? c : '#888888';
}

function buildH5Tree(file, path) {
  const frag = document.createDocumentFragment();
  let node;
  try { node = path === '/' ? file : file.get(path); } catch(e) { return frag; }
  if (!node) return frag;

  // Attributes
  if (node.attrs) {
    try {
      const attrKeys = Object.keys(node.attrs);
      attrKeys.forEach(key => {
        const attr = document.createElement('div');
        attr.className = 'h5-attr';
        let val;
        try { val = node.attrs[key].value; } catch(e) { val = '(unreadable)'; }
        if (val instanceof Float64Array || val instanceof Float32Array || val instanceof Int32Array) {
          val = val.length <= 6 ? Array.from(val).join(', ') : `[${val.length} values]`;
        }
        if (typeof val === 'string' && val.length > 50) val = val.slice(0, 50) + '...';
        attr.textContent = `${key}: ${val}`;
        frag.appendChild(attr);
      });
    } catch(e) {}
  }

  // Children (groups and datasets)
  let keys;
  try { keys = node.keys(); } catch(e) { return frag; }

  keys.forEach(key => {
    const childPath = path === '/' ? key : path + '/' + key;
    let child;
    try { child = file.get(childPath); } catch(e) { return; }
    if (!child) return;

    const isGroup = child.keys !== undefined;

    if (isGroup) {
      // Group. Honor remembered expansion state so toggling the tree panel
      // doesn't collapse everything the user had opened.
      const expandKey = 'psg:' + childPath;
      const isExpanded = expandedH5Paths.has(expandKey);

      const groupEl = document.createElement('div');
      groupEl.className = isExpanded ? 'h5-group open' : 'h5-group';
      groupEl.textContent = key + '/';

      // Event annotations (in-place under annotations/) get a leading checkbox
      // to toggle their highlight + a trailing color swatch (shown when checked).
      if (path === 'annotations' && typeof eventKeys !== 'undefined'
          && eventKeys.includes(key)) {
        decorateEventGroupHeader(groupEl, key);
      }

      const childrenEl = document.createElement('div');
      childrenEl.className = 'h5-node';
      childrenEl.style.display = isExpanded ? '' : 'none';
      if (isExpanded) {
        // Eagerly rebuild children so deeper expansion state is restored too
        childrenEl.appendChild(buildH5Tree(file, childPath));
      }

      groupEl.onclick = (e) => {
        e.stopPropagation();
        const open = groupEl.classList.toggle('open');
        if (open) expandedH5Paths.add(expandKey);
        else expandedH5Paths.delete(expandKey);
        childrenEl.style.display = open ? '' : 'none';
        if (open && childrenEl.children.length === 0) {
          childrenEl.appendChild(buildH5Tree(file, childPath));
        }
        try { saveTrajUIState(); } catch(_) {}
      };

      frag.appendChild(groupEl);
      frag.appendChild(childrenEl);
    } else {
      // Dataset
      const dsEl = document.createElement('div');
      dsEl.className = 'h5-dataset';
      let meta = '';
      try {
        const shape = child.shape;
        const dtype = child.dtype;
        meta = `[${shape.join('\u00d7')}] ${dtype}`;
      } catch(e) {}
      dsEl.innerHTML = `${key}<span class="h5-meta">${meta}</span>`;

      // Dataset attrs
      const dsAttrs = document.createDocumentFragment();
      if (child.attrs) {
        try {
          Object.keys(child.attrs).forEach(ak => {
            const attr = document.createElement('div');
            attr.className = 'h5-attr';
            let val;
            try { val = child.attrs[ak].value; } catch(e) { val = '(unreadable)'; }
            attr.textContent = `${ak}: ${val}`;
            dsAttrs.appendChild(attr);
          });
        } catch(e) {}
      }

      frag.appendChild(dsEl);
      frag.appendChild(dsAttrs);
    }
  });

  return frag;
}

// Build a 3-level nested tree of the loaded .traj.h5 (channel → tr → probe)
// with a checkbox beside each probe leaf. Checking the box appends the probe
// to `activeChannels` as a slow signal; unchecking removes it and redraws.
// Honors `expandedH5Paths` so the previously-open sub-tree is restored when
// the panel is re-rendered. Inline layout (no flex) so rows match the
// surrounding .h5-dataset rows in font-size and line-height.
function buildTrajTreeWithCheckboxes() {
  const frag = document.createDocumentFragment();

  // Group trajSignals by channel, then by tr
  const byCh = new Map();
  for (const sig of trajSignals) {
    if (!byCh.has(sig.ch)) byCh.set(sig.ch, new Map());
    const byTr = byCh.get(sig.ch);
    const trKey = sig.tr + 's';
    if (!byTr.has(trKey)) byTr.set(trKey, []);
    byTr.get(trKey).push(sig);
  }

  for (const [ch, byTr] of byCh.entries()) {
    const chPath = `traj/${ch}`;
    const chExpanded = expandedH5Paths.has(chPath);

    const chEl = document.createElement('div');
    chEl.className = chExpanded ? 'h5-group open' : 'h5-group';
    chEl.textContent = ch + '/';
    frag.appendChild(chEl);

    const chBody = document.createElement('div');
    chBody.className = 'h5-node';
    chBody.style.display = chExpanded ? '' : 'none';
    chEl.onclick = (e) => {
      e.stopPropagation();
      const open = chEl.classList.toggle('open');
      if (open) expandedH5Paths.add(chPath);
      else expandedH5Paths.delete(chPath);
      chBody.style.display = open ? '' : 'none';
      try { saveTrajUIState(); } catch(_) {}
    };
    frag.appendChild(chBody);

    for (const [trKey, sigs] of byTr.entries()) {
      const trPath = `${chPath}/${trKey}`;
      const trExpanded = expandedH5Paths.has(trPath);

      const trEl = document.createElement('div');
      trEl.className = trExpanded ? 'h5-group open' : 'h5-group';
      trEl.textContent = trKey + '/';
      chBody.appendChild(trEl);

      const trBody = document.createElement('div');
      trBody.className = 'h5-node';
      trBody.style.display = trExpanded ? '' : 'none';
      trEl.onclick = (e) => {
        e.stopPropagation();
        const open = trEl.classList.toggle('open');
        if (open) expandedH5Paths.add(trPath);
        else expandedH5Paths.delete(trPath);
        trBody.style.display = open ? '' : 'none';
        try { saveTrajUIState(); } catch(_) {}
      };
      chBody.appendChild(trBody);

      for (const sig of sigs) {
        // Match existing .h5-dataset inline-flow pattern. An inline-sized
        // checkbox is prepended so the row keeps the same line-height as
        // the surrounding dataset rows. We pin font-size/family explicitly
        // because applyFontDelta walks form controls and caches their UA
        // font-size (~13px), which otherwise compounds across rebuilds and
        // makes the traj row visibly larger than the signals/ rows.
        const row = document.createElement('div');
        row.className = 'h5-dataset';
        row.style.cssText = "font-size:11px;font-family:'JetBrains Mono',monospace;line-height:1.4";

        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = activeChannels.includes(sig.name);
        cb.style.cssText = 'vertical-align:middle;margin:0 4px 0 0;width:11px;height:11px;cursor:pointer;font-size:11px';
        cb.onclick = (e) => e.stopPropagation();
        cb.onchange = () => {
          const idx = activeChannels.indexOf(sig.name);
          if (cb.checked && idx < 0) {
            activeChannels.push(sig.name);
          } else if (!cb.checked && idx >= 0) {
            activeChannels.splice(idx, 1);
          }
          try { saveSettings({ activeChannels }); } catch(_) {}
          try { saveTrajUIState(); } catch(_) {}
          try { refreshChannelPanel(); } catch(_) {}
          try { drawWaveforms(); } catch(_) {}
        };
        row.appendChild(cb);

        const meta = ` [${sig.length}] float32`;
        row.insertAdjacentHTML('beforeend', `${sig.pk}<span class="h5-meta">${meta}</span>`);

        // Per-probe log₁₀ toggle. Reuses the existing .toggle-switch style
        // but scaled down to fit the tree row; a small "log" label precedes
        // it. When toggled we invalidate both the slice cache and the
        // auto-scale ymax, then redraw if the probe is currently active.
        const logLabel = document.createElement('span');
        logLabel.textContent = 'log';
        logLabel.className = 'h5-meta';
        logLabel.style.cssText = 'margin-left:10px;vertical-align:middle';
        row.appendChild(logLabel);

        const logSwitch = document.createElement('label');
        logSwitch.className = 'toggle-switch';
        logSwitch.title = 'Toggle log₁₀ transform for this probe';
        logSwitch.style.cssText = 'vertical-align:middle;margin-left:4px;transform:scale(0.7);transform-origin:left center';
        const logInput = document.createElement('input');
        logInput.type = 'checkbox';
        logInput.checked = trajLogEnabled.has(sig.name);
        logInput.onclick = (e) => e.stopPropagation();
        logInput.onchange = () => {
          if (logInput.checked) trajLogEnabled.add(sig.name);
          else trajLogEnabled.delete(sig.name);
          // Evict cached slices for this sig (cache key includes L/R flag
          // but the old entries are stale under the new unit semantics)
          const prefix = `traj:${sig.name}:`;
          for (const k of Object.keys(epochCache)) {
            if (k.startsWith(prefix)) delete epochCache[k];
          }
          // Force recompute of auto-scale ymax under the new transform
          delete globalYmax[sig.name];
          // Force slow-cache invalidation (the key depends only on names,
          // so flipping log wouldn't otherwise trigger a repaint)
          if (typeof _slowCacheKey !== 'undefined') _slowCacheKey = '';
          try { saveTrajUIState(); } catch(_) {}
          if (activeChannels.includes(sig.name)) {
            try { drawWaveforms(); } catch(_) {}
          }
          // If this probe is currently overlaid on the hypnogram, reload
          // its full-night data under the new log mode and redraw.
          if (hypnoTrajName === sig.name) {
            try { reloadHypnoTrajOverlay(); } catch(_) {}
          }
        };
        const logSlider = document.createElement('span');
        logSlider.className = 'toggle-slider';
        logSwitch.appendChild(logInput);
        logSwitch.appendChild(logSlider);
        row.appendChild(logSwitch);

        trBody.appendChild(row);
      }
    }
  }

  return frag;
}
