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
  body.appendChild(buildH5Tree(psgFile, '/'));

  // Append local annotations under annotations/
  try {
    const localKeys = await listLocalAnnoKeys(lastFileName);
    if (localKeys.length > 0) {
      const localHeader = document.createElement('div');
      localHeader.className = 'h5-group open';
      localHeader.textContent = 'annotations/ (local)';
      localHeader.style.color = '#fb923c';
      body.appendChild(localHeader);

      const localNode = document.createElement('div');
      localNode.className = 'h5-node';
      for (const key of localKeys) {
        const anno = await loadLocalAnno(lastFileName, key);
        if (!anno) continue;

        const grpEl = document.createElement('div');
        grpEl.className = 'h5-group open';
        grpEl.textContent = key + '/';
        grpEl.style.color = '#fb923c';
        localNode.appendChild(grpEl);

        const details = document.createElement('div');
        details.className = 'h5-node';

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

        // Export button
        const expBtn = document.createElement('button');
        expBtn.className = 'btn';
        expBtn.style.cssText = 'margin:4px 0 4px 16px;font-size:9px;padding:2px 8px';
        expBtn.textContent = '⬇ Export .anno.json';
        expBtn.onclick = () => exportAnnotation(key);
        details.appendChild(expBtn);

        localNode.appendChild(details);
      }
      body.appendChild(localNode);
    }
  } catch(e) { console.warn('Failed to list local annotations:', e); }

  panel.classList.add('active');
  if (fontDelta !== 0) applyFontDelta();
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
      // Group
      const groupEl = document.createElement('div');
      groupEl.className = 'h5-group';
      groupEl.textContent = key + '/';

      const childrenEl = document.createElement('div');
      childrenEl.className = 'h5-node';
      childrenEl.style.display = 'none';

      groupEl.onclick = (e) => {
        e.stopPropagation();
        const open = groupEl.classList.toggle('open');
        childrenEl.style.display = open ? '' : 'none';
        if (open && childrenEl.children.length === 0) {
          childrenEl.appendChild(buildH5Tree(file, childPath));
        }
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
