// =============================================================================
// labels.js — Waveform label panel construction and tick canvas
// =============================================================================

// chHeights: array of per-channel heights (or single number for backward compat)
function buildWaveformLabels(labelData, chHeights) {
  currentLabelData = labelData;
  const heights = Array.isArray(chHeights) ? chHeights : labelData.map(() => chHeights);
  const container = document.getElementById('waveformLabels');
  // Include traj log state in the key so toggling log on a probe forces a
  // full rebuild (the label text changes from "PROBE" to "log PROBE").
  const labelKey = labelData.map((d, i) => {
    const isTrajLog = typeof d.name === 'string' && d.name.startsWith('traj::') &&
                      typeof trajLogEnabled !== 'undefined' && trajLogEnabled.has(d.name);
    return d.name + ':' + heights[i] + (isTrajLog ? ':L' : '');
  }).join(',');
  const needsFullRebuild = labelKey !== prevLabelKey;

  // Compute cumulative Y positions
  const yTops = [];
  let cumY = 0;
  for (let i = 0; i < labelData.length; i++) {
    yTops.push(cumY);
    cumY += heights[i] || 0;
  }
  const totalH = cumY;

  if (needsFullRebuild) {
    prevLabelKey = labelKey;
    const resizeHandle = document.getElementById('labelResize');
    container.innerHTML = '';
    container.appendChild(resizeHandle);

    const tc = document.createElement('canvas');
    tc.id = 'labelTickCanvas';
    tc.className = 'waveform-label-ticks';
    container.appendChild(tc);

    labelData.forEach((d, i) => {
      const el = document.createElement('div');
      el.className = 'waveform-label-item';
      el.style.top = yTops[i] + 'px';
      el.style.height = heights[i] + 'px';
      el.style.cursor = 'pointer';

      const isTraj = typeof d.name === 'string' && d.name.startsWith('traj::');
      if (isTraj) {
        // 3-row layout: source channel name / probe name / sampling frequency
        const parts = d.name.split('::');
        const source = parts[1] || '';
        const probe  = parts[3] || '';
        const isLog = typeof trajLogEnabled !== 'undefined' && trajLogEnabled.has(d.name);

        const nameSpan = document.createElement('div');
        nameSpan.className = 'waveform-label-name';
        nameSpan.style.color = d.color;
        nameSpan.textContent = source;

        const probeSpan = document.createElement('div');
        probeSpan.className = 'waveform-label-info';
        probeSpan.style.color = d.color;
        probeSpan.textContent = isLog ? ('log ' + probe) : probe;

        const infoSpan = document.createElement('div');
        infoSpan.className = 'waveform-label-info';
        infoSpan.textContent = formatSfreq(d.sfreq);

        el.appendChild(nameSpan);
        el.appendChild(probeSpan);
        el.appendChild(infoSpan);
      } else {
        const nameSpan = document.createElement('div');
        nameSpan.className = 'waveform-label-name';
        nameSpan.style.color = d.color;
        nameSpan.textContent = d.name;

        const infoSpan = document.createElement('div');
        infoSpan.className = 'waveform-label-info';
        infoSpan.textContent = formatSfreq(d.sfreq);

        el.appendChild(nameSpan);
        el.appendChild(infoSpan);
      }

      // Right-click context menu for pin/isolate
      el.oncontextmenu = ((idx) => (e) => {
        e.preventDefault();
        const live = currentLabelData[idx];
        if (!live) return;
        showLabelCtxMenu(e.clientX, e.clientY, live);
      })(i);

      // Middle-click on a traj label → toggle whole-night overlay on
      // the hypnogram background. mousedown is used because `click`
      // doesn't fire reliably for the middle button across browsers.
      el.onmousedown = ((idx) => (e) => {
        if (e.button !== 1) return;
        e.preventDefault();
        const live = currentLabelData[idx];
        if (!live || typeof live.name !== 'string' ||
            !live.name.startsWith('traj::')) return;
        try { setHypnoTrajOverlay(live.name); } catch(err) { console.warn(err); }
      })(i);
      // Some browsers still emit auxclick for middle; swallow it to
      // avoid double-toggling.
      el.onauxclick = (e) => { if (e.button === 1) e.preventDefault(); };

      container.appendChild(el);
    });
    if (fontDelta !== 0) applyFontDelta();
  }

  // Always redraw tick canvas
  const tickCanvas = document.getElementById('labelTickCanvas');
  if (!tickCanvas) return;
  tickCanvas.width = labelWidth;
  tickCanvas.height = totalH;
  tickCanvas.style.width = labelWidth + 'px';
  tickCanvas.style.height = totalH + 'px';

  const tCtx = tickCanvas.getContext('2d');
  const tickLen = 6;
  const tickColor = darkMode ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.15)';

  labelData.forEach((d, i) => {
    const h = heights[i];
    const yCenter = yTops[i] + h / 2;
    const isFixed = d.isFixed;
    const isIsolated = d.isIsolated;
    const textColor = isFixed
      ? (darkMode ? 'rgba(255,255,255,0.8)' : 'rgba(0,0,0,0.7)')
      : isIsolated
        ? (darkMode ? 'rgba(255,200,100,0.5)' : 'rgba(180,120,0,0.4)')
        : (darkMode ? 'rgba(255,255,255,0.35)' : 'rgba(0,0,0,0.25)');

    const margin = 8;
    const maxPx = h / 2 - margin;
    let tickValue = niceRound(d.yHalfRange * 0.6);
    let tickPx = d.yHalfRange > 0 ? (tickValue / d.yHalfRange) * (h / 2) : 0;
    let guard = 0;
    while (tickPx > maxPx && tickValue > 0 && guard++ < 15) {
      tickValue = niceRound(tickValue * 0.4);
      tickPx = (tickValue / d.yHalfRange) * (h / 2);
    }

    // Center tick
    tCtx.strokeStyle = tickColor;
    tCtx.lineWidth = 1;
    tCtx.beginPath();
    tCtx.moveTo(labelWidth - tickLen, yCenter);
    tCtx.lineTo(labelWidth, yCenter);
    tCtx.stroke();

    // +/- ticks
    if (tickPx > 0) {
      const yTop = yCenter - tickPx;
      const yBot = yCenter + tickPx;
      const label = formatYScale(tickValue, d.unit);

      tCtx.strokeStyle = tickColor;
      tCtx.beginPath();
      tCtx.moveTo(labelWidth - tickLen, yTop);
      tCtx.lineTo(labelWidth, yTop);
      tCtx.stroke();
      tCtx.beginPath();
      tCtx.moveTo(labelWidth - tickLen, yBot);
      tCtx.lineTo(labelWidth, yBot);
      tCtx.stroke();

      tCtx.fillStyle = textColor;
      tCtx.font = (isFixed ? 'bold ' : '') + '9px "JetBrains Mono", monospace';
      tCtx.textAlign = 'right';
      tCtx.fillText(label, labelWidth - tickLen - 2, yTop + 4);
      tCtx.fillText('-' + label, labelWidth - tickLen - 2, yBot + 4);
    }
  });
}
