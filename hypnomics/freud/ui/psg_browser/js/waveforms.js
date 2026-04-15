// =============================================================================
// waveforms.js — Waveform grid, traces, and main draw routine
// =============================================================================

// Slow render cache
let _slowCacheKey = '';
let _slowOffscreen = null;
let _slowLabelCache = null;

// Shared: draw grid on a canvas
function drawGrid(ctx, w, h, nCh, chHeight, tDuration, gridBase) {
  ctx.strokeStyle = `rgba(${gridBase},0.06)`;
  ctx.lineWidth = 1;
  for (let i = 1; i < nCh; i++) {
    const y = i * chHeight;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }
  // Vertical: per second, adaptive interval
  const secInterval = tDuration <= 60 ? 1 : tDuration <= 300 ? 5 : tDuration <= 600 ? 10 : 30;
  const majorInterval = tDuration <= 60 ? 5 : tDuration <= 300 ? 30 : tDuration <= 600 ? 60 : 300;
  for (let s = 0; s <= tDuration; s += secInterval) {
    const x = (s / tDuration) * w;
    ctx.strokeStyle = s % majorInterval === 0
      ? `rgba(${gridBase},0.12)` : `rgba(${gridBase},0.04)`;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }
}

// Shared: draw traces on a canvas
function drawTraces(ctx, w, chDataList, chHeight) {
  const labelData = [];
  chDataList.forEach((d, chIdx) => {
    const group = getYmaxGroup(d.chName);
    const isPinned = pinnedChannels[d.chName] != null;
    const isIsolated = isolatedChannels[d.chName] === true;
    const isFixed = isPinned;

    let ymax;
    if (isPinned) {
      ymax = pinnedChannels[d.chName];
    } else if (autoScaleGlobal && globalYmax[d.chName]) {
      // Global 99th percentile (precomputed for entire recording)
      ymax = globalYmax[d.chName];
    } else {
      // Per-epoch std-based (default when auto-scale is OFF)
      ymax = d.std * 3;
      if (ymax <= 0) ymax = 1;
    }
    const yCenter = chIdx * chHeight + chHeight / 2;
    const pxPerUnit = (chHeight / 2) / ymax;

    ctx.strokeStyle = d.color;
    ctx.lineWidth = 1.2;
    ctx.beginPath();

    const nSamples = d.data.length;
    if (nSamples <= w * 2) {
      // Few samples — draw all points
      for (let i = 0; i < nSamples; i++) {
        const x = (i / d.numSamples) * w;
        const y = yCenter - (d.data[i] - d.mean) * pxPerUnit;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
    } else {
      // Downsample: min/max per pixel column preserves peaks
      const samplesPerPx = nSamples / w;
      for (let px = 0; px < w; px++) {
        const start = Math.floor(px * samplesPerPx);
        const end = Math.min(Math.floor((px + 1) * samplesPerPx), nSamples);
        let mn = d.data[start], mx = d.data[start];
        for (let j = start + 1; j < end; j++) {
          const v = d.data[j];
          if (v < mn) mn = v;
          if (v > mx) mx = v;
        }
        const yMin = yCenter - (mx - d.mean) * pxPerUnit;
        const yMax = yCenter - (mn - d.mean) * pxPerUnit;
        if (px === 0) ctx.moveTo(px, yMin); else ctx.lineTo(px, yMin);
        ctx.lineTo(px, yMax);
      }
    }
    ctx.stroke();

    // Prefer unit from d.ch (works for both real channels and traj pseudo-
    // channels); fall back to the global `channels` array for legacy paths.
    const ch = channels.find(c => c.name === d.chName);
    const unit = d.ch?.unit || ch?.unit || '';
    labelData.push({ name: d.chName, sfreq: d.sfreq, color: d.color, yHalfRange: ymax, group, isFixed, isPinned, isIsolated, unit });
  });
  return labelData;
}

function drawWaveforms() {
  if (!psgFile || activeChannels.length === 0) {
    // Clear both canvases
    const bgColor = getComputedStyle(document.documentElement).getPropertyValue('--bg-waveform').trim();
    [waveformCanvas, slowCanvas].forEach(c => {
      const ctx = c.getContext('2d'); ctx.fillStyle = bgColor; ctx.fillRect(0, 0, c.width, c.height);
    });
    waveformDivider.classList.remove('active');
    slowCanvas.classList.remove('active');
    return;
  }

  // Ensure any active traj signals have a precomputed p99 in globalYmax
  // while auto-scale is on. Cheap when nothing's missing.
  if (autoScaleGlobal) {
    try { ensureTrajGlobalYmax(); } catch(e) { console.warn(e); }
  }

  const sortedActive = [...activeChannels].sort((a, b) => getSignalOrder(a) - getSignalOrder(b));
  const fastChs = sortedActive.filter(n => getSignalSpeed(n) === 'fast');
  const slowChs = sortedActive.filter(n => getSignalSpeed(n) === 'slow');
  const hasSlow = slowChs.length > 0;
  const hasFast = fastChs.length > 0;

  waveformDivider.classList.toggle('active', hasSlow && hasFast);
  slowCanvas.style.display = hasSlow ? '' : 'none';

  // Equal channel height: total wrap height divided by total channels
  const wrap = document.getElementById('waveformWrap');
  const wrapW = wrap.clientWidth;
  const dividerH = (hasSlow && hasFast) ? 2 : 0;
  const totalChs = fastChs.length + slowChs.length;
  const wrapH = wrap.clientHeight - dividerH;
  const chHeight = totalChs > 0 ? wrapH / totalChs : 0;

  const fastH = Math.round(chHeight * fastChs.length);
  const slowH = Math.round(chHeight * slowChs.length);

  waveformCanvas.width = wrapW;
  waveformCanvas.height = fastH || wrapH;
  waveformCanvas.style.height = (fastH || wrapH) + 'px';
  if (hasSlow) {
    slowCanvas.width = wrapW;
    slowCanvas.height = slowH;
    slowCanvas.style.height = slowH + 'px';
  }

  const bgColor = getComputedStyle(document.documentElement).getPropertyValue('--bg-waveform').trim();
  const gridBase = darkMode ? '255,255,255' : '0,0,0';
  const epochStart = viewStartSec;

  let allLabelData = [];

  // --- Fast channels ---
  if (hasFast) {
    const ctx = waveformCanvas.getContext('2d');
    const w = waveformCanvas.width, h = waveformCanvas.height;
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, w, h);

    const chHeight = h / fastChs.length;
    drawGrid(ctx, w, h, fastChs.length, chHeight, fastWindowSec, gridBase);

    const fastData = fastChs.map(n => readChannelData(n, epochStart, fastWindowSec)).filter(Boolean);
    const labelData = drawTraces(ctx, w, fastData, chHeight);
    allLabelData.push(...labelData);

    // Time axis
    const timeColor = darkMode ? 'rgba(255,255,255,0.3)' : 'rgba(0,0,0,0.25)';
    ctx.fillStyle = timeColor;
    ctx.font = '9px "JetBrains Mono", monospace';
    const fastTickInt = fastWindowSec <= 5 ? 1 : fastWindowSec <= 15 ? 2 : 5;
    for (let s = 0; s <= fastWindowSec; s += fastTickInt) {
      ctx.fillText(formatTime(epochStart + s), (s / fastWindowSec) * w + 2, h - 4);
    }
  } else {
    const ctx = waveformCanvas.getContext('2d');
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);
  }

  // --- Slow channels ---
  // Slow trace cache: only re-render traces when the slow window shifts
  if (hasSlow) {
    const ctx = slowCanvas.getContext('2d');
    const w = slowCanvas.width, h = slowCanvas.height;

    // Compute slow window
    const epochMid = currentEpoch * 30 + 15;
    const slowHalf = slowWindowSec / 2;
    let slowStart = epochMid - slowHalf;
    let slowEnd = slowStart + slowWindowSec;
    if (slowStart < 0) { slowStart = 0; slowEnd = slowWindowSec; }
    if (slowEnd > duration) { slowEnd = duration; slowStart = Math.max(0, slowEnd - slowWindowSec); }

    // Cache key for slow traces
    const slowKey = `${slowStart}:${slowWindowSec}:${w}:${h}:${slowChs.join(',')}:${autoScaleGlobal}:${darkMode}`;
    let slowLabelData;

    if (slowKey !== _slowCacheKey) {
      // Expensive: re-render traces to offscreen canvas
      _slowCacheKey = slowKey;
      if (!_slowOffscreen || _slowOffscreen.width !== w || _slowOffscreen.height !== h) {
        _slowOffscreen = document.createElement('canvas');
        _slowOffscreen.width = w;
        _slowOffscreen.height = h;
      }
      const oCtx = _slowOffscreen.getContext('2d');
      oCtx.fillStyle = bgColor;
      oCtx.fillRect(0, 0, w, h);

      const chHeight = h / slowChs.length;
      drawGrid(oCtx, w, h, slowChs.length, chHeight, slowWindowSec, gridBase);

      const slowData = slowChs.map(n => readChannelData(n, slowStart, slowWindowSec)).filter(Boolean);
      _slowLabelCache = drawTraces(oCtx, w, slowData, chHeight);

      // Time axis
      const timeColor = darkMode ? 'rgba(255,255,255,0.3)' : 'rgba(0,0,0,0.25)';
      oCtx.fillStyle = timeColor;
      oCtx.font = '9px "JetBrains Mono", monospace';
      const tickInt = slowWindowSec <= 120 ? 10 : slowWindowSec <= 300 ? 30 : 60;
      const firstTick = Math.ceil(slowStart / tickInt) * tickInt;
      for (let t = firstTick; t <= slowEnd; t += tickInt) {
        const x = ((t - slowStart) / slowWindowSec) * w;
        oCtx.fillText(formatTime(t), x + 2, h - 4);
      }
    }
    slowLabelData = _slowLabelCache || [];
    allLabelData.push(...slowLabelData);

    // Cheap: blit cached traces + draw cursor overlay
    if (_slowOffscreen && _slowOffscreen.width > 0 && _slowOffscreen.height > 0) {
      ctx.drawImage(_slowOffscreen, 0, 0);
    }

    // Cursor: 30s epoch bracket
    const epoch30Start = currentEpoch * 30;
    const outerX0 = ((epoch30Start - slowStart) / slowWindowSec) * w;
    const outerX1 = ((epoch30Start + 30 - slowStart) / slowWindowSec) * w;

    ctx.fillStyle = darkMode ? 'rgba(251,146,60,0.08)' : 'rgba(251,146,60,0.1)';
    ctx.fillRect(outerX0, 0, outerX1 - outerX0, h);
    ctx.strokeStyle = 'rgba(251,146,60,0.5)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(outerX0, 0); ctx.lineTo(outerX0, h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(outerX1, 0); ctx.lineTo(outerX1, h); ctx.stroke();
    ctx.setLineDash([]);

    // Nested fast window (only when zoomed in)
    if (fastWindowSec < 30) {
      const innerX0 = ((epochStart - slowStart) / slowWindowSec) * w;
      const innerX1 = ((epochStart + fastWindowSec - slowStart) / slowWindowSec) * w;
      ctx.fillStyle = darkMode ? 'rgba(251,146,60,0.12)' : 'rgba(251,146,60,0.15)';
      ctx.fillRect(innerX0, 0, innerX1 - innerX0, h);
      ctx.strokeStyle = 'rgba(251,146,60,0.7)';
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(innerX0, 0); ctx.lineTo(innerX0, h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(innerX1, 0); ctx.lineTo(innerX1, h); ctx.stroke();
    }

    // Epoch splitters + stage labels across the slow view
    if (annotations) {
      const intervals = annotations.intervals;
      const labels = annotations.labels;
      ctx.setLineDash([3, 4]);
      ctx.strokeStyle = darkMode ? 'rgba(251,146,60,0.2)' : 'rgba(251,146,60,0.25)';
      ctx.lineWidth = 0.5;
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';

      // Draw epoch boundaries and stage labels
      const firstEpoch = Math.floor(slowStart / 30);
      const lastEpoch = Math.ceil(slowEnd / 30);
      for (let ep = firstEpoch; ep <= lastEpoch; ep++) {
        const epT = ep * 30;
        const x = ((epT - slowStart) / slowWindowSec) * w;

        // Epoch boundary line
        if (epT > slowStart && epT < slowEnd) {
          ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
        }

        // Stage label at top of epoch region
        if (epT >= slowStart && epT + 30 <= slowEnd) {
          const midX = ((epT + 15 - slowStart) / slowWindowSec) * w;
          // Find stage for this epoch
          for (let i = 0; i < labels.length; i++) {
            const t0 = intervals[i * 2];
            const t1 = intervals[i * 2 + 1];
            if (epT >= t0 && epT < t1) {
              const sName = stageName(labels[i]);
              const sColor = stageColor(labels[i]);
              if (sColor) {
                ctx.fillStyle = sColor;
                ctx.globalAlpha = 0.8;
                ctx.font = '600 18px "DM Sans", sans-serif';
                ctx.fillText(sName, midX, 16);
                ctx.globalAlpha = 1;
                ctx.font = '9px "JetBrains Mono", monospace';
              }
              break;
            }
          }
        }
      }
      ctx.setLineDash([]);
    }
  }

  // Labels: uniform height, but first slow channel absorbs divider gap
  const labelHeights = [];
  let slowIdx = 0;
  allLabelData.forEach(d => {
    if (getSignalSpeed(d.name) !== 'fast' && slowIdx === 0 && hasFast) {
      labelHeights.push(chHeight + dividerH);
      slowIdx++;
    } else {
      labelHeights.push(chHeight);
    }
  });
  buildWaveformLabels(allLabelData, labelHeights);
}
