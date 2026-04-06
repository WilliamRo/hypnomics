// =============================================================================
// waveforms.js — Waveform grid, traces, and main draw routine
// =============================================================================

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
function drawTraces(ctx, w, chDataList, chHeight, gainFactor, groupStdMax) {
  const labelData = [];
  chDataList.forEach((d, chIdx) => {
    const group = getYmaxGroup(d.chName);
    const isFixed = fixedYmax[group] != null;
    const ymax = isFixed ? fixedYmax[group] : (groupStdMax[group] * 3 / gainFactor);
    const yCenter = chIdx * chHeight + chHeight / 2;
    const pxPerUnit = (chHeight / 2) / ymax;

    ctx.strokeStyle = d.color;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (let i = 0; i < d.data.length; i++) {
      const x = (i / d.numSamples) * w;
      const y = yCenter - (d.data[i] - d.mean) * pxPerUnit;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    labelData.push({ name: d.chName, sfreq: d.sfreq, color: d.color, yHalfRange: ymax, group, isFixed });
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
  const gainFactor = gain / 50;
  const epochStart = currentEpoch * 30;

  let allLabelData = [];

  // --- Fast channels ---
  if (hasFast) {
    const ctx = waveformCanvas.getContext('2d');
    const w = waveformCanvas.width, h = waveformCanvas.height;
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, w, h);

    const chHeight = h / fastChs.length;
    drawGrid(ctx, w, h, fastChs.length, chHeight, 30, gridBase);

    const fastData = fastChs.map(n => readChannelData(n, epochStart, 30)).filter(Boolean);
    const groupStdMax = {};
    fastData.forEach(d => {
      const g = getYmaxGroup(d.chName);
      groupStdMax[g] = Math.max(groupStdMax[g] || 0, d.std);
    });

    const labelData = drawTraces(ctx, w, fastData, chHeight, gainFactor, groupStdMax);
    allLabelData.push(...labelData);

    // Time axis
    const timeColor = darkMode ? 'rgba(255,255,255,0.3)' : 'rgba(0,0,0,0.25)';
    ctx.fillStyle = timeColor;
    ctx.font = '9px "JetBrains Mono", monospace';
    for (let s = 0; s <= 30; s += 5) {
      ctx.fillText(formatTime(epochStart + s), (s / 30) * w + 2, h - 4);
    }
  } else {
    const ctx = waveformCanvas.getContext('2d');
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);
  }

  // --- Slow channels ---
  if (hasSlow) {
    const ctx = slowCanvas.getContext('2d');
    const w = slowCanvas.width, h = slowCanvas.height;
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, w, h);

    // Slow window centered on current epoch
    const slowHalf = slowWindowSec / 2;
    let slowStart = epochStart + 15 - slowHalf; // center of current epoch
    let slowEnd = slowStart + slowWindowSec;
    if (slowStart < 0) { slowStart = 0; slowEnd = slowWindowSec; }
    if (slowEnd > duration) { slowEnd = duration; slowStart = Math.max(0, slowEnd - slowWindowSec); }

    const chHeight = h / slowChs.length;
    drawGrid(ctx, w, h, slowChs.length, chHeight, slowWindowSec, gridBase);

    const slowData = slowChs.map(n => readChannelData(n, slowStart, slowWindowSec)).filter(Boolean);
    const groupStdMax = {};
    slowData.forEach(d => {
      const g = getYmaxGroup(d.chName);
      groupStdMax[g] = Math.max(groupStdMax[g] || 0, d.std);
    });

    const labelData = drawTraces(ctx, w, slowData, chHeight, gainFactor, groupStdMax);
    allLabelData.push(...labelData);

    // Cursor bracket: current 30s epoch in slow view
    const cursorX0 = ((epochStart - slowStart) / slowWindowSec) * w;
    const cursorX1 = ((epochStart + 30 - slowStart) / slowWindowSec) * w;
    ctx.fillStyle = darkMode ? 'rgba(251,146,60,0.08)' : 'rgba(251,146,60,0.1)';
    ctx.fillRect(cursorX0, 0, cursorX1 - cursorX0, h);
    ctx.strokeStyle = 'rgba(251,146,60,0.7)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(cursorX0, 0); ctx.lineTo(cursorX0, h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cursorX1, 0); ctx.lineTo(cursorX1, h); ctx.stroke();
    ctx.setLineDash([]);

    // Time axis for slow
    const timeColor = darkMode ? 'rgba(255,255,255,0.3)' : 'rgba(0,0,0,0.25)';
    ctx.fillStyle = timeColor;
    ctx.font = '9px "JetBrains Mono", monospace';
    const tickInt = slowWindowSec <= 120 ? 10 : slowWindowSec <= 300 ? 30 : 60;
    const firstTick = Math.ceil(slowStart / tickInt) * tickInt;
    for (let t = firstTick; t <= slowEnd; t += tickInt) {
      const x = ((t - slowStart) / slowWindowSec) * w;
      ctx.fillText(formatTime(t), x + 2, h - 4);
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
