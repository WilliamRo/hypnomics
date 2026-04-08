// =============================================================================
// hypnogram.js — Hypnogram and time axis rendering
// =============================================================================

// (6) Hypnogram rendering — staircase style
// Stage Y positions: W=0, REM=1, N1=2, N2=3, N3=4
// (Hardcoded stage constants removed — now using dynamic stageMap)

function drawHypnogram() {
  updateHypnoYAxis();
  const canvas = hypnogramCanvas;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;

  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--bg-surface').trim();
  ctx.fillRect(0, 0, w, h);

  if (!annotations) return;

  const intervals = annotations.intervals;
  const labels = annotations.labels;
  const pad = 4;
  const plotH = h - pad * 2;

  // Visible time range
  const tStart = visibleStart() * 30;
  const tEnd = (visibleEnd() + 1) * 30;
  const tRange = tEnd - tStart;

  function timeToX(t) { return ((t - tStart) / tRange) * w; }
  function stageYFromRow(row) {
    return pad + (row / (STAGE_ROWS - 1)) * plotH;
  }

  // Draw horizontal grid lines
  const gridColor = darkMode ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';
  ctx.strokeStyle = gridColor;
  ctx.lineWidth = 0.5;
  for (let r = 0; r < STAGE_ROWS; r++) {
    const y = pad + (r / (STAGE_ROWS - 1)) * plotH;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  // Draw staircase (only visible portion, skip unknown stages)
  ctx.lineWidth = 1.5;
  let prevDrawnRow = null;
  for (let i = 0; i < labels.length; i++) {
    const t0 = intervals[i * 2];
    const t1 = intervals[i * 2 + 1];
    if (t1 <= tStart || t0 >= tEnd) continue;

    const row = stageRow(labels[i]);
    if (row === undefined) { prevDrawnRow = null; continue; } // gap for unknown

    const color = stageColor(labels[i]);
    const x0 = timeToX(Math.max(t0, tStart));
    const x1 = timeToX(Math.min(t1, tEnd));
    const y = stageYFromRow(row);

    ctx.strokeStyle = color;
    ctx.lineWidth = (row === 1) ? 7.5 : 1.5; // REM (row 1) is 5x thicker
    ctx.beginPath();

    // Vertical transition from previous known stage
    if (prevDrawnRow !== null && prevDrawnRow !== row && t0 >= tStart) {
      ctx.lineWidth = 1.5; // transitions stay thin
      ctx.moveTo(x0, stageYFromRow(prevDrawnRow));
      ctx.lineTo(x0, y);
      ctx.stroke();
      ctx.beginPath();
      ctx.lineWidth = (row === 1) ? 7.5 : 1.5;
    }

    ctx.moveTo(x0, y);
    ctx.lineTo(x1, y);
    ctx.stroke();
    prevDrawnRow = row;
  }

  // (6.1) Current epoch marker — orange dashed bracket + triangle
  const mx0 = timeToX(currentEpoch * 30);
  const mx1 = timeToX(currentEpoch * 30 + 30);
  const mxMid = (mx0 + mx1) / 2;
  const markerColor = 'rgba(251,146,60,0.7)'; // transparent orange

  // Subtle fill
  ctx.fillStyle = darkMode ? 'rgba(251,146,60,0.08)' : 'rgba(251,146,60,0.1)';
  ctx.fillRect(mx0, 0, mx1 - mx0, h);

  // Dashed bracket lines
  ctx.strokeStyle = markerColor;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(mx0, 0); ctx.lineTo(mx0, h);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(mx1, 0); ctx.lineTo(mx1, h);
  ctx.stroke();
  ctx.setLineDash([]);


  // (6.2) Draw time axis (separate canvas below)
  drawTimeAxis(tStart, tEnd, tRange, timeToX, mx0, mx1);
}

function drawTimeAxis(tStart, tEnd, tRange, timeToX, mx0, mx1) {
  const ta = document.getElementById('timeAxisCanvas');
  const tCtx = ta.getContext('2d');
  const tw = ta.width, th = ta.height;

  tCtx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--bg-panel').trim();
  tCtx.fillRect(0, 0, tw, th);

  const tickTextColor = darkMode ? 'rgba(255,255,255,0.4)' : 'rgba(0,0,0,0.35)';
  const tickLineColor = darkMode ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.12)';

  // Auto interval
  let tickInterval;
  if (tRange <= 3600) tickInterval = 300;
  else if (tRange <= 14400) tickInterval = 1800;
  else tickInterval = 3600;

  tCtx.font = '8px "JetBrains Mono", monospace';
  tCtx.textAlign = 'center';

  const firstTick = Math.ceil(tStart / tickInterval) * tickInterval;
  for (let t = firstTick; t < tEnd; t += tickInterval) {
    const x = timeToX(t);
    // Tick line
    tCtx.strokeStyle = tickLineColor;
    tCtx.lineWidth = 1;
    tCtx.beginPath();
    tCtx.moveTo(x, 0);
    tCtx.lineTo(x, 4);
    tCtx.stroke();
    // Label
    const hh = Math.floor(t / 3600);
    const mm = Math.floor((t % 3600) / 60);
    tCtx.fillStyle = tickTextColor;
    tCtx.fillText(`${hh}:${String(mm).padStart(2, '0')}`, x, 12);
  }

  // Epoch marker on time axis — orange bar + triangle
  tCtx.fillStyle = 'rgba(251,146,60,0.5)';
  tCtx.fillRect(mx0, 0, mx1 - mx0, 3);

  const mxMid = (mx0 + mx1) / 2;
  const triH = 5, triW = 6;
  tCtx.fillStyle = 'rgba(251,146,60,0.7)';
  tCtx.beginPath();
  tCtx.moveTo(mxMid - triW, triH);
  tCtx.lineTo(mxMid + triW, triH);
  tCtx.lineTo(mxMid, 0);
  tCtx.closePath();
  tCtx.fill();
}
