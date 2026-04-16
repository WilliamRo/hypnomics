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

  if (totalEpochs <= 0) return;

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

  // (6.0) Traj background overlay — middle-click a traj label to toggle.
  // Whole-night trajectory drawn in gray, scaled so the p_margin /
  // (100 - p_margin) percentiles align with the N3 / W rows. Invertible.
  if (hypnoTrajName && hypnoTrajRawData && hypnoTrajRawData.length > 0 &&
      hypnoTrajTr > 0) {
    const n = hypnoTrajRawData.length;
    const tr = hypnoTrajTr;
    const denom = (hypnoTrajPHigh - hypnoTrajPLow) || 1;
    const yW = stageYFromRow(0);
    const yN3 = stageYFromRow(STAGE_ROWS - 1);
    // Map normalized value [0..1] to y: 0 → N3 row, 1 → W row
    // (inverted: 0 → W row, 1 → N3 row)
    const yForValue = hypnoTrajInvert
      ? (v) => yW + ((v - hypnoTrajPLow) / denom) * (yN3 - yW)
      : (v) => yN3 - ((v - hypnoTrajPLow) / denom) * (yN3 - yW);

    ctx.strokeStyle = darkMode ? 'rgba(200,200,200,0.38)' : 'rgba(80,80,80,0.35)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    let started = false;
    // Skip samples whose time falls outside the visible window (cheap clip)
    const iFirst = Math.max(0, Math.floor(tStart / tr));
    const iLast  = Math.min(n - 1, Math.ceil(tEnd / tr));
    // Downsample when there are more samples than pixels (keeps polyline
    // cost bounded for high-resolution trajectories).
    const span = iLast - iFirst + 1;
    if (span <= w * 2) {
      for (let i = iFirst; i <= iLast; i++) {
        const v = hypnoTrajRawData[i];
        if (!isFinite(v)) { started = false; continue; }
        const x = timeToX(i * tr);
        const y = yForValue(v);
        if (!started) { ctx.moveTo(x, y); started = true; }
        else ctx.lineTo(x, y);
      }
    } else {
      // Min/max decimation per pixel column
      const samplesPerPx = span / w;
      for (let px = 0; px < w; px++) {
        const i0 = iFirst + Math.floor(px * samplesPerPx);
        const i1 = Math.min(iLast, iFirst + Math.floor((px + 1) * samplesPerPx));
        let mn = Infinity, mx = -Infinity;
        for (let i = i0; i <= i1; i++) {
          const v = hypnoTrajRawData[i];
          if (!isFinite(v)) continue;
          if (v < mn) mn = v;
          if (v > mx) mx = v;
        }
        if (mn === Infinity) { started = false; continue; }
        const yMin = yForValue(mx);
        const yMax = yForValue(mn);
        if (!started) { ctx.moveTo(px, yMin); started = true; }
        else ctx.lineTo(px, yMin);
        ctx.lineTo(px, yMax);
      }
    }
    ctx.stroke();
  }

  // Draw staircase (only when annotations exist AND showHypnogram is true)
  if (annotations && showHypnogram) {
    const intervals = annotations.intervals;
    const labels = annotations.labels;
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
