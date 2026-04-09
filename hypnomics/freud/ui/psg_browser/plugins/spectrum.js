// =============================================================================
// Plugin: Spectrum — PSD via Welch's method for current fast window
// =============================================================================

(function() {
  let xMinHz = 0.5;
  let xMaxHz = 20;
  let normalized = true;
  let logScale = false;
  let yMinManual = -4;  // used when density+log
  let yMaxManual = 0.2; // used when density+!log

  // Right Y axis (band ratios)
  let rLogScale = true;
  let rYMinLog = -3;    // ymin when log on (ymax fixed to 0)
  let rYMaxLin = 0.6;   // ymax when log off (ymin fixed to 0)

  // Total power bands: fixed 0.5–30 Hz per hypnomics convention
  const TOTAL_POWER_LO = 0.5;
  const TOTAL_POWER_HI = 30;

  // EEG channel color by region (Frontal→red, Central→orange, Occipital→yellow)
  // Fallback order for other regions: Parietal→green, Temporal→cyan
  const REGION_COLORS = {
    F: '#e05555', C: '#e0923a', O: '#d4b83a',
    P: '#5aad6a', T: '#4aadad', A: '#aa7adb',
  };
  const REGION_FALLBACK = '#b0b0b0';

  // Montage overrides: secondary electrode determines region color
  const MONTAGE_OVERRIDES = { 'Pz-Oz': 'O', 'Fpz-Cz': 'F' };

  function _getEEGStyle(chName) {
    const stripped = chName.replace(/^eeg\s*/i, '').trim();
    // Check montage overrides
    const overrideRegion = MONTAGE_OVERRIDES[stripped];
    // Determine region from first letter of electrode
    const bare = stripped.split('-')[0].trim();
    const regionChar = overrideRegion || bare.charAt(0).toUpperCase();
    const color = REGION_COLORS[regionChar] || REGION_FALLBACK;
    // M1 reference → dashed, M2 or no ref → solid
    const dashed = /M1/i.test(chName);
    return { color, dashed };
  }

  registerPlugin({
    name: 'spectrum',
    category: 'Analysis',
    description: 'Power spectral density (Welch) for EEG channels',
    channels: ['EEG'],

    buildControls(container) {
      const _sep = () => { const s = document.createElement('span'); s.style.cssText = 'color:var(--border);margin:0 4px;user-select:none'; s.textContent = '|'; return s; };
      const _dash = () => { const s = document.createElement('span'); s.textContent = '–'; s.style.cssText = 'margin:0 2px;user-select:none'; return s; };
      const _toggle = (label, checked, onChange) => {
        const lbl = document.createElement('label'); lbl.textContent = label; lbl.style.cursor = 'pointer';
        const wrap = document.createElement('label'); wrap.className = 'toggle-switch'; wrap.style.marginLeft = '2px';
        const chk = document.createElement('input'); chk.type = 'checkbox'; chk.checked = checked;
        const sl = document.createElement('span'); sl.className = 'toggle-slider';
        wrap.appendChild(chk); wrap.appendChild(sl);
        chk.onchange = () => onChange(chk.checked);
        return { lbl, wrap, chk };
      };

      // === Row 1: X range + Density ===
      const row1 = document.createElement('div'); row1.className = 'analysis-ctrl-row';

      const xLabel = document.createElement('label'); xLabel.textContent = 'X';
      const xMinSl = document.createElement('input');
      xMinSl.type = 'range'; xMinSl.min = '0'; xMinSl.max = '5'; xMinSl.step = '0.5';
      xMinSl.value = xMinHz; xMinSl.style.maxWidth = '50px';
      const xMinV = document.createElement('span'); xMinV.className = 'ctrl-val'; xMinV.textContent = xMinHz + '';
      xMinSl.oninput = () => { xMinHz = parseFloat(xMinSl.value); xMinV.textContent = xMinHz + ''; _renderAnalysisPlugin(); };

      const xMaxSl = document.createElement('input');
      xMaxSl.type = 'range'; xMaxSl.min = '10'; xMaxSl.max = '50'; xMaxSl.step = '5'; xMaxSl.value = xMaxHz;
      const xMaxV = document.createElement('span'); xMaxV.className = 'ctrl-val'; xMaxV.textContent = xMaxHz + ' Hz';
      xMaxSl.oninput = () => { xMaxHz = parseInt(xMaxSl.value); xMaxV.textContent = xMaxHz + ' Hz'; _renderAnalysisPlugin(); };

      const density = _toggle('Density', normalized, v => { normalized = v; _syncLeftY(); _renderAnalysisPlugin(); });

      [xLabel, xMinSl, xMinV, _dash(), xMaxSl, xMaxV, _sep(), density.lbl, density.wrap].forEach(e => row1.appendChild(e));
      container.appendChild(row1);

      // === Row 2: Left Y (PSD) ===
      const row2 = document.createElement('div'); row2.className = 'analysis-ctrl-row';

      const yLabel = document.createElement('label'); yLabel.textContent = 'L'; yLabel.style.color = _css('--text-dim');
      const yMinSl = document.createElement('input');
      yMinSl.type = 'range'; yMinSl.min = '-6'; yMinSl.max = '-1'; yMinSl.step = '0.5';
      yMinSl.value = yMinManual; yMinSl.style.maxWidth = '50px';
      const yMinV = document.createElement('span'); yMinV.className = 'ctrl-val'; yMinV.textContent = yMinManual + '';
      yMinSl.oninput = () => { yMinManual = parseFloat(yMinSl.value); yMinV.textContent = yMinManual + ''; _renderAnalysisPlugin(); };

      const yMaxSl = document.createElement('input');
      yMaxSl.type = 'range'; yMaxSl.min = '0.05'; yMaxSl.max = '0.8'; yMaxSl.step = '0.05'; yMaxSl.value = yMaxManual;
      const yMaxV = document.createElement('span'); yMaxV.className = 'ctrl-val'; yMaxV.textContent = yMaxManual + '';
      yMaxSl.oninput = () => { yMaxManual = parseFloat(yMaxSl.value); yMaxV.textContent = yMaxManual + ''; _renderAnalysisPlugin(); };

      const log = _toggle('Log', logScale, v => { logScale = v; _syncLeftY(); _renderAnalysisPlugin(); });

      [yLabel, yMinSl, yMinV, _dash(), yMaxSl, yMaxV, _sep(), log.lbl, log.wrap].forEach(e => row2.appendChild(e));
      container.appendChild(row2);

      // === Row 3: Right Y (Band ratios) ===
      const row3 = document.createElement('div'); row3.className = 'analysis-ctrl-row';

      const rLabel = document.createElement('label'); rLabel.textContent = 'R'; rLabel.style.color = _css('--accent');
      const rMinSl = document.createElement('input');
      rMinSl.type = 'range'; rMinSl.min = '-6'; rMinSl.max = '-0.5'; rMinSl.step = '0.5';
      rMinSl.value = rYMinLog; rMinSl.style.maxWidth = '50px';
      const rMinV = document.createElement('span'); rMinV.className = 'ctrl-val'; rMinV.textContent = rYMinLog + '';
      rMinSl.oninput = () => { rYMinLog = parseFloat(rMinSl.value); rMinV.textContent = rYMinLog + ''; _renderAnalysisPlugin(); };

      const rMaxSl = document.createElement('input');
      rMaxSl.type = 'range'; rMaxSl.min = '0.1'; rMaxSl.max = '1.0'; rMaxSl.step = '0.05';
      rMaxSl.value = rYMaxLin;
      const rMaxV = document.createElement('span'); rMaxV.className = 'ctrl-val'; rMaxV.textContent = rYMaxLin + '';
      rMaxSl.oninput = () => { rYMaxLin = parseFloat(rMaxSl.value); rMaxV.textContent = rYMaxLin + ''; _renderAnalysisPlugin(); };

      const rLog = _toggle('Log', rLogScale, v => { rLogScale = v; _syncRightY(); _renderAnalysisPlugin(); });

      [rLabel, rMinSl, rMinV, _dash(), rMaxSl, rMaxV, _sep(), rLog.lbl, rLog.wrap].forEach(e => row3.appendChild(e));
      container.appendChild(row3);

      // Sync left Y controls (PSD axis) based on density/log state
      function _syncLeftY() {
        const yMinEnabled = normalized && logScale;
        yMinSl.disabled = !yMinEnabled;
        yMinV.className = 'ctrl-val' + (yMinEnabled ? '' : ' disabled');
        yMinV.textContent = yMinEnabled ? (yMinManual + '') : '0';

        const yMaxEnabled = normalized && !logScale;
        yMaxSl.disabled = !yMaxEnabled;
        yMaxV.className = 'ctrl-val' + (yMaxEnabled ? '' : ' disabled');
        yMaxV.textContent = yMaxEnabled ? (yMaxManual + '') : 'auto';
      }
      // Sync right Y controls (band ratio axis)
      function _syncRightY() {
        // log on: ymax fixed to 0 (1.0 in linear), ymin slider active
        // log off: ymin fixed to 0, ymax slider active
        rMinSl.disabled = !rLogScale;
        rMinV.className = 'ctrl-val' + (rLogScale ? '' : ' disabled');
        rMinV.textContent = rLogScale ? (rYMinLog + '') : '0';

        rMaxSl.disabled = rLogScale;
        rMaxV.className = 'ctrl-val' + (rLogScale ? ' disabled' : '');
        rMaxV.textContent = rLogScale ? '0' : (rYMaxLin + '');
      }
      _syncLeftY();
      _syncRightY();
    },

    render(canvas, info) {
      const ctx = canvas.getContext('2d');
      const W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      if (!psgFile || !channels || channels.length === 0) {
        ctx.fillStyle = _css('--text-dim');
        ctx.font = '11px "JetBrains Mono", monospace';
        ctx.fillText('No data', 10, 20);
        return;
      }

      // (1) EEG channels only, sorted by region (F → C → P → O)
      const eegChs = channels
        .filter(ch => activeChannels.includes(ch.name) && getSignalType(ch.name).type === 'EEG')
        .sort((a, b) => getSignalOrder(a.name) - getSignalOrder(b.name));
      if (eegChs.length === 0) {
        ctx.fillStyle = _css('--text-dim');
        ctx.font = '11px "JetBrains Mono", monospace';
        ctx.fillText('No EEG channels visible', 10, 20);
        return;
      }

      const sfreq = eegChs[0].sfreq || 100;
      const nSamples = Math.round(fastWindowSec * sfreq);

      // (2) Welch parameters
      const segLen = Math.min(nSamples, nextPow2(Math.round(sfreq * 2)));
      const overlap = Math.floor(segLen / 2);
      const nfft = segLen;
      const freqBins = nfft / 2 + 1;
      const freqRes = sfreq / nfft;
      const minFreqIdx = Math.max(1, Math.ceil(xMinHz / freqRes));
      const maxFreqIdx = Math.min(freqBins - 1, Math.floor(xMaxHz / freqRes));
      const xSpan = xMaxHz - xMinHz;

      // Hanning window
      const win = new Float32Array(segLen);
      let winPow = 0;
      for (let i = 0; i < segLen; i++) {
        win[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (segLen - 1)));
        winPow += win[i] * win[i];
      }

      // (3) Compute PSD + per-channel style
      const allPsd = [];
      const allBandRatios = []; // [{ delta, theta, alpha, sigma, beta }, ...]
      const chStyles = []; // { color, dashed }
      const chNames = [];

      for (const ch of eegChs) {
        const result = readChannelData(ch.name, viewStartSec, fastWindowSec);
        const raw = result ? result.data : null;
        if (!raw || raw.length < segLen) continue;

        const psd = new Float64Array(freqBins);
        let nSegs = 0;
        for (let start = 0; start + segLen <= raw.length; start += segLen - overlap) {
          const seg = new Float32Array(nfft);
          for (let i = 0; i < segLen; i++) seg[i] = raw[start + i] * win[i];
          const fft = _realFFT(seg, nfft);
          for (let k = 0; k <= nfft / 2; k++) {
            const re = fft[2 * k], im = fft[2 * k + 1];
            psd[k] += (re * re + im * im);
          }
          nSegs++;
        }
        if (nSegs === 0) continue;

        const scale = 1 / (nSegs * sfreq * winPow);
        for (let k = 0; k < freqBins; k++) psd[k] *= scale;
        for (let k = 1; k < freqBins - 1; k++) psd[k] *= 2;

        // Compute band ratios from raw PSD (before any display normalization)
        // Total power = 0.5–30 Hz per hypnomics convention
        const bandRatios = _computeBandRatios(psd, freqRes, freqBins);

        // Normalize to density (area = 1) if enabled (display only)
        if (normalized) {
          let total = 0;
          for (let k = 0; k < freqBins; k++) total += psd[k] * freqRes;
          if (total > 0) for (let k = 0; k < freqBins; k++) psd[k] /= total;
        }

        allPsd.push(psd);
        allBandRatios.push(bandRatios);
        chStyles.push(_getEEGStyle(ch.name));
        chNames.push(ch.name);
      }

      if (allPsd.length === 0) return;

      // (4) Layout
      const pad = { top: 18, right: 42, bottom: 32, left: 42 };
      const plotW = W - pad.left - pad.right;
      const plotH = H - pad.top - pad.bottom;
      if (plotH < 40) return;

      // Y-axis range
      let yMin, yMax;
      if (normalized && !logScale) {
        // Density + linear: fixed 0 to manual ymax
        yMin = 0;
        yMax = yMaxManual;
      } else if (normalized && logScale) {
        // Density + log: manual ymin to 0
        yMin = yMinManual;
        yMax = 0;
      } else {
        // Absolute: auto range
        yMin = Infinity; yMax = -Infinity;
        for (const psd of allPsd) {
          for (let k = minFreqIdx; k <= maxFreqIdx; k++) {
            if (psd[k] > 0) {
              const v = logScale ? Math.log10(psd[k]) : psd[k];
              if (v < yMin) yMin = v;
              if (v > yMax) yMax = v;
            }
          }
        }
        if (logScale) {
          if (!isFinite(yMin)) { yMin = -10; yMax = 0; }
          const margin = (yMax - yMin) * 0.05 || 0.5;
          yMin -= margin; yMax += margin;
        } else {
          if (!isFinite(yMin)) { yMin = 0; yMax = 1; }
          yMin = 0; yMax *= 1.05;
        }
      }
      const yRange = yMax - yMin || 1;

      const dimColor = _css('--text-dim');
      const borderColor = _css('--border');
      const textColor = _css('--text');

      // (5) AASM band zones
      const bands = [
        { name: 'δ', lo: 0.5, hi: 4, color: 'rgba(60,130,220,0.08)' },
        { name: 'θ', lo: 4, hi: 8, color: 'rgba(50,180,80,0.08)' },
        { name: 'α', lo: 8, hi: 13, color: 'rgba(220,200,40,0.08)' },
        { name: 'σ', lo: 11, hi: 16, color: 'rgba(230,150,40,0.08)' },
        { name: 'β', lo: 16, hi: 30, color: 'rgba(210,60,60,0.08)' },
      ];
      for (const b of bands) {
        if (b.hi <= xMinHz || b.lo >= xMaxHz) continue;
        const x0 = pad.left + (Math.max(b.lo, xMinHz) - xMinHz) / xSpan * plotW;
        const x1 = pad.left + (Math.min(b.hi, xMaxHz) - xMinHz) / xSpan * plotW;
        ctx.fillStyle = b.color;
        ctx.fillRect(x0, pad.top, x1 - x0, plotH);
      }

      // Band labels — inside zones, LaTeX-style italic serif
      ctx.font = 'italic 16px "Times New Roman", serif';
      ctx.textAlign = 'center';
      for (const b of bands) {
        if (b.hi <= xMinHz || b.lo >= xMaxHz) continue;
        const visLo = Math.max(b.lo, xMinHz), visHi = Math.min(b.hi, xMaxHz);
        const xMid = pad.left + ((visLo + visHi) / 2 - xMinHz) / xSpan * plotW;
        ctx.fillStyle = b.color.replace(/[\d.]+\)$/, '0.55)');
        ctx.fillText(b.name, xMid, pad.top + 24);
      }

      // (6) Grid
      ctx.strokeStyle = borderColor;
      ctx.lineWidth = 0.5;
      ctx.setLineDash([2, 3]);

      // Y grid
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.fillStyle = dimColor;
      ctx.textAlign = 'right';
      if (logScale) {
        const yTickStart = Math.ceil(yMin);
        const yTickEnd = Math.floor(yMax);
        for (let t = yTickStart; t <= yTickEnd; t++) {
          const y = pad.top + plotH - ((t - yMin) / yRange) * plotH;
          ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y); ctx.stroke();
          ctx.fillText('1e' + t, pad.left - 4, y + 3);
        }
      } else {
        // Linear ticks — ~5 ticks
        const yStep = _niceStep(yMax, 5);
        for (let t = 0; t <= yMax; t += yStep) {
          const y = pad.top + plotH - (t / yRange) * plotH;
          ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y); ctx.stroke();
          ctx.fillText(t < 0.01 ? t.toExponential(0) : t.toPrecision(2), pad.left - 4, y + 3);
        }
      }

      // Right Y axis (band ratios) — ticks on right side, no grid lines (would clutter)
      ctx.textAlign = 'left';
      let rMin, rMax;
      if (rLogScale) { rMin = rYMinLog; rMax = 0; }
      else { rMin = 0; rMax = rYMaxLin; }
      const rRange = rMax - rMin || 1;
      const rMap = (val) => pad.top + plotH - ((val - rMin) / rRange) * plotH;

      // Right axis line
      ctx.setLineDash([]);
      ctx.strokeStyle = borderColor;
      ctx.beginPath();
      ctx.moveTo(pad.left + plotW, pad.top);
      ctx.lineTo(pad.left + plotW, pad.top + plotH);
      ctx.stroke();

      // Right axis ticks + labels
      ctx.fillStyle = dimColor;
      if (rLogScale) {
        const tStart = Math.ceil(rMin);
        const tEnd = Math.floor(rMax);
        for (let t = tStart; t <= tEnd; t++) {
          const y = rMap(t);
          ctx.beginPath(); ctx.moveTo(pad.left + plotW, y); ctx.lineTo(pad.left + plotW + 3, y); ctx.stroke();
          ctx.fillText('1e' + t, pad.left + plotW + 5, y + 3);
        }
      } else {
        const rStep = _niceStep(rMax, 5);
        for (let t = 0; t <= rMax + 1e-9; t += rStep) {
          const y = rMap(t);
          ctx.beginPath(); ctx.moveTo(pad.left + plotW, y); ctx.lineTo(pad.left + plotW + 3, y); ctx.stroke();
          ctx.fillText(t.toFixed(rStep < 0.1 ? 2 : 1), pad.left + plotW + 5, y + 3);
        }
      }
      ctx.setLineDash([2, 3]);

      // X grid
      ctx.textAlign = 'center';
      ctx.fillStyle = dimColor;
      const xStep = xSpan <= 10 ? 1 : xSpan <= 20 ? 2 : xSpan <= 35 ? 5 : 10;
      const xTickStart = Math.ceil(xMinHz / xStep) * xStep;
      for (let f = xTickStart; f <= xMaxHz; f += xStep) {
        const x = pad.left + (f - xMinHz) / xSpan * plotW;
        ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + plotH); ctx.stroke();
        ctx.fillText(f + '', x, pad.top + plotH + 12);
      }
      ctx.setLineDash([]);

      // X-axis unit — below ticks
      ctx.fillText('Hz', pad.left + plotW / 2, pad.top + plotH + 22);

      // (7) PSD lines
      ctx.lineWidth = 1.4;
      for (let c = 0; c < allPsd.length; c++) {
        const psd = allPsd[c];
        const s = chStyles[c];
        ctx.strokeStyle = s.color;
        if (s.dashed) ctx.setLineDash([4, 3]);
        else ctx.setLineDash([]);
        ctx.beginPath();
        let started = false;
        for (let k = minFreqIdx; k <= maxFreqIdx; k++) {
          const freq = k * freqRes;
          const x = pad.left + (freq - xMinHz) / xSpan * plotW;
          const val = logScale ? (psd[k] > 0 ? Math.log10(psd[k]) : yMin) : psd[k];
          const y = pad.top + plotH - ((val - yMin) / yRange) * plotH;
          if (!started) { ctx.moveTo(x, y); started = true; }
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
      ctx.setLineDash([]);

      // (7.5) Band ratio points (right Y axis)
      const bandDefs = [
        { key: 'delta', lo: 0.5, hi: 4 },
        { key: 'theta', lo: 4, hi: 8 },
        { key: 'alpha', lo: 8, hi: 13 },
        { key: 'sigma', lo: 11, hi: 16 },
        { key: 'beta',  lo: 16, hi: 30 },
      ];
      for (let c = 0; c < allBandRatios.length; c++) {
        const ratios = allBandRatios[c];
        const s = chStyles[c];
        for (const b of bandDefs) {
          // Only show if band is visible in current x range
          if (b.hi <= xMinHz || b.lo >= xMaxHz) continue;
          const visLo = Math.max(b.lo, xMinHz);
          const visHi = Math.min(b.hi, xMaxHz);
          const xMid = pad.left + ((visLo + visHi) / 2 - xMinHz) / xSpan * plotW;
          // Get ratio and map to right axis
          const ratio = ratios[b.key];
          if (ratio == null || ratio <= 0) continue;
          const val = rLogScale ? Math.log10(ratio) : ratio;
          if (val < rMin) continue;
          const y = rMap(Math.min(val, rMax));
          // Draw point (filled circle)
          ctx.fillStyle = s.color;
          ctx.beginPath();
          ctx.arc(xMid, y, 4, 0, Math.PI * 2);
          ctx.fill();
          // Outline for visibility
          ctx.strokeStyle = darkMode ? '#000' : '#fff';
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }

      // (8) Legend — top-right with background
      ctx.font = '11px "JetBrains Mono", monospace';
      const legLineH = 15;
      const legPadX = 8, legPadY = 5;
      // Measure widths
      let legMaxW = 0;
      for (const n of chNames) legMaxW = Math.max(legMaxW, ctx.measureText(n).width);
      const legW = 20 + legMaxW + legPadX * 2;
      const legH = chNames.length * legLineH + legPadY * 2;
      const legX = pad.left + plotW - legW - 4;
      const legY = pad.top + 4;
      // Background
      const bgColor = darkMode ? 'rgba(30,30,30,0.85)' : 'rgba(255,255,255,0.85)';
      ctx.fillStyle = bgColor;
      ctx.strokeStyle = borderColor;
      ctx.lineWidth = 0.5;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.roundRect(legX, legY, legW, legH, 3);
      ctx.fill();
      ctx.stroke();
      // Entries
      for (let c = 0; c < chNames.length; c++) {
        const y = legY + legPadY + c * legLineH + legLineH / 2;
        const s = chStyles[c];
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 2;
        if (s.dashed) ctx.setLineDash([4, 3]);
        else ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(legX + legPadX, y);
        ctx.lineTo(legX + legPadX + 14, y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = textColor;
        ctx.textAlign = 'left';
        ctx.fillText(chNames[c], legX + legPadX + 18, y + 3);
      }

      // Info
      const modeStr = normalized ? 'Density' : 'Absolute';
      info.textContent = `L: Welch PSD [${modeStr}] | R: Band ratio (/0.5–30Hz) | ${segLen}pt, ${freqRes.toFixed(2)}Hz/bin`;
    },

    onNavigate() {
      const canvas = document.getElementById('analysisCanvas');
      const info = document.getElementById('analysisInfo');
      const body = document.getElementById('analysisBody');
      canvas.width = body.clientWidth;
      canvas.height = body.clientHeight - info.offsetHeight;
      info.innerHTML = '';
      this.render(canvas, info);
    }
  });

  // --- Helpers ---

  // Compute relative power ratio per band (band_power / total_power)
  // Total power = integral of PSD from 0.5 to 30 Hz (hypnomics convention)
  function _computeBandRatios(psd, freqRes, freqBins) {
    const integrate = (loHz, hiHz) => {
      const k0 = Math.max(1, Math.ceil(loHz / freqRes));
      const k1 = Math.min(freqBins - 1, Math.floor(hiHz / freqRes));
      let sum = 0;
      for (let k = k0; k <= k1; k++) sum += psd[k] * freqRes;
      return sum;
    };
    const total = integrate(TOTAL_POWER_LO, TOTAL_POWER_HI) + 1e-12;
    return {
      delta: integrate(0.5, 4) / total,
      theta: integrate(4, 8) / total,
      alpha: integrate(8, 13) / total,
      sigma: integrate(11, 16) / total,
      beta:  integrate(16, 30) / total,
    };
  }

  function _niceStep(max, targetTicks) {
    const rough = max / targetTicks;
    const pow = Math.pow(10, Math.floor(Math.log10(rough)));
    const norm = rough / pow;
    const nice = norm < 1.5 ? 1 : norm < 3.5 ? 2 : norm < 7.5 ? 5 : 10;
    return nice * pow;
  }

  function _css(prop) {
    return getComputedStyle(document.documentElement).getPropertyValue(prop).trim();
  }

  function nextPow2(n) {
    let p = 1;
    while (p < n) p <<= 1;
    return p;
  }

  function _realFFT(x, N) {
    const buf = new Float32Array(2 * N);
    for (let i = 0; i < N; i++) buf[2 * i] = x[i];

    for (let i = 1, j = 0; i < N; i++) {
      let bit = N >> 1;
      while (j & bit) { j ^= bit; bit >>= 1; }
      j ^= bit;
      if (i < j) {
        let t = buf[2*i]; buf[2*i] = buf[2*j]; buf[2*j] = t;
        t = buf[2*i+1]; buf[2*i+1] = buf[2*j+1]; buf[2*j+1] = t;
      }
    }

    for (let len = 2; len <= N; len *= 2) {
      const half = len / 2;
      const angle = -2 * Math.PI / len;
      const wRe = Math.cos(angle), wIm = Math.sin(angle);
      for (let i = 0; i < N; i += len) {
        let curRe = 1, curIm = 0;
        for (let j = 0; j < half; j++) {
          const a = 2 * (i + j), b = 2 * (i + j + half);
          const tRe = curRe * buf[b] - curIm * buf[b+1];
          const tIm = curRe * buf[b+1] + curIm * buf[b];
          buf[b] = buf[a] - tRe; buf[b+1] = buf[a+1] - tIm;
          buf[a] += tRe; buf[a+1] += tIm;
          const tmpRe = curRe * wRe - curIm * wIm;
          curIm = curRe * wIm + curIm * wRe;
          curRe = tmpRe;
        }
      }
    }

    const out = new Float32Array((N / 2 + 1) * 2);
    for (let i = 0; i < out.length; i++) out[i] = buf[i];
    return out;
  }
})();
