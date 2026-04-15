// =============================================================================
// data.js — HDF5 data reading utilities (Float16 fallback, channel data)
// =============================================================================

// (2.5) Float16 -> Float32 conversion utility
function float16ToFloat32Array(uint8Buf) {
  const u16 = new Uint16Array(uint8Buf.buffer, uint8Buf.byteOffset, uint8Buf.byteLength / 2);
  const f32 = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const sign = (h >> 15) & 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x3ff;
    if (exp === 0) {
      f32[i] = (sign ? -1 : 1) * (2 ** -14) * (frac / 1024);
    } else if (exp === 0x1f) {
      f32[i] = frac ? NaN : (sign ? -Infinity : Infinity);
    } else {
      f32[i] = (sign ? -1 : 1) * (2 ** (exp - 15)) * (1 + frac / 1024);
    }
  }
  return f32;
}

// (3) Init h5wasm
async function initH5() {
  await h5wasm.ready;
  h5Ready = true;
}

// (3.1) Read dataset with Float16 fallback
function readDataset(ds, ranges) {
  const meta = ds.metadata;
  const isFloat16 = meta.type === 1 && meta.size === 2;  // H5T_FLOAT, 2 bytes
  if (!isFloat16) {
    return ranges ? ds.slice(ranges) : ds.value;
  }
  // Manual read: get raw bytes, convert Float16 -> Float32
  const Module2 = h5wasm.Module;
  const shape = meta.shape;
  let total_size, count, offset, strides;
  if (ranges) {
    const ndim = shape.length;
    count = new BigInt64Array(ndim);
    offset = new BigInt64Array(ndim);
    strides = new BigInt64Array(ndim);
    for (let i = 0; i < ndim; i++) {
      const r = ranges[i] || [0, shape[i]];
      offset[i] = BigInt(r[0]);
      count[i] = BigInt(r[1] - r[0]);
      strides[i] = BigInt(r[2] || 1);
    }
    total_size = count.reduce((a, b) => a * b, 1n);
  } else {
    const ndim = shape.length;
    count = new BigInt64Array(ndim);
    offset = new BigInt64Array(ndim);
    strides = new BigInt64Array(ndim);
    for (let i = 0; i < ndim; i++) {
      count[i] = BigInt(shape[i]);
      offset[i] = 0n;
      strides[i] = 1n;
    }
    total_size = count.reduce((a, b) => a * b, 1n);
  }
  const nbytes = 2 * Number(total_size);
  const data_ptr = Module2._malloc(nbytes);
  try {
    Module2.get_dataset_data(ds.file_id, ds.path, count, offset, strides, BigInt(data_ptr));
    const raw = Module2.HEAPU8.slice(data_ptr, data_ptr + nbytes);
    return float16ToFloat32Array(raw);
  } finally {
    Module2._free(data_ptr);
  }
}

// (7) Read channel data for a given time range
function readChannelData(chName, tStart, tDuration) {
  // Branch for trajectory pseudo-channels — stored in .traj.h5 at a much
  // lower effective sfreq (= 1/tr). We slice by epoch index and return a
  // shape-compatible object for drawTraces.
  if (typeof chName === 'string' && chName.startsWith('traj::')) {
    return readTrajData(chName, tStart, tDuration);
  }
  const ch = channels.find(c => c.name === chName);
  if (!ch) return null;
  const sfreq = ch.sfreq;
  const startSample = Math.floor(tStart * sfreq);
  const numSamples = Math.floor(tDuration * sfreq);
  const endSample = Math.min(startSample + numSamples, ch.length);
  if (startSample >= ch.length) return null;

  const cacheKey = `${chName}:${tStart}:${tDuration}:${filterEnabled ? 'F' : 'R'}`;
  let cached = epochCache[cacheKey];
  if (!cached) {
    let data;
    if (filterEnabled && filteredData[chName]) {
      // Zero-copy slice from precomputed filtered array
      data = filteredData[chName].subarray(startSample, endSample);
    } else {
      const ds = psgFile.get(`signals/${chName}`);
      data = readDataset(ds, [[startSample, endSample]]);
    }
    let sum = 0, sum2 = 0;
    for (let i = 0; i < data.length; i++) { sum += data[i]; sum2 += data[i] * data[i]; }
    const mean = sum / data.length;
    const std = Math.sqrt(sum2 / data.length - mean * mean) || 1;
    cached = { data, mean, std, numSamples };
    epochCache[cacheKey] = cached;
  }
  return { ...cached, ch, color: getSignalColor(chName), sfreq: ch.sfreq, chName };
}

// (7.1) Read a slice of a trajectory probe as a pseudo-signal. Returns the
// same shape as readChannelData so the slow renderer can consume it.
function readTrajData(chName, tStart, tDuration) {
  if (!trajFile) return null;
  const sig = trajSignals.find(s => s.name === chName);
  if (!sig) return null;

  const tr = sig.tr;
  const nEpochs = sig.length;
  // Mirror readChannelData's floor convention: iStart = floor, numSamples =
  // how many samples fit in the full window, iEnd = clipped at nEpochs.
  // If the window overruns the recording end, data.length < numSamples and
  // drawTraces correctly leaves the tail blank.
  const iStart = Math.max(0, Math.floor(tStart / tr));
  const numSamples = Math.floor(tDuration / tr);
  const iEnd = Math.min(iStart + numSamples, nEpochs);
  if (iEnd <= iStart) return null;

  const isLog = trajLogEnabled.has(chName);
  const cacheKey = `traj:${chName}:${iStart}:${iEnd}:${isLog ? 'L' : 'R'}`;
  let cached = epochCache[cacheKey];
  if (!cached) {
    let data;
    try {
      const ds = trajFile.get(`traj/${sig.ch}/${sig.tr}s/${sig.pk}`);
      data = readDataset(ds, [[iStart, iEnd]]);
    } catch(e) {
      console.warn('readTrajData failed:', e);
      return null;
    }
    // Force Float32Array for downstream code that expects typed arrays
    if (!(data instanceof Float32Array)) {
      const f32 = new Float32Array(data.length);
      for (let i = 0; i < data.length; i++) f32[i] = data[i];
      data = f32;
    }
    // log₁₀ transform — clamp to 1e-12 so zeros / negatives become -12
    // instead of -Infinity / NaN (keeps the polyline continuous).
    if (isLog) {
      const EPS = 1e-12;
      const logged = new Float32Array(data.length);
      for (let i = 0; i < data.length; i++) {
        logged[i] = Math.log10(Math.max(data[i], EPS));
      }
      data = logged;
    }
    let sum = 0, sum2 = 0;
    for (let i = 0; i < data.length; i++) { sum += data[i]; sum2 += data[i] * data[i]; }
    const mean = sum / data.length;
    const std = Math.sqrt(sum2 / data.length - mean * mean) || 1;
    cached = { data, mean, std, numSamples };
    epochCache[cacheKey] = cached;
  }
  // `ch`: minimal stand-in object for rendering/label code. Unit is 'a.u.'
  // (or 'log a.u.' when log mode is on) to keep formatYScale out of the
  // voltage branch. `d.ch.unit` is picked up by drawTraces.
  const unit = isLog ? 'log a.u.' : 'a.u.';
  const pseudoCh = { name: chName, unit, sfreq: 1 / tr, length: nEpochs };
  return { ...cached, ch: pseudoCh, color: getSignalColor(chName), sfreq: 1 / tr, chName };
}

// (7.2) Walk activeChannels and ensure every active traj signal has a
// precomputed p99 in globalYmax. Cheap (just a lookup) when nothing is
// missing; called at the top of drawWaveforms() whenever autoScaleGlobal
// is on so toggling a checkbox or flipping the auto-scale switch is
// enough to trigger the computation.
function ensureTrajGlobalYmax() {
  if (!trajFile) return;
  for (const name of activeChannels) {
    if (typeof name !== 'string' || !name.startsWith('traj::')) continue;
    if (globalYmax[name] != null) continue;
    const sig = trajSignals.find(s => s.name === name);
    if (sig) computeTrajGlobalYmax(sig);
  }
}

// (7.3) Hypnogram background overlay — read full-night traj once, apply
// log₁₀ if needed, cache in state. Called from the middle-click handler.
// Pass null to clear. Pass the same name again to toggle off.
// Show or hide the settings-panel tuning controls depending on whether
// an overlay is currently active.
function updateTrajOverlayControlsVisibility() {
  const wrap = document.getElementById('trajOverlayControls');
  if (!wrap) return;
  wrap.style.display = hypnoTrajName ? '' : 'none';
}

// Force the currently-displayed hypnogram overlay to re-read its data,
// picking up any state change that affects values (e.g., a log-mode flip
// for this probe). A no-op if nothing is overlaid. Works by clearing the
// current name and re-invoking setHypnoTrajOverlay; setHypnoTrajOverlay's
// toggle-off short-circuit is avoided because hypnoTrajName is null at
// the time of the re-entry.
function reloadHypnoTrajOverlay() {
  if (!hypnoTrajName) return;
  const name = hypnoTrajName;
  hypnoTrajName = null;
  hypnoTrajRawData = null;
  setHypnoTrajOverlay(name);
}

function setHypnoTrajOverlay(name) {
  if (!trajFile || !name) {
    hypnoTrajName = null;
    hypnoTrajRawData = null;
    updateTrajOverlayControlsVisibility();
    try { drawHypnogram(); } catch(_) {}
    return;
  }
  // Toggle off if already showing this name
  if (hypnoTrajName === name) {
    hypnoTrajName = null;
    hypnoTrajRawData = null;
    updateTrajOverlayControlsVisibility();
    try { drawHypnogram(); } catch(_) {}
    return;
  }
  const sig = trajSignals.find(s => s.name === name);
  if (!sig) { console.warn('setHypnoTrajOverlay: unknown name', name); return; }

  try {
    const ds = trajFile.get(`traj/${sig.ch}/${sig.tr}s/${sig.pk}`);
    let data = readDataset(ds);
    if (!(data instanceof Float32Array)) {
      const f32 = new Float32Array(data.length);
      for (let i = 0; i < data.length; i++) f32[i] = data[i];
      data = f32;
    }
    // Match readTrajData's log transform so the overlay and the slow trace
    // use the same values.
    if (trajLogEnabled.has(name)) {
      const EPS = 1e-12;
      const logged = new Float32Array(data.length);
      for (let i = 0; i < data.length; i++) {
        logged[i] = Math.log10(Math.max(data[i], EPS));
      }
      data = logged;
    }
    hypnoTrajRawData = data;
    hypnoTrajTr = sig.tr;
    hypnoTrajName = name;
    recomputeHypnoTrajPercentiles();
  } catch(e) {
    console.warn('setHypnoTrajOverlay failed:', e);
    hypnoTrajName = null;
    hypnoTrajRawData = null;
  }
  updateTrajOverlayControlsVisibility();
  try { drawHypnogram(); } catch(_) {}
}

// Recompute p_low / p_high from the cached full-night data using the
// current hypnoTrajPctMargin. Called when the margin slider moves or
// when a new overlay is loaded.
function recomputeHypnoTrajPercentiles() {
  if (!hypnoTrajRawData || hypnoTrajRawData.length === 0) return;
  const n = hypnoTrajRawData.length;
  // Copy and sort (tiny array — ~10^3–10^4 values)
  const sorted = Float32Array.from(hypnoTrajRawData);
  sorted.sort();
  const pct = Math.max(0, Math.min(50, hypnoTrajPctMargin)) / 100;
  const iLow = Math.max(0, Math.min(n - 1, Math.floor(n * pct)));
  const iHigh = Math.max(0, Math.min(n - 1, Math.floor(n * (1 - pct))));
  hypnoTrajPLow = sorted[iLow];
  hypnoTrajPHigh = sorted[iHigh];
  // Degenerate case: flat data → give a small band to avoid divide-by-zero
  if (!(hypnoTrajPHigh > hypnoTrajPLow)) {
    hypnoTrajPHigh = hypnoTrajPLow + 1;
  }
}

// Compute the p99 of |value - mean| for a full trajectory and stash it in
// globalYmax[sig.name]. Respects the per-sig log mode so the amplitude
// matches what the renderer actually draws.
function computeTrajGlobalYmax(sig) {
  if (!trajFile || !sig) return;
  try {
    const ds = trajFile.get(`traj/${sig.ch}/${sig.tr}s/${sig.pk}`);
    const raw = readDataset(ds);
    const n = raw.length;
    if (n === 0) return;

    // Respect log₁₀ mode: compute the p99 on the transformed values so the
    // autoscaled amplitude matches what the renderer actually draws.
    let values;
    if (trajLogEnabled.has(sig.name)) {
      const EPS = 1e-12;
      values = new Float32Array(n);
      for (let i = 0; i < n; i++) values[i] = Math.log10(Math.max(raw[i], EPS));
    } else {
      values = raw;
    }

    let sum = 0;
    for (let i = 0; i < n; i++) sum += values[i];
    const mean = sum / n;
    const absVals = new Float32Array(n);
    for (let i = 0; i < n; i++) absVals[i] = Math.abs(values[i] - mean);
    absVals.sort();
    const idx = Math.min(Math.floor(n * 0.99), n - 1);
    globalYmax[sig.name] = absVals[idx] || 1;
  } catch(e) {
    console.warn('computeTrajGlobalYmax failed for', sig.name, e);
  }
}
