// =============================================================================
// dsp.js — Pure-JS DSP: Butterworth IIR filter design + filtfilt
// =============================================================================

// AASM Part III recommended filter bands
const AASM_FILTER_BANDS = {
  EEG: { lo: 0.3, hi: 35 },
  EOG: { lo: 0.3, hi: 35 },
  EMG: { lo: 10, hi: 100 },
  ECG: { lo: 0.3, hi: 100 },
};

// (1) 2nd-order Butterworth coefficient computation
//     Returns { b: Float64Array(3), a: Float64Array(3) }
function butterCoeffs(Wn, btype) {
  // Wn = normalized cutoff frequency (0 < Wn < 1, where 1 = Nyquist)
  // btype = 'lowpass' | 'highpass'
  // Uses bilinear transform with frequency pre-warping for 2nd-order Butterworth

  const gamma = Math.tan(Math.PI * Wn / 2);
  const g2 = gamma * gamma;
  const sqrt2 = Math.SQRT2;

  let b, a;
  if (btype === 'lowpass') {
    const D = g2 + sqrt2 * gamma + 1;
    b = new Float64Array([g2 / D, 2 * g2 / D, g2 / D]);
    a = new Float64Array([1, 2 * (g2 - 1) / D, (g2 - sqrt2 * gamma + 1) / D]);
  } else {
    // highpass: apply LP-to-HP transform
    const D = g2 + sqrt2 * gamma + 1;
    b = new Float64Array([1 / D, -2 / D, 1 / D]);
    a = new Float64Array([1, 2 * (g2 - 1) / D, (g2 - sqrt2 * gamma + 1) / D]);
  }
  return { b, a };
}

// (2) IIR filter — direct-form-II transposed (forward only)
//     Uses Float64 internal state for numerical stability
function lfilter(b, a, x) {
  const n = x.length;
  const y = new Float32Array(n);
  // State variables (direct-form-II transposed, 2nd order)
  let z0 = 0, z1 = 0;
  const b0 = b[0], b1 = b[1], b2 = b[2];
  const a1 = a[1], a2 = a[2];

  for (let i = 0; i < n; i++) {
    const xi = x[i];
    const yi = b0 * xi + z0;
    z0 = b1 * xi - a1 * yi + z1;
    z1 = b2 * xi - a2 * yi;
    y[i] = yi;
  }
  return y;
}

// (3) Zero-phase filter (filtfilt): forward + backward, with reflect padding
function filtfilt(b, a, x) {
  const n = x.length;
  if (n < 12) return new Float32Array(x); // too short to filter

  // (3.1) Reflect-pad edges to reduce transients
  const padLen = Math.min(3 * 3, Math.floor(n / 3)); // 9 samples or 1/3 of data
  const padded = new Float32Array(n + 2 * padLen);

  // Left reflection: x[padLen], x[padLen-1], ..., x[1]
  for (let i = 0; i < padLen; i++) {
    padded[i] = 2 * x[0] - x[padLen - i];
  }
  // Copy original
  padded.set(x, padLen);
  // Right reflection
  for (let i = 0; i < padLen; i++) {
    padded[n + padLen + i] = 2 * x[n - 1] - x[n - 2 - i];
  }

  // (3.2) Forward pass
  let fwd = lfilter(b, a, padded);

  // (3.3) Reverse in place
  fwd.reverse();

  // (3.4) Backward pass
  let bwd = lfilter(b, a, fwd);

  // (3.5) Reverse again and trim padding
  bwd.reverse();
  return bwd.subarray(padLen, padLen + n);
}

// (4) Apply AASM bandpass filter to a channel
//     Cascade: highpass(lo) → lowpass(hi), both via filtfilt (zero-phase)
function applyAASMFilter(data, sfreq, signalType) {
  const band = AASM_FILTER_BANDS[signalType];
  if (!band) return data; // No filter for Resp, SpO2, Limb, Other

  const nyq = sfreq / 2;
  let lo = band.lo;
  let hi = band.hi;

  // Clamp to valid range
  if (hi >= nyq) hi = nyq * 0.99;
  if (lo <= 0) lo = 0.01;
  if (lo >= hi) return data; // invalid band

  // (4.1) Highpass at lo
  const hp = butterCoeffs(lo / nyq, 'highpass');
  let y = filtfilt(hp.b, hp.a, data);

  // (4.2) Lowpass at hi
  const lp = butterCoeffs(hi / nyq, 'lowpass');
  y = filtfilt(lp.b, lp.a, y);

  return y;
}
