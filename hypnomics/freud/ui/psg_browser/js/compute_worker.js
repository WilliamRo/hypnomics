// =============================================================================
// compute_worker.js — Web Worker for heavy signal computation
// =============================================================================

// (1) Load h5wasm inside worker
let h5Module = null;

async function initH5() {
  importScripts('../lib/h5wasm.js');
  h5Module = await h5wasm.ready;
}

// (2) Compute global 99th percentile for all channels
async function computeGlobalScale(arrayBuffer) {
  if (!h5Module) await initH5();

  const { FS, File } = h5Module;
  const fname = '_worker_psg.h5';
  FS.writeFile(fname, new Uint8Array(arrayBuffer));
  const f = new File(fname, 'r');

  const sigGroup = f.get('signals');
  const chNames = sigGroup.keys();
  const results = {};
  const n = chNames.length;

  for (let ci = 0; ci < n; ci++) {
    const name = chNames[ci];
    postMessage({ type: 'progress', channel: name, current: ci + 1, total: n });

    const ds = f.get(`signals/${name}`);
    const data = ds.value;

    // (2.1) Subsample: max 100k samples
    const step = Math.max(1, Math.floor(data.length / 100000));
    const nSub = Math.ceil(data.length / step);
    let sum = 0;
    for (let i = 0; i < data.length; i += step) sum += data[i];
    const mean = sum / nSub;

    // (2.2) Compute 99th percentile of |data - mean|
    const absVals = new Float32Array(nSub);
    let j = 0;
    for (let i = 0; i < data.length; i += step) absVals[j++] = Math.abs(data[i] - mean);
    absVals.sort();
    const idx = Math.min(Math.floor(nSub * 0.99), nSub - 1);
    results[name] = absVals[idx] || 1;
  }

  f.close();
  FS.unlink(fname);
  return results;
}

// (3) Message handler
self.onmessage = async (e) => {
  const { type, arrayBuffer } = e.data;

  if (type === 'computeGlobalScale') {
    try {
      const results = await computeGlobalScale(arrayBuffer);
      postMessage({ type: 'done', results });
    } catch (err) {
      postMessage({ type: 'error', message: err.message });
    }
  }
};
