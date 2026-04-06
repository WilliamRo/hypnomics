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
  const ch = channels.find(c => c.name === chName);
  if (!ch) return null;
  const sfreq = ch.sfreq;
  const startSample = Math.floor(tStart * sfreq);
  const numSamples = Math.floor(tDuration * sfreq);
  const endSample = Math.min(startSample + numSamples, ch.length);
  if (startSample >= ch.length) return null;

  const cacheKey = `${chName}:${tStart}:${tDuration}`;
  let cached = epochCache[cacheKey];
  if (!cached) {
    const ds = psgFile.get(`signals/${chName}`);
    const data = readDataset(ds, [[startSample, endSample]]);
    let sum = 0, sum2 = 0;
    for (let i = 0; i < data.length; i++) { sum += data[i]; sum2 += data[i] * data[i]; }
    const mean = sum / data.length;
    const std = Math.sqrt(sum2 / data.length - mean * mean) || 1;
    cached = { data, mean, std, numSamples };
    epochCache[cacheKey] = cached;
  }
  return { ...cached, ch, color: getSignalColor(chName), sfreq: ch.sfreq, chName };
}
