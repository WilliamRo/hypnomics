# Copyright 2024 Wei Luo. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =-===========================================================================-
"""PSG recording protocol — HDF5-based format for polysomnography data.

File layout (.psg.h5):
  signals/
    <channel_name>/    (dataset, chunked along time axis)
      .attrs: {sfreq: float}
  annotations/
    <key>/
      intervals        (float64, [N, 2])
      labels           (int32, [N])       # omitted for event-only
      .attrs: {label_names: [...]}        # or {is_event: True}
  meta/
    .attrs: {scalar keys}
    <key>              (dataset for arrays)
  .attrs: {label: str}
"""
import h5py
import numpy as np

from roma import console



class PSG:
  """Lazy reader/writer for .psg.h5 files."""

  EXT = '.psg.h5'
  EPOCH_LEN = 30  # seconds

  def __init__(self, path):
    """Open a .psg.h5 file for lazy reading.

    Args:
      path: path to .psg.h5 file.
    """
    self._path = path
    self._file = None

  # region: Properties

  @property
  def file(self):
    if self._file is None:
      self._file = h5py.File(self._path, 'r')
    return self._file

  @property
  def label(self):
    return self.file.attrs.get('label', '')

  @property
  def channel_names(self):
    if 'signals' not in self.file: return []
    return list(self.file['signals'].keys())

  @property
  def meta(self):
    if 'meta' not in self.file: return {}
    grp = self.file['meta']
    result = dict(grp.attrs)
    for k in grp:
      result[k] = grp[k][:]
    return result

  @property
  def annotation_keys(self):
    if 'annotations' not in self.file: return []
    return list(self.file['annotations'].keys())

  @property
  def duration(self):
    """Total recording duration in seconds (from longest channel)."""
    chs = self.channel_names
    if not chs: return 0.0
    return max(
      self.file['signals'][ch].shape[0] / self.sfreq(ch) for ch in chs)

  # endregion: Properties

  # region: Reading

  def sfreq(self, channel=None):
    """Get sampling frequency for a channel (or first channel if None)."""
    chs = self.channel_names
    assert len(chs) > 0, 'No channels in this PSG file'
    if channel is None:
      channel = chs[0]
    return float(self.file['signals'][channel].attrs['sfreq'])

  def read(self, channels=None, tmin=None, tmax=None):
    """Read signal data, optionally cropped by time.

    Time boundaries must be multiples of 30 seconds (epoch-aligned).

    Args:
      channels: list of channel names (None = all).
      tmin: start time in seconds (None = beginning). Must be multiple of 30.
      tmax: end time in seconds (None = end). Must be multiple of 30.

    Returns:
      np.ndarray of shape [L, C], dtype matches stored dtype.
      All requested channels must have the same sfreq.
    """
    if channels is None:
      channels = self.channel_names
    assert len(channels) > 0, 'No channels specified'

    # Validate epoch alignment
    if tmin is not None:
      assert tmin % self.EPOCH_LEN == 0, (
        f'tmin ({tmin}) must be a multiple of {self.EPOCH_LEN}s')
    if tmax is not None:
      assert tmax % self.EPOCH_LEN == 0, (
        f'tmax ({tmax}) must be a multiple of {self.EPOCH_LEN}s')
    if tmin is not None and tmax is not None:
      assert tmin < tmax, f'tmin ({tmin}) must be < tmax ({tmax})'

    # Check sfreq consistency before I/O
    sfreqs = {ch: float(self.file['signals'][ch].attrs['sfreq'])
              for ch in channels}
    unique_sfreqs = set(sfreqs.values())
    assert len(unique_sfreqs) == 1, (
      f'Channels have different sfreqs: {sfreqs}')
    sf = unique_sfreqs.pop()

    # Read data
    arrays = []
    for ch in channels:
      ds = self.file['signals'][ch]
      i_start = int(tmin * sf) if tmin is not None else 0
      i_end = int(tmax * sf) if tmax is not None else ds.shape[0]
      i_start = max(0, i_start)
      i_end = min(ds.shape[0], i_end)
      arrays.append(ds[i_start:i_end])

    # Truncate to shortest (minor rounding differences)
    min_len = min(a.shape[0] for a in arrays)
    arrays = [a[:min_len] for a in arrays]

    return np.stack(arrays, axis=-1)

  def epochs(self, channels=None, epoch_len=30, tmin=None, tmax=None,
             dtype=None):
    """Read fixed-length epochs, lazily via crop read.

    Args:
      channels: list of channel names (None = all).
      epoch_len: epoch duration in seconds (default 30).
      tmin: start time in seconds (None = beginning).
      tmax: end time in seconds (None = end).
      dtype: cast to this dtype (None = keep stored dtype).

    Returns:
      np.ndarray of shape [n_epochs, epoch_samples, n_channels].
    """
    if channels is None:
      channels = self.channel_names

    sfreq = self.sfreq(channels[0])
    samples_per_epoch = int(epoch_len * sfreq)

    data = self.read(channels, tmin=tmin, tmax=tmax)
    n_epochs = data.shape[0] // samples_per_epoch
    data = data[:n_epochs * samples_per_epoch]
    data = data.reshape(n_epochs, samples_per_epoch, -1)

    if dtype is not None:
      data = data.astype(dtype)
    return data

  def get_annotation(self, key='stage Ground-Truth'):
    """Read annotation by key.

    Returns:
      (intervals, labels, label_names) where
        intervals: np.ndarray float64 [N, 2]
        labels: np.ndarray int32 [N] or None (event-only)
        label_names: list of str or []
    """
    assert key in self.annotation_keys, (
      f"Annotation '{key}' not found. Available: {self.annotation_keys}")
    grp = self.file['annotations'][key]
    intervals = grp['intervals'][:]
    if grp.attrs.get('is_event', False):
      return intervals, None, []
    labels = grp['labels'][:]
    assert len(intervals) == len(labels), (
      f'Shape mismatch: {len(intervals)} intervals vs {len(labels)} labels')
    label_names = list(grp.attrs.get('label_names', []))
    return intervals, labels, label_names

  def get_annotation_cropped(self, key='stage Ground-Truth',
                             tmin=None, tmax=None):
    """Read annotation cropped to a time window.

    Returns same format as get_annotation, filtered to [tmin, tmax).
    """
    intervals, labels, label_names = self.get_annotation(key)
    if tmin is None and tmax is None:
      return intervals, labels, label_names

    mask = np.ones(len(intervals), dtype=bool)
    if tmin is not None:
      mask &= intervals[:, 1] > tmin
    if tmax is not None:
      mask &= intervals[:, 0] < tmax

    intervals = intervals[mask]
    if labels is not None:
      labels = labels[mask]

    # Clip interval boundaries
    if tmin is not None:
      intervals[:, 0] = np.maximum(intervals[:, 0], tmin)
    if tmax is not None:
      intervals[:, 1] = np.minimum(intervals[:, 1], tmax)

    return intervals, labels, label_names

  # endregion: Reading

  # region: Writing

  def append_annotation(self, key, intervals, labels=None, label_names=None):
    """Add an annotation to an existing .psg.h5 file.

    Args:
      key: annotation key (e.g., 'stage ModelV1').
      intervals: list/array of (start, end) pairs.
      labels: list/array of int labels (None for event-only).
      label_names: list of str label names.
    """
    # Close read handle before opening for append
    self.close()
    with h5py.File(self._path, 'a') as f:
      if 'annotations' not in f:
        f.create_group('annotations')
      anno_grp = f['annotations']
      assert key not in anno_grp, (
        f"Annotation '{key}' already exists. Remove it first.")
      key_grp = anno_grp.create_group(key)
      key_grp.create_dataset(
        'intervals', data=np.asarray(intervals, dtype=np.float64))
      if labels is not None:
        key_grp.create_dataset(
          'labels', data=np.asarray(labels, dtype=np.int32))
        key_grp.attrs['label_names'] = label_names or []
      else:
        key_grp.attrs['is_event'] = True

    # Reset file handle to pick up changes
    self.close()

  # endregion: Writing

  # region: Creation

  @classmethod
  def from_raw(cls, path, signal_dict, annotations=None, meta=None,
               label='', dtype=np.float32, chunk_seconds=300):
    """Create a .psg.h5 file from raw data.

    Args:
      path: output file path.
      signal_dict: {channel_name: (data_1d, sfreq), ...}
        data_1d: np.ndarray of shape [L].
      annotations: {key: (intervals, labels, label_names), ...}
        intervals: np.ndarray or list of (start, end) pairs in seconds.
        labels: np.ndarray or list of int, or None for event-only.
        label_names: list of str.
      meta: dict of metadata {key: value}.
      label: recording label (e.g., patient ID).
      dtype: storage dtype for signals (default np.float32).
      chunk_seconds: chunk size for HDF5 storage in seconds.

    Returns:
      PSG instance for the created file.
    """
    # Validate signal_dict
    for ch_name, (data, sfreq) in signal_dict.items():
      assert isinstance(data, np.ndarray) and data.ndim == 1, (
        f"Channel '{ch_name}': data must be 1-D ndarray, got shape {data.shape}")
      assert sfreq > 0, (
        f"Channel '{ch_name}': sfreq must be > 0, got {sfreq}")
      assert '/' not in ch_name, (
        f"Channel name '{ch_name}' contains '/' which is invalid for HDF5")

    with h5py.File(path, 'w') as f:
      f.attrs['label'] = label

      # Signals
      sig_grp = f.create_group('signals')
      for ch_name, (data, sfreq) in signal_dict.items():
        data_stored = data.astype(dtype)
        chunk_samples = int(chunk_seconds * sfreq)
        chunks = (min(chunk_samples, len(data_stored)),)
        ds = sig_grp.create_dataset(
          ch_name, data=data_stored, chunks=chunks)
        ds.attrs['sfreq'] = float(sfreq)

      # Annotations
      if annotations:
        anno_grp = f.create_group('annotations')
        for key, anno_tuple in annotations.items():
          key_grp = anno_grp.create_group(key)
          intervals, labels, label_names = anno_tuple
          key_grp.create_dataset(
            'intervals', data=np.asarray(intervals, dtype=np.float64))
          if labels is not None:
            key_grp.create_dataset(
              'labels', data=np.asarray(labels, dtype=np.int32))
            key_grp.attrs['label_names'] = label_names or []
          else:
            key_grp.attrs['is_event'] = True

      # Meta
      if meta:
        meta_grp = f.create_group('meta')
        for k, v in meta.items():
          if isinstance(v, (str, int, float, bool, np.integer, np.floating)):
            meta_grp.attrs[k] = v
          elif isinstance(v, np.ndarray):
            meta_grp.create_dataset(k, data=v)
          else:
            try:
              meta_grp.attrs[k] = v
            except TypeError:
              console.show_status(
                f"Skipped non-serializable meta key '{k}' "
                f"(type: {type(v).__name__})", prompt='[Warning]')

    return cls(path)

  @classmethod
  def from_sg(cls, sg, path, channels=None, dtype=np.float32,
              chunk_seconds=300):
    """Create a .psg.h5 from a SignalGroup.

    Args:
      sg: pictor SignalGroup object.
      path: output file path.
      channels: list of channel names (None = all).
      dtype: storage dtype for signals (default np.float32).
      chunk_seconds: chunk size for HDF5 storage in seconds.

    Returns:
      PSG instance for the created file.
    """
    # Build signal_dict
    signal_dict = {}
    for ds in sg.digital_signals:
      for i, ch in enumerate(ds.channels_names):
        if channels is not None and ch not in channels:
          continue
        signal_dict[ch] = (ds.data[:, i], ds.sfreq)

    # Build annotations (both staged and event-only)
    annotations = {}
    for key, anno in sg.annotations.items():
      intervals = np.array(anno.intervals, dtype=np.float64)
      if anno.annotations is not None:
        labels = np.array(anno.annotations, dtype=np.int32)
        label_names = list(anno.labels) if anno.labels else []
      else:
        labels, label_names = None, []
      annotations[key] = (intervals, labels, label_names)

    # Build meta from sg.properties (warn on drops)
    meta = {}
    for k, v in sg.properties.items():
      if isinstance(v, (int, float, str, bool, np.integer, np.floating)):
        meta[k] = v
      elif isinstance(v, (list, tuple, np.ndarray)):
        meta[k] = np.asarray(v)
      elif isinstance(v, dict):
        for dk, dv in v.items():
          if isinstance(dv, (int, float, str, bool, np.integer, np.floating)):
            meta[f'{k}.{dk}'] = dv
          else:
            console.show_status(
              f"Skipped non-serializable meta '{k}.{dk}' "
              f"(type: {type(dv).__name__})", prompt='[Warning]')
      else:
        console.show_status(
          f"Skipped non-serializable meta '{k}' "
          f"(type: {type(v).__name__})", prompt='[Warning]')

    return cls.from_raw(path, signal_dict, annotations, meta,
                        label=sg.label or '', dtype=dtype,
                        chunk_seconds=chunk_seconds)

  # endregion: Creation

  # region: Lifecycle

  def close(self):
    if self._file is not None:
      self._file.close()
      self._file = None

  def __enter__(self):
    return self

  def __exit__(self, *args):
    self.close()

  def __del__(self):
    self.close()

  # endregion: Lifecycle
