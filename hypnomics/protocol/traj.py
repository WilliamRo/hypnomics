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
"""Trajectory protocol — HDF5 cache of per-subject feature trajectories.

Trajectories are derived from `.psg.h5` signals by applying probes over
fixed-length epochs. Stored parallel to `_psg/` under `_traj/`, one
`.traj.h5` per subject. See protocol/TRAJ_DESIGN.md for the full spec.

File layout (.traj.h5):
  annotations/              (mirror of .psg.h5 — all scorings)
    <key>/
      intervals             (float64, [N, 2])
      labels                (int32, [N])          # omitted for event-only
      .attrs: {label_names: [...]}                # or {is_event: True}
  traj/
    <channel_name>/
      <time_res>s/          (e.g., '30s', '10s')
        <probe_key>         (dataset, 1-D float, length = n_epochs)
      .attrs: {time_resolution: int}              # on the <time_res>s group
  meta/
    .attrs: {scalar keys}
    <key>                   (dataset for arrays)
  .attrs: {label: str, source_psg: str (optional)}
"""
import h5py
import numpy as np

from roma import console



class Traj:
  """Lazy reader/writer for .traj.h5 files."""

  EXT = '.traj.h5'
  PSG_EPOCH_LEN = 30  # seconds — traj time_resolution must divide this

  def __init__(self, path):
    """Open a .traj.h5 file for lazy reading.

    Args:
      path: path to .traj.h5 file.
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
  def source_psg(self):
    return self.file.attrs.get('source_psg', '')

  @property
  def channels(self):
    if 'traj' not in self.file: return []
    return list(self.file['traj'].keys())

  @property
  def annotation_keys(self):
    if 'annotations' not in self.file: return []
    return list(self.file['annotations'].keys())

  @property
  def meta(self):
    if 'meta' not in self.file: return {}
    grp = self.file['meta']
    result = dict(grp.attrs)
    for k in grp:
      result[k] = grp[k][:]
    return result

  def time_resolutions(self, channel=None):
    """List available time resolutions (in seconds) for a channel.

    Args:
      channel: channel name (None = use first channel).

    Returns:
      List of int time resolutions, e.g., [30, 10, 1].
    """
    chs = self.channels
    if not chs: return []
    if channel is None: channel = chs[0]
    assert channel in chs, (
      f"Channel '{channel}' not found. Available: {chs}")
    ch_grp = self.file['traj'][channel]
    trs = []
    for key in ch_grp.keys():
      # Keys like '30s', '10s' — strip trailing 's'
      if key.endswith('s'):
        try: trs.append(int(key[:-1]))
        except ValueError: continue
    return sorted(trs)

  def probe_keys(self, channel=None, time_resolution=None):
    """List available probe keys.

    Args:
      channel: channel name (None = use first channel).
      time_resolution: int (None = use first available).

    Returns:
      List of probe key strings.
    """
    chs = self.channels
    if not chs: return []
    if channel is None: channel = chs[0]
    assert channel in chs, (
      f"Channel '{channel}' not found. Available: {chs}")
    trs = self.time_resolutions(channel)
    if not trs: return []
    if time_resolution is None: time_resolution = trs[0]
    assert time_resolution in trs, (
      f"time_resolution {time_resolution} not found. Available: {trs}")
    tr_grp = self.file['traj'][channel][f'{time_resolution}s']
    return list(tr_grp.keys())

  # endregion: Properties

  # region: Reading

  def get(self, channel, time_resolution, probe_key):
    """Read a trajectory as a 1-D array.

    Args:
      channel: channel name.
      time_resolution: int, time resolution in seconds.
      probe_key: probe key string.

    Returns:
      np.ndarray of shape [n_epochs], values in epoch order.
    """
    assert 'traj' in self.file, 'No traj group in this file'
    assert channel in self.file['traj'], (
      f"Channel '{channel}' not found. Available: {self.channels}")
    tr_key = f'{time_resolution}s'
    assert tr_key in self.file['traj'][channel], (
      f"time_resolution {time_resolution} not found for channel '{channel}'. "
      f"Available: {self.time_resolutions(channel)}")
    tr_grp = self.file['traj'][channel][tr_key]
    assert probe_key in tr_grp, (
      f"Probe '{probe_key}' not found for ({channel}, {time_resolution}s). "
      f"Available: {list(tr_grp.keys())}")
    return tr_grp[probe_key][:]

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

    if tmin is not None:
      intervals[:, 0] = np.maximum(intervals[:, 0], tmin)
    if tmax is not None:
      intervals[:, 1] = np.minimum(intervals[:, 1], tmax)

    return intervals, labels, label_names

  def get_cloud(self, channel, time_resolution, probe_key,
                annotation_key='stage Ground-Truth'):
    """Generate a stage-stratified cloud on-the-fly from a trajectory.

    Groups trajectory values by sleep stage according to the chosen
    annotation. Requires a stage annotation (not event-only).

    Args:
      channel: channel name.
      time_resolution: int, time resolution in seconds.
      probe_key: probe key string.
      annotation_key: annotation key to use for staging.

    Returns:
      dict: {stage_name: [values], ...} — values are float.
    """
    traj = self.get(channel, time_resolution, probe_key)
    intervals, labels, label_names = self.get_annotation(annotation_key)
    if labels is None:
      raise ValueError(
        f"Annotation '{annotation_key}' is event-only and cannot be used "
        f"for cloud generation.")
    assert len(label_names) > 0, (
      f"Annotation '{annotation_key}' has no label_names.")

    # Assign a stage index to each trajectory epoch via interval coverage.
    # An epoch i spans [i*tr, (i+1)*tr]; find which interval contains it.
    n_epochs = len(traj)
    tr = float(time_resolution)
    stage_idx = np.full(n_epochs, -1, dtype=np.int32)
    for (t1, t2), lbl in zip(intervals, labels):
      i1 = int(round(t1 / tr))
      i2 = int(round(t2 / tr))
      i1 = max(0, i1)
      i2 = min(n_epochs, i2)
      if i1 < i2: stage_idx[i1:i2] = lbl

    # Build the cloud dict
    cloud = {name: [] for name in label_names}
    for i, lbl in enumerate(stage_idx):
      if 0 <= lbl < len(label_names):
        cloud[label_names[lbl]].append(float(traj[i]))
    return cloud

  # endregion: Reading

  # region: Creation

  @classmethod
  def from_raw(cls, path, traj_data, annotations=None, meta=None,
               label='', source_psg='', dtype=np.float32):
    """Create a .traj.h5 file from in-memory trajectory data.

    Args:
      path: output file path.
      traj_data: dict mapping (channel, time_resolution, probe_key) to a
        1-D np.ndarray. Example:
          {('EEG Fpz-Cz', 30, 'AMP-1'): np.array([0.1, 0.2, ...]), ...}
      annotations: {key: (intervals, labels, label_names), ...}
        intervals: np.ndarray or list of (start, end) pairs in seconds.
        labels: np.ndarray or list of int, or None for event-only.
        label_names: list of str.
      meta: dict of metadata {key: value}.
      label: recording label (e.g., patient ID).
      source_psg: optional path/label of the source .psg.h5.
      dtype: storage dtype for trajectory values (default np.float32).

    Returns:
      Traj instance for the created file.
    """
    # Validate traj_data
    for (ch, tr, pk), values in traj_data.items():
      assert isinstance(values, np.ndarray) and values.ndim == 1, (
        f"traj_data[{(ch, tr, pk)}]: expected 1-D ndarray, "
        f"got shape {values.shape}")
      assert '/' not in ch, (
        f"Channel name '{ch}' contains '/' which is invalid for HDF5")
      assert '/' not in pk, (
        f"Probe key '{pk}' contains '/' which is invalid for HDF5")
      assert cls.PSG_EPOCH_LEN % int(tr) == 0, (
        f"time_resolution {tr} must divide {cls.PSG_EPOCH_LEN}")

    with h5py.File(path, 'w') as f:
      f.attrs['label'] = label
      if source_psg:
        f.attrs['source_psg'] = source_psg

      # Trajectories
      traj_grp = f.create_group('traj')
      for (ch, tr, pk), values in traj_data.items():
        # Get or create channel group
        ch_grp = traj_grp.require_group(ch)
        # Get or create time_resolution group
        tr_key = f'{int(tr)}s'
        tr_grp = ch_grp.require_group(tr_key)
        tr_grp.attrs['time_resolution'] = int(tr)
        # Write probe dataset
        assert pk not in tr_grp, (
          f"Duplicate entry for ({ch}, {tr}s, {pk})")
        tr_grp.create_dataset(pk, data=values.astype(dtype))

      # Annotations (mirror of .psg.h5 format)
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

      # Meta (same write conventions as PSG.from_raw)
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
  def from_psg(cls, psg, path, probes, channels=None, time_resolutions=(30,),
               copy_meta=True, copy_annotations=True, dtype=np.float32,
               probe_configs=None, source_psg=None,
               channel_should_exist=True, skip_invalid=False,
               verbose=True, on_progress=None):
    """Extract trajectories from an open PSG and write to .traj.h5.

    For each (channel, time_resolution), the signal is reshaped into
    consecutive non-overlapping epochs of length `time_resolution` seconds,
    and each probe is applied to each epoch, producing a 1-D array of
    feature values in epoch order.

    Two probe interfaces are supported:

    1. **Dict of callables**: `{probe_key: callable(signal_1d, sfreq, **cfg)
       -> float}`. Each probe is called once per epoch; `probe_configs`
       supplies per-probe kwargs.
    2. **ProbeGroup-like** (duck-typed): any object with a `probe_keys`
       attribute (list of strings) and a `_generate_feature_dict(signal_1d)`
       method that returns `{probe_key: value}` for all keys in one call.
       This is efficient when multiple probes share expensive computation
       (e.g., a single PSD drives several band ratios).

    Args:
      psg: open PSG instance (hypnomics.protocol.psg.PSG).
      path: output .traj.h5 file path.
      probes: dict of probes OR a ProbeGroup-like object.
      channels: list of channel names to extract (None = all channels in
        the PSG).
      time_resolutions: iterable of int, each must divide PSG_EPOCH_LEN (30).
      copy_meta: whether to copy PSG meta into the .traj.h5.
      copy_annotations: whether to mirror PSG annotations into the .traj.h5.
        Must be True for get_cloud() to work.
      dtype: storage dtype for trajectory values.
      probe_configs: optional dict-of-dicts `{probe_key: {**kwargs}}` passed
        as extra kwargs to the corresponding probe callable. Ignored for
        ProbeGroup interface (configure the group at construction).
      source_psg: optional path/label of the source PSG; if None, uses
        `psg._path`.
      channel_should_exist: if True (default), requested channels must all
        exist in the PSG or an assertion fires. If False, missing channels
        are warned and silently skipped.
      skip_invalid: if True, any (channel, time_resolution, probe_key)
        whose values contain NaN or Inf is warned and omitted from the
        output file. Default False → invalid values are written as-is.
      verbose: if True (default), print per-(channel, time_resolution)
        status lines showing the probe set and epoch count.
      on_progress: optional callable `on_progress(done, total) -> None`
        invoked after each (channel, time_resolution) unit completes or
        is skipped. `total = len(channels) * len(time_resolutions)` is
        fixed upfront. Used by `generate_trajs` to keep the batch-level
        ETA bar updating smoothly during long single-file runs.

    Returns:
      Traj instance for the created file.
    """
    # (0) Resolve arguments
    if channels is None: channels = psg.channel_names
    assert len(channels) > 0, 'No channels to extract'
    is_probe_group = (
      not isinstance(probes, dict)
      and hasattr(probes, '_generate_feature_dict')
      and hasattr(probes, 'probe_keys'))
    assert isinstance(probes, dict) or is_probe_group, (
      f"`probes` must be a dict of callables or a ProbeGroup-like object "
      f"(with `probe_keys` and `_generate_feature_dict`), "
      f"got {type(probes).__name__}")
    n_probes = len(probes.probe_keys) if is_probe_group else len(probes)
    assert n_probes > 0, 'No probes provided'
    if probe_configs is None: probe_configs = {}

    time_resolutions = tuple(int(tr) for tr in time_resolutions)
    for tr in time_resolutions:
      assert cls.PSG_EPOCH_LEN % tr == 0, (
        f"time_resolution {tr} must divide {cls.PSG_EPOCH_LEN}")

    # (1) Build traj_data by walking channels × time_resolutions × probes
    available_channels = set(psg.channel_names)

    def _store(ch, tr, pk, values):
      if skip_invalid and (np.isnan(values).any() or np.isinf(values).any()):
        n_bad = int(np.isnan(values).sum() + np.isinf(values).sum())
        console.warning(
          f"Invalid values in `{ch}` tr={tr}s `{pk}` "
          f"({n_bad}/{len(values)} bad); skipping this entry.")
        return
      traj_data[(ch, tr, pk)] = values

    pk_list_for_log = (list(probes.probe_keys) if is_probe_group
                       else list(probes.keys()))

    total_units = len(channels) * len(time_resolutions)
    units_done = [0]  # boxed to allow mutation inside _tick closure

    def _tick():
      units_done[0] += 1
      if on_progress is not None:
        try: on_progress(units_done[0], total_units)
        except Exception: pass  # never let a progress callback break writing

    traj_data = {}
    for ch in channels:
      if ch not in available_channels:
        if channel_should_exist:
          raise AssertionError(
            f"Channel `{ch}` not found in PSG. "
            f"Available: {sorted(available_channels)}")
        console.warning(f"Channel `{ch}` not in PSG, skipping.")
        for _ in time_resolutions: _tick()
        continue

      sfreq = psg.sfreq(ch)
      # Read the full signal for this channel (1-D)
      signal = psg.read([ch])[:, 0]
      if verbose:
        console.supplement(
          f"`{ch}`: {len(signal)} samples @ {sfreq:g}Hz "
          f"({len(signal) / sfreq:.1f}s)", level=2)

      for tr in time_resolutions:
        samples_per_epoch = int(round(tr * sfreq))
        assert samples_per_epoch > 0, (
          f"samples_per_epoch={samples_per_epoch} for tr={tr}, sfreq={sfreq}")
        n_epochs = len(signal) // samples_per_epoch
        if n_epochs == 0:
          console.show_status(
            f"Channel '{ch}' tr={tr}s: signal too short "
            f"({len(signal)} samples < {samples_per_epoch}), skipping",
            prompt='[Warning]')
          _tick()
          continue
        trimmed = signal[:n_epochs * samples_per_epoch]
        epochs = trimmed.reshape(n_epochs, samples_per_epoch)

        if verbose:
          console.supplement(
            f"tr={tr}s: {n_epochs} epochs x {len(pk_list_for_log)} probes "
            f"[{', '.join(pk_list_for_log)}]", level=3)

        if is_probe_group:
          # One call per epoch returns all probe values at once
          pk_list = list(probes.probe_keys)
          buffers = {pk: np.empty(n_epochs, dtype=dtype) for pk in pk_list}
          for i in range(n_epochs):
            fd = probes._generate_feature_dict(epochs[i])
            for pk in pk_list: buffers[pk][i] = fd[pk]
          for pk in pk_list: _store(ch, tr, pk, buffers[pk])
        else:
          # Dict of callables — one call per (probe, epoch)
          for pk, probe in probes.items():
            cfg = probe_configs.get(pk, {})
            values = np.empty(n_epochs, dtype=dtype)
            for i in range(n_epochs):
              values[i] = probe(epochs[i], sfreq, **cfg)
            _store(ch, tr, pk, values)

        _tick()

    # (2) Mirror annotations from the source PSG
    annotations = None
    if copy_annotations:
      annotations = {}
      for key in psg.annotation_keys:
        annotations[key] = psg.get_annotation(key)

    # (3) Copy meta
    meta = psg.meta if copy_meta else None

    # (4) Resolve source_psg
    if source_psg is None:
      source_psg = getattr(psg, '_path', '') or ''

    return cls.from_raw(path, traj_data, annotations=annotations, meta=meta,
                        label=psg.label, source_psg=str(source_psg),
                        dtype=dtype)

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
