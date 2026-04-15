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
"""Batch operations for .psg.h5 → .traj.h5 conversion.

Provides `generate_trajs`, a reusable helper mirroring the feature set of
`Freud.generate_clouds` but targeting the per-subject `.traj.h5` layout:

  - Source as an explicit `psg_file_list` OR a `psg_dir + pattern` walk.
  - Smart skip: filters out files where the existing `.traj.h5` already
    contains every requested (channel, time_resolution, probe_key).
  - Dry run via `return_psg_file_list=True` (returns the pending list and
    the total count, no processing).
  - `channel_should_exist=False` to tolerate missing channels.
  - `skip_invalid=True` to warn and drop NaN/Inf probe values.
  - `gc.collect()` + best-effort `_BUFFER.clear()` (power_probes cache)
    after each file for large batches.
  - Per-file ETA via `roma.console.print_progress`.
"""
from hypnomics.protocol import PSG, Traj
from roma import console, finder

import gc
import os
import time
import numpy as np



def generate_trajs(traj_dir, probes,
                   psg_file_list=None, psg_dir=None, pattern='*.psg.h5',
                   channels=None, time_resolutions=(30,), overwrite=False,
                   dtype=np.float32, copy_meta=True, copy_annotations=True,
                   channel_should_exist=True, skip_invalid=False,
                   return_psg_file_list=False, verbose=True):
  """Batch convert `.psg.h5` files to `.traj.h5` with ETA reporting.

  Files that already contain every requested (channel, time_resolution,
  probe_key) combination are filtered out BEFORE the main loop (unless
  `overwrite=True`), so the ETA reflects only real work.

  Source selection (exactly one):
    - `psg_file_list`: explicit list of paths (takes priority if both given).
    - `psg_dir` + `pattern`: walked via `roma.finder.walk`.

  Args:
    traj_dir: output directory; created if missing.
    probes: one of —
      (a) a dict `{probe_key: callable(signal_1d, sfreq, **cfg) -> float}`,
      (b) a ProbeGroup-like instance with `probe_keys` and
          `_generate_feature_dict(signal_1d) -> {pk: value}`,
      (c) a callable factory `probes(psg) -> dict | ProbeGroup`, invoked
          once per file (useful when probes need per-file `fs`).
    psg_file_list: explicit list of `.psg.h5` paths. Mutually exclusive
      with `psg_dir`.
    psg_dir: directory to walk for `.psg.h5` files.
    pattern: glob for the directory walk (default `'*.psg.h5'`).
    channels: one of —
      (a) None → all channels in each PSG,
      (b) list of channel names (static),
      (c) callable `channels(psg) -> list` (per-file filter,
          e.g., EEG-only).
    time_resolutions: iterable of int seconds; each must divide 30.
    overwrite: if False, skip files where every requested combo already
      exists in the `.traj.h5`.
    dtype: storage dtype for trajectory values.
    copy_meta: forwarded to `Traj.from_psg`.
    copy_annotations: forwarded to `Traj.from_psg`. Must be True for
      downstream `get_cloud` to work.
    channel_should_exist: if True (default), requested channels must exist
      in each PSG or the file fails. If False, missing channels are warned
      and skipped (the file is still processed for whatever channels DO
      exist).
    skip_invalid: if True, any (channel, time_resolution, probe_key) whose
      computed values contain NaN or Inf is warned and omitted from the
      output. Default False → invalid values are written as-is. NOTE:
      matches `freud.generate_clouds` behavior — a probe that always
      produces NaN will be retried on every subsequent run (the smart
      filter sees the resulting .traj.h5 as incomplete). If that happens
      for deterministic reasons, fix the probe.
    return_psg_file_list: if True, skip the main loop and return
      `(pending_paths, n_all)` — a dry run for large-scale job planning.
    verbose: if True (default), print per-(channel, time_resolution)
      status lines inside each file. Set to False for quiet runs.

  Returns:
    - If `return_psg_file_list=True`: `(list_of_pending_psg_paths, n_all)`.
    - Otherwise: dict with counts
      `{'total', 'converted', 'skipped', 'failed'}`.
  """
  # (0) Resolve source
  if psg_file_list is None:
    if psg_dir is None:
      raise ValueError(
        '`psg_file_list` or `psg_dir` must be provided.')
    psg_file_list = finder.walk(psg_dir, pattern=pattern)
    console.show_status(
      f'Walked `{psg_dir}` with pattern `{pattern}`: '
      f'{len(psg_file_list)} file(s) found.')
  elif psg_dir is not None:
    console.warning(
      'Both `psg_file_list` and `psg_dir` provided; using `psg_file_list`.')

  os.makedirs(traj_dir, exist_ok=True)
  n_all = len(psg_file_list)
  tr_list = tuple(int(tr) for tr in time_resolutions)

  # (1) Determine expected probe_keys up-front (assumes uniform across files).
  expected_pks = _expected_probe_keys(probes, psg_file_list)

  # (2) Filter to pending files
  pending, pending_paths = [], []
  n_new, n_incomplete = 0, 0
  for psg_path in psg_file_list:
    pid = os.path.basename(psg_path).replace(PSG.EXT, '')
    traj_path = os.path.join(traj_dir, f'{pid}{Traj.EXT}')

    if overwrite:
      pending.append((psg_path, traj_path, pid))
      pending_paths.append(psg_path)
      continue

    if not os.path.exists(traj_path):
      pending.append((psg_path, traj_path, pid))
      pending_paths.append(psg_path)
      n_new += 1
      continue

    # File exists — resolve expected channels for this file and check coverage
    try:
      if channels is None or callable(channels):
        with PSG(psg_path) as p: ch_for_file = _resolve_channels(channels, p)
      else:
        ch_for_file = list(channels)
    except Exception:
      pending.append((psg_path, traj_path, pid))
      pending_paths.append(psg_path)
      continue

    if _traj_is_complete(traj_path, ch_for_file, tr_list, expected_pks):
      continue
    pending.append((psg_path, traj_path, pid))
    pending_paths.append(psg_path)
    n_incomplete += 1

  n_pending = len(pending)
  n_skip = n_all - n_pending

  reason_bits = []
  if n_new > 0: reason_bits.append(f'{n_new} new')
  if n_incomplete > 0: reason_bits.append(f'{n_incomplete} incomplete')
  if overwrite: reason_bits.append('overwrite=True')
  reason_str = f' ({", ".join(reason_bits)})' if reason_bits else ''

  console.show_status(
    f'Batch traj gen: {n_all} .psg.h5 file(s), '
    f'{n_pending} to process{reason_str}, {n_skip} already complete.')
  console.show_status(f'Output directory: `{traj_dir}`')
  console.show_status(f'Time resolutions: {tr_list}s')
  console.show_status(f'Expected probes: {list(expected_pks)}')

  # (3) Dry run?
  if return_psg_file_list:
    return pending_paths, n_all

  if n_pending == 0:
    return {'total': n_all, 'converted': 0, 'skipped': n_skip, 'failed': 0}

  # (4) Process pending files. ETA is baked into the per-file status line
  #     because an in-place progress bar gets shredded by the verbose
  #     supplement() lines — `\r`-overwrite can't coexist with streaming
  #     text. Inline ETA is always visible.
  n_failed = 0
  batch_start = time.time()

  for i, (psg_path, traj_path, pid) in enumerate(pending):
    elapsed = time.time() - batch_start
    if i == 0:
      eta_suffix = ''  # no history yet
    else:
      avg_per_file = elapsed / i
      remaining = avg_per_file * (n_pending - i)
      eta_suffix = (f' (elapsed {_fmt_duration(elapsed)}, '
                    f'ETA {_fmt_duration(remaining)})')
    console.show_status(
      f'[{i + 1}/{n_pending}] Converting `{pid}` ...{eta_suffix}')

    try:
      with PSG(psg_path) as psg:
        chs = _resolve_channels(channels, psg)
        if len(chs) == 0:
          console.warning(f'  No channels for `{pid}`, skipping.')
          n_failed += 1
          continue

        probes_obj = _resolve_probes(probes, psg)

        Traj.from_psg(psg, traj_path, probes_obj,
                      channels=chs,
                      time_resolutions=time_resolutions,
                      copy_meta=copy_meta,
                      copy_annotations=copy_annotations,
                      dtype=dtype,
                      channel_should_exist=channel_should_exist,
                      skip_invalid=skip_invalid,
                      verbose=verbose)
    except Exception as e:
      console.warning(f'  Failed on `{pid}`: {type(e).__name__}: {e}')
      n_failed += 1
      if os.path.exists(traj_path):
        try: os.remove(traj_path)
        except OSError: pass

    # Memory hygiene between files: clear the power_probes PSD cache
    # if it's in use, then force a gc cycle.
    _clear_probe_buffers()
    gc.collect()

  total_elapsed = time.time() - batch_start

  n_converted = n_pending - n_failed
  console.show_status(
    f'Done in {_fmt_duration(total_elapsed)}: '
    f'{n_converted} converted, {n_skip} skipped, {n_failed} failed '
    f'(total {n_all}).')
  return {'total': n_all, 'converted': n_converted,
          'skipped': n_skip, 'failed': n_failed}



# region: Helpers

def _expected_probe_keys(probes, psg_file_list):
  """Determine the set of probe keys the user expects, without running
  probes. Assumes probes are uniform across files.

  For a dict or ProbeGroup-like object, `probe_keys` is directly available.
  For a callable factory, we instantiate it once with the first PSG just
  to read its `probe_keys` attribute; the PSG is closed immediately.
  """
  if isinstance(probes, dict):
    return tuple(probes.keys())
  if hasattr(probes, 'probe_keys'):
    return tuple(probes.probe_keys)
  if callable(probes):
    if not psg_file_list:
      raise ValueError(
        'Cannot determine expected probe_keys: `probes` is a factory but '
        '`psg_file_list` is empty.')
    with PSG(psg_file_list[0]) as p:
      resolved = probes(p)
      if hasattr(resolved, 'probe_keys'):
        return tuple(resolved.probe_keys)
      if isinstance(resolved, dict):
        return tuple(resolved.keys())
      raise TypeError(
        f'Probes factory returned an unexpected type: '
        f'{type(resolved).__name__}')
  raise TypeError(
    f'`probes` must be dict, ProbeGroup-like, or callable factory; '
    f'got {type(probes).__name__}')


def _traj_is_complete(traj_path, channels, time_resolutions, probe_keys):
  """Return True if `traj_path` contains every requested
  (channel, time_resolution, probe_key) combination.

  Any exception during inspection → treat as incomplete (force regen).
  """
  try:
    t = Traj(traj_path)
    try:
      existing_channels = set(t.channels)
      for ch in channels:
        if ch not in existing_channels: return False
        existing_trs = set(t.time_resolutions(ch))
        for tr in time_resolutions:
          if int(tr) not in existing_trs: return False
          existing_pks = set(t.probe_keys(channel=ch, time_resolution=tr))
          for pk in probe_keys:
            if pk not in existing_pks: return False
      return True
    finally:
      t.close()
  except Exception:
    return False


def _resolve_channels(channels, psg):
  """Resolve `channels` argument to a concrete list for the given PSG."""
  if channels is None: return psg.channel_names
  if callable(channels): return list(channels(psg))
  return list(channels)


def _resolve_probes(probes, psg):
  """Resolve `probes` argument to a concrete probe object.

  Order of precedence (first match wins):
    1. `dict` → return as-is
    2. ProbeGroup-like (has `_generate_feature_dict` and `probe_keys`)
       → return as-is
    3. `callable` → invoke with `psg` to get per-file probes
    4. else → TypeError
  """
  if probes is None:
    raise ValueError('`probes` must not be None')
  if isinstance(probes, dict):
    return probes
  if (hasattr(probes, '_generate_feature_dict')
      and hasattr(probes, 'probe_keys')):
    return probes
  if callable(probes):
    return probes(psg)
  raise TypeError(
    f'`probes` must be a dict, ProbeGroup-like object, or callable '
    f'factory; got {type(probes).__name__}')


def _fmt_duration(seconds):
  """Format a duration in seconds as a compact string.
    45      -> '45s'
    125     -> '2m05s'
    3725    -> '1h02m'
    90000   -> '25h00m'
  """
  s = max(0, int(round(seconds)))
  if s < 60: return f'{s}s'
  if s < 3600:
    m, ss = divmod(s, 60)
    return f'{m}m{ss:02d}s'
  h, rem = divmod(s, 3600)
  m = rem // 60
  return f'{h}h{m:02d}m'


def _clear_probe_buffers():
  """Best-effort clearing of the `power_probes._BUFFER` cache between
  files. Silently no-op if the module isn't imported."""
  try:
    from hypnomics.hypnoprints.probes.wavestats.power_probes import _BUFFER
    _BUFFER.clear()
  except Exception:
    pass

# endregion: Helpers
