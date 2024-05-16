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
# ====-========================================================================-
from collections import OrderedDict
from hypnomics.hypnoprints.probes import pl
from pictor.objects.signals.digital_signal import DigitalSignal
from pictor.objects.signals.signal_group import Annotation
from pictor.objects.signals.signal_group import SignalGroup

import numpy as np



STAGE_KEYS = ('W', 'N1', 'N2', 'N3', 'R')


def extract_hypnocloud_from_signal_group(
    sg: SignalGroup, channels, time_resolution=30,
    stage_key='stage Ground-Truth', probe_configs={}) -> dict:
  """Extract hypno-cloud from signal group using given probes.
  Support R&K, AASM stage annotations. If R&K annotations are provided,
  N3 and N4 stages will be merged into N3.

  Args:
    sg: SignalGroup, should contain only one DigitalSignal
    channels: str or list of str
    time_resolution: int, time resolution in seconds, should be a factor of 30
    stage_key: str, key of the stage annotation

  Returns: dict, hypnocloud[channel_key][probe_key][stage_key] = [v1, ..., vN]
  """
  # I Sanity check
  # I-i Check input
  assert isinstance(sg, SignalGroup)
  if isinstance(channels, str): channels = [channels]
  ds: DigitalSignal = sg.digital_signals[0]
  for chn in channels:
    assert chn in ds.channels_names
  # I-ii Check annotations
  if stage_key not in sg.annotations: raise ValueError(
    f"!! `{stage_key}` not found in {sg.label}'s annotations !!")
  # I-iii Check time resolution
  if 30 % time_resolution != 0:
    raise NotImplementedError("!! Time resolution should be a factor of 30 !!")

  # II Extract stage clusters
  channels_names, fs = ds.channels_names, ds.sfreq
  # segments = {'W': list_0, 'N1': list_1, ..., 'R': list_4}
  #   list_i[j].shape = [L_ij, C], i = 0, 1, ..., 4, j = 0, 1, ..., N_i
  segments = get_sg_stage_epoch_dict(sg, stage_key, time_resolution)

  chn_stag_prob_dict = {chn: {} for chn in channels}
  for chn in channels:
    chn_index = channels_names.index(chn)
    for stag_key in STAGE_KEYS:
      prob_dict = {}
      for k, f in pl.extractors.items(): prob_dict[k] = np.array(
        [f(s[:, chn_index], fs, **probe_configs.get(k, {}))
         for s in segments[stag_key]])
      chn_stag_prob_dict[chn][stag_key] = prob_dict

  return chn_stag_prob_dict


def extract_hypnoprints_from_hypnocloud(
    hypnocloud: dict, config='default', return_dict=False) -> np.ndarray:
  """Extract hypnoprints from hypnocloud based on config.
  """
  if config != 'default': raise NotImplementedError(
    "!! Only default config is supported for now !!")

  weights = {'W': 0.2, 'N1': 0.5, 'N3': 1, 'R': 1}

  x_dict = OrderedDict()

  # (1) Proportion
  x_dict.update(hypno_proportion(hypnocloud))

  # (2) Shape
  x_dict.update(hypno_shape_1(hypnocloud))

  if return_dict: return x_dict
  return np.array(list(x_dict.values()))

# region: Feature Extractors

def hypno_proportion(cloud: dict):
  x_dict = OrderedDict()

  ordered_chn_key = sorted(cloud.keys())
  ordered_prob_key = sorted(cloud[ordered_chn_key[0]]['W'].keys())
  ck0, pk0 = ordered_chn_key[0], ordered_prob_key[0]

  N = sum([len(cloud[ck0][sk][pk0]) for sk in STAGE_KEYS if sk != 'W'])
  for sk in STAGE_KEYS:
    x_dict[f'<{sk}>/M'] = len(cloud[ck0][sk][pk0]) / N

  return x_dict


def hypno_shape_1(cloud: dict):
  x_dict = OrderedDict()

  ordered_chn_key = sorted(cloud.keys())
  ordered_prob_key = sorted(cloud[ordered_chn_key[0]]['W'].keys())
  ck0, pk0 = ordered_chn_key[0], ordered_prob_key[0]

  for ck in ordered_chn_key:
    # Merge data
    data = {sk: np.vstack([cloud[ck][sk][pk] for pk in ordered_prob_key])
            for sk in STAGE_KEYS if len(cloud[ck][sk][pk0]) > 0}
    N2_mu = np.mean(data['N2'], axis=1)

    for sk in STAGE_KEYS:
      if sk == 'N2': continue
      mu = np.mean(data[sk], axis=1)
      coord = (mu - N2_mu)
      for pk, x in zip(ordered_prob_key, coord):
        xk = f'<{ck.split(" ")[-1]}|{sk}|{pk[0]}>coord'
        x_dict[xk] = x

  return x_dict

# endregion: Feature Extractors


# region: Utilities

def get_sg_stage_epoch_dict(sg: SignalGroup, stage_key, time_resolution=30):
  """Get stage epoch dict from signal group.
  Should be called by confident users only."""
  STAGE_EPOCH_KEY = f'HYPNOMICS_KEY_SEGMENTS_{stage_key}_{time_resolution}'

  assert 30 % time_resolution == 0

  def _init_sg_stage_epoch_dict():
    ds = sg.digital_signals[0]
    T = int(ds.sfreq * time_resolution)
    # Get annotation
    anno: Annotation = sg.annotations[stage_key]
    # Get reshaped tape
    E = ds.data.shape[0] // T
    # Remove the last epoch if it's incomplete
    tape = ds.data[:E * T]
    tape = tape.reshape([E, T, ds.data.shape[-1]])
    # Generate map_dict
    map_dict = get_stage_map_dict(sg, stage_key)

    se_dict, cursor = {k: [] for k in STAGE_KEYS}, 0
    for interval, anno_id in zip(anno.intervals, anno.annotations):
      n = int((interval[-1] - interval[0]) / time_resolution)
      sid = map_dict[anno_id]
      if sid is not None:
        skey = STAGE_KEYS[map_dict[anno_id]]
        for i in range(cursor, cursor + n):
          # Sometimes annotation interval is longer than the signal
          if i < len(tape): se_dict[skey].append(tape[i])
      cursor += n

    return se_dict

  return sg.get_from_pocket(
    STAGE_EPOCH_KEY, initializer=_init_sg_stage_epoch_dict)


def get_stage_map_dict(sg: SignalGroup, stage_key):
  KEY_MAP_DICT = 'HYPNOMICS_KEY_STAGE_MAP_DICT'

  anno: Annotation = sg.annotations[stage_key]

  def _init_map_dict(labels):
    map_dict = {}
    for i, label in enumerate(labels):
      if 'W' in label: j = 0
      elif '1' in label: j = 1
      elif '2' in label: j = 2
      elif '3' in label or '4' in label: j = 3
      elif 'R' in label: j = 4
      else: j = None
      map_dict[i] = j
    return map_dict

  return sg.get_from_pocket(
    KEY_MAP_DICT, initializer=lambda: _init_map_dict(anno.labels))

# endregion: Utilities
