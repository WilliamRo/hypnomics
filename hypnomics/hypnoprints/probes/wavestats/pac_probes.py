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
# ====-==================================================================-======
from collections import OrderedDict
from hypnomics.hypnoprints.probes.probe_group import ProbeGroup

import numpy as np



class PAC_MI(ProbeGroup):

  band_keys = ['delta_low', 'delta_high', 'theta', 'alpha',
               'beta_low', 'beta_high']
  method = 'tort'


  def __init__(self, fs):
    super().__init__()
    self.fs = fs


  @property
  def probe_keys(self) -> list:
    keys = []
    for i in range(len(self.band_keys) - 1):
      for j in range(i + 1, len(self.band_keys)):
        key = f'TMI-{self.band_keys[i].upper()}-{self.band_keys[j].upper()}'
        keys.append(key)
    return keys


  def calc_tort_2010_mi(self, low_freq_s: np.ndarray, high_freq_s: np.ndarray,
                        n_bins=18) -> float:
    from scipy.signal import hilbert

    # (0) Sanity check
    assert self.method == 'tort', 'Only tort is supported now.'
    assert len(low_freq_s) == len(high_freq_s)

    # (1) Calculate phase
    analytic_signal = hilbert(low_freq_s.flatten())
    phase = np.angle(analytic_signal)

    # (2) Calculate amplitude
    analytic_signal = hilbert(high_freq_s.flatten())
    amplitude = np.abs(analytic_signal)

    # (3) Calculate MI
    # (3.1) Calculate bin edges and assign indices (more robust than manual calculation)
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1, endpoint=True)
    # Convert to 0-based index
    bin_indices = np.digitize(phase, bin_edges, right=False) - 1

    # Handle edge case where phase equals π (should go into last bin)
    bin_indices[phase == np.pi] = n_bins - 1

    # (3.2) Calculate bin means using vectorized operations (faster than loops)
    bin_sums = np.bincount(bin_indices, weights=amplitude, minlength=n_bins)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)

    # Avoid division by zero in empty bins
    bin_means = np.divide(bin_sums, bin_counts, out=np.zeros(n_bins),
                          where=bin_counts > 0)

    # Normalize (handle case where total sum is zero to avoid division by zero)
    total = bin_means.sum()
    if total == 0:
      normalized_means = np.zeros_like(bin_means)
    else:
      normalized_means = bin_means / total

    # (3.3) Calculate modulation index (MI = 1 - entropy, where entropy is sum(p_i * log(p_i)))
    # Add epsilon to avoid log(0) issues
    epsilon = np.finfo(float).eps
    entropy = -np.sum(normalized_means * np.log(normalized_means + epsilon))
    modulation_index = 1 - (entropy / np.log(n_bins))  # Normalize entropy by maximum possible

    return modulation_index

    # # (3) Bin amplitude according to phase
    # phase_bins = [[] for _ in range(n_bins)]
    # for p, a in zip(phase, amplitude):
    #   # p in (-pi, pi)
    #   bin_index = int((p + np.pi) // (2 * np.pi / n_bins))
    #   phase_bins[bin_index].append(a)
    #
    # # (4) Normalize bin means
    # bin_means = [np.mean(phase_bins[i]) if len(phase_bins[i]) > 0 else 0
    #              for i in range(n_bins)]
    # bin_means = bin_means / np.sum(bin_means)
    #
    # # (5) Calculate MI as MI = D_KL(P, U) / log(N)
    # P = np.array(bin_means)
    # U = np.array([1 / n_bins] * n_bins)
    # mi_value = entropy(P + 1e-10, U + 1e-10) / np.log(n_bins)
    #
    # return mi_value


  def generate_clouds(self, segments, channel_index, sg=None, tr=None) -> dict:
    """Returns a dictionary of clouds for each probe key.

    Args:
      segments: a dictionary of lists of multichannel physiological data
        e.g., {'W': [array_1, ...], 'N1': [...], ...},
              where array_1.shape = (L, n_channels)
      channel_index: int, index of the channel to be processed
      sg: SignalGroup, optional, the signal group object

    Returns:
      clouds_dict: OrderedDict,
        e.g., {probe_key: {stage_key: [value_1, value_2, ...], ...}, ...}
    """
    from hypnomics.hypnoprints.hp_extractor import STAGE_KEYS
    from hypnomics.hypnoprints.hp_extractor import DEFAULT_STAGE_KEY
    from hypnomics.hypnoprints.hp_extractor import get_sg_stage_epoch_dict
    from pictor.objects.signals.eeg import SignalGroup, EEG

    # (0) Sanity check
    assert len(self.probe_keys) > 0
    assert isinstance(sg, SignalGroup)

    # (1) Initialization
    clouds_dict = OrderedDict()
    for pk in self.probe_keys:
      clouds_dict[pk] = OrderedDict()
      for sk in STAGE_KEYS: clouds_dict[pk][sk] = []

    eeg: EEG = sg.as_eeg()
    se_dict = OrderedDict()
    for band in self.band_keys:
      band_sg: SignalGroup = eeg.get_band_as_sg(band, channel_index)
      se_dict[band] = get_sg_stage_epoch_dict(band_sg, DEFAULT_STAGE_KEY, tr)

    # (2) Generate clouds for each stage
    for sk in STAGE_KEYS:
      for i in range(len(self.band_keys) - 1):
        phase_key = self.band_keys[i]
        for j in range(i + 1, len(self.band_keys)):
          amp_key = self.band_keys[j]

          key = f'TMI-{phase_key.upper()}-{amp_key.upper()}'
          for low_s, high_s in zip(se_dict[phase_key][sk], se_dict[amp_key][sk]):
            mi = self.calc_tort_2010_mi(low_s, high_s)
            clouds_dict[key][sk].append(mi)

    # (-1) Return
    return clouds_dict
