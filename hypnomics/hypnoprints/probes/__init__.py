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
# ======-=========================================================-=============
from .wavestats.amp_probes import estimate_amp_envelope_v0
from .wavestats.amp_probes import estimate_amp_envelope_v1
from .wavestats.freq_probes import estimate_average_freq
from .wavestats.freq_probes import estimate_freq_stft_v1
from .wavestats.fractal import estimate_higuchi_fd

from .wavestats.pac_probes import PAC_MI

from .wavestats.power_probes import estimate_total_power
from .wavestats.power_probes import estimate_delta_power_rela
from .wavestats.power_probes import estimate_theta_power_rela
from .wavestats.power_probes import estimate_alpha_power_rela
from .wavestats.power_probes import estimate_beta_power_rela
from .wavestats.power_probes import PowerProbes

from .wavestats.sun17 import mean_absolute_gradient
from .wavestats.sun17 import kurtosis
from .wavestats.sun17 import sample_entropy
from .wavestats.sun17 import relative_power_stats
from .wavestats.sun17 import band_kurtosis

from .wavestats.senate import spectral_edge_frequency
from .wavestats.senate import BandSpectralCentroid
from .wavestats.senate import lempel_ziv
from .wavestats.senate import permutation_entropy



class ProbeLibrary(object):
  amplitude_h = estimate_amp_envelope_v0
  amplitude = estimate_amp_envelope_v1
  frequency_st = estimate_average_freq
  frequency_stft = estimate_freq_stft_v1
  higuchi_fd = estimate_higuchi_fd
  total_power = estimate_total_power
  relative_power_delta = estimate_delta_power_rela
  relative_power_theta = estimate_theta_power_rela
  relative_power_alpha = estimate_alpha_power_rela
  relative_power_beta = estimate_beta_power_rela

  class_power_group = PowerProbes
  class_pac_mi_group = PAC_MI
  class_band_spectral_centroid = BandSpectralCentroid

  mean_absolute_gradient = mean_absolute_gradient
  kurtosis = kurtosis
  sample_entropy = sample_entropy
  relative_power_stats = relative_power_stats
  band_kurtosis = band_kurtosis

  spectral_edge_frequency = spectral_edge_frequency
  lempel_ziv = lempel_ziv
  permutation_entropy = permutation_entropy


  extractors = {
    'amplitude': amplitude,
    'amplitude_h': amplitude_h,
    'frequency_ft': frequency_st,
    'frequency_stft': frequency_stft,
    'higuchi_fd': higuchi_fd,
    'total_power': total_power,
    'relative_power_delta': relative_power_delta,
    'relative_power_theta': relative_power_theta,
    'relative_power_alpha': relative_power_alpha,
    'relative_power_beta': relative_power_beta,
    'mean_absolute_gradient': mean_absolute_gradient,
    'kurtosis': kurtosis,
    'sample_entropy': sample_entropy,
    'relative_power_stats': relative_power_stats,
    'band_kurtosis': band_kurtosis,
    'spectral_edge_frequency': spectral_edge_frequency,
    'lempel_ziv': lempel_ziv,
    'permutation_entropy': permutation_entropy,
  }

  PROBE_GROUP_25 = 'RBP;BSC;TMI6;SEF;HFD;LEM;AMP'

  @classmethod
  def get_probe_keys(cls, probe_config):
    assert isinstance(probe_config, (list, tuple, str))

    probe_keys = []

    # Pillar I: Intra-Band Features
    # - Relative Band Power
    if 'RBP' in probe_config: probe_keys.extend(
      ['PR-DELTA_TOTAL', 'PR-THETA_TOTAL', 'PR-ALPHA_TOTAL',
       'PR-BETA_TOTAL', 'PR-SIGMA_TOTAL'])

    # - Band Spectral Centroid
    if 'BSC' in probe_config:
      bsc = BandSpectralCentroid(fs=None)
      probe_keys.extend(bsc.probe_keys)

    # Pillar II: Inter-Band Features
    # - Cross-Frequency Coupling (CFC)
    if 'TMI6' in probe_config:
      tmi = PAC_MI(fs=None, method='tort')
      probe_keys.extend(tmi.probe_keys)

    # Pillar III: Overall Signal Features
    if 'SEF' in probe_config: probe_keys.append('SEF-95')
    # - Complexity
    if 'HFD' in probe_config: probe_keys.append('HFD-10')
    if 'LEM' in probe_config: probe_keys.append('LEMPEL_ZIV')
    if 'AMP' in probe_config: probe_keys.append('HAMP')

    return probe_keys


pl = ProbeLibrary
