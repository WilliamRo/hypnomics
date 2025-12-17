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
from .wavestats.senate import BandPeakFrequency
from .wavestats.senate import lempel_ziv
from .wavestats.senate import permutation_entropy

from collections import OrderedDict



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
  class_band_peak_frequency = BandPeakFrequency

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
  PROBE_GROUP_26 = 'LRP;BPF;TMI6;HFD;AMP'

  @classmethod
  def get_probe_keys(cls, probe_config, for_generation=False):
    assert isinstance(probe_config, (list, tuple, str))

    probe_keys = []

    # Pillar I: Intra-Band Features
    # - Relative Band Power
    if 'RBP' in probe_config:
      if for_generation: probe_keys.append('power_group')
      else: probe_keys.extend( [
        'PR-DELTA_TOTAL', 'PR-THETA_TOTAL', 'PR-ALPHA_TOTAL',
        'PR-BETA_TOTAL', 'PR-SIGMA_TOTAL'])

    if 'LRP' in probe_config:
      if for_generation: probe_keys.append('power_group_log')
      else:
        pp = PowerProbes(fs=None, log=True)
        probe_keys.extend(pp.probe_keys[1:])

    # - Band Peak Frequency
    if 'BPF' in probe_config:
      if for_generation: probe_keys.append('BPF')
      else:
        bpf = BandPeakFrequency(fs=None)
        probe_keys.extend(bpf.probe_keys)

    # - Band Spectral Centroid
    if 'BSC' in probe_config:
      if for_generation: probe_keys.append('BSC')
      else:
        bsc = BandSpectralCentroid(fs=None)
        probe_keys.extend(bsc.probe_keys)

    # Pillar II: Inter-Band Features
    # - Cross-Frequency Coupling (CFC)
    if 'TMI6' in probe_config:
      if for_generation: probe_keys.append('tort')
      else:
        tmi = PAC_MI(fs=None, method='tort')
        probe_keys.extend(tmi.probe_keys)

    # Pillar III: Overall Signal Features
    if 'SEF' in probe_config: probe_keys.append('SEF-95')
    # - Complexity
    if 'HFD' in probe_config: probe_keys.append('HFD-10')
    if 'LEM' in probe_config: probe_keys.append('LEMPEL_ZIV')
    if 'AMP' in probe_config: probe_keys.append('HAMP')

    return probe_keys


def get_extractor_dict(keys, **kwargs):
  od = OrderedDict()

  for key in keys:
    if '-' in key: name, arg = key.split('-')
    else: name, arg = key, None

    if name == 'AMP':
      _fs = kwargs.get('fs')
      _ws = float(arg)
      od[key] = lambda s, fs=_fs, ws=_ws: ProbeLibrary.amplitude(
        s, fs=fs, window_size=ws)
    elif name == 'HAMP':
      od[key] = ProbeLibrary.amplitude_h
    elif name == 'FREQ':
      _fs = kwargs.get('fs')
      _fmax = float(arg)
      od[key] = lambda s, fs=_fs, fmax=_fmax: ProbeLibrary.frequency_stft(
        s, fs=fs, fmax=fmax)
    elif name == 'GFREQ':
      _fs = kwargs.get('fs')
      _fmax = float(arg)
      od[key] = lambda s, fs=_fs, fmax=_fmax: ProbeLibrary.frequency_st(
        s, fs=fs, fmax=fmax)
    elif name in ('P', 'RP'):
      _fs = kwargs.get('fs')
      od[key] = lambda s, fs=_fs, band=arg: ProbeLibrary.total_power(
        s, fs=fs, band=band)
    elif name == 'MAG':
      od[key] = lambda s: ProbeLibrary.mean_absolute_gradient(s)
    elif name == 'KURT':
      od[key] = lambda s: ProbeLibrary.kurtosis(s)
    elif name == 'ENTROPY':
      od[key] = lambda s: ProbeLibrary.sample_entropy(s)
    elif name == 'RPS':
      _fs = kwargs.get('fs')
      _b1, _b2, _st = arg.split('_')
      _st = {'95': '95th percentile', 'MIN': 'min',
             'AVG': 'mean', 'STD': 'std'}[_st]
      od[key] = lambda s, fs=_fs, b1=_b1, b2=_b2, st=_st: ProbeLibrary.relative_power_stats(
        s, fs=fs, band1=b1, band2=b2, stat_key=st)
    elif name == 'BKURT':
      _fs = kwargs.get('fs')
      od[key] = lambda s, fs=_fs, band=arg: ProbeLibrary.band_kurtosis(
        s, fs=fs, band=band)
    elif name == 'power_group':
      _fs = kwargs.get('fs')
      od[key] = ProbeLibrary.class_power_group(_fs)
    elif name == 'power_group_log':
      _fs = kwargs.get('fs')
      od[key] = ProbeLibrary.class_power_group(_fs, log=True)
    elif name in ('pac_mi_group', 'pac_mi', 'tort'):
      _fs = kwargs.get('fs')
      od[key] = ProbeLibrary.class_pac_mi_group(_fs, method='tort')
    elif name in ('BSC',):
      _fs = kwargs.get('fs')
      od[key] = ProbeLibrary.class_band_spectral_centroid(_fs)
    elif name in ('BPF',):
      _fs = kwargs.get('fs')
      od[key] = ProbeLibrary.class_band_peak_frequency(_fs)
    elif name in ('hfd', 'HFD'):
      if arg is None: _k = 10  # default
      else: _k = int(arg)
      od[key] = lambda s: ProbeLibrary.higuchi_fd(s, k=_k)
    elif name in ('LEM', 'LEMPEL_ZIV'):
      od[key] = lambda s: ProbeLibrary.lempel_ziv(s)
    elif name == 'SEF':
      if arg is None: _ep = 0.95  # default
      else: _ep = float(arg) / 100
      _fs = kwargs.get('fs')
      od[key] = lambda s: ProbeLibrary.spectral_edge_frequency(
        s, fs=_fs, edge_percent=_ep)
    else: raise KeyError(f'Unknown key: {key}')

  return od


pl = ProbeLibrary
