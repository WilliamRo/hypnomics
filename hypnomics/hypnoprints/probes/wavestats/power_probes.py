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
from mne.time_frequency import psd_array_multitaper

try:
  from scipy.integrate import simps
except:
  from scipy.integrate import simpson as simps

from scipy.signal import welch, periodogram

import numpy as np



_BUFFER = {}  # Don't use, for compatibility only
MT_ADAPTIVE = True

# Be consistent with the default `bandwidth` in multitaper method
WEL_FREQ_RESOLUTION = 0.125


def estimate_power(s: np.ndarray, fs: float, fmin=0.5, fmax=30,
                   band='total', band_ref=None, method='multitaper',
                   **kwargs):
  """Estimate total-/relative-power of a signal.

  Args:
    s: np.ndarray, input signal
    fs: float, sampling frequency
    fmin: float, lower bound of frequency band
    fmax: float, upper bound of frequency band
    band: str, should be in ('total', 'delta', 'theta', 'alpha', 'beta', sigma)
    band_ref: str, same as band, None by default. If provided, return relative
              power, i.e., band/band_ref
    method: str, should be in ('multitaper', 'welch', 'periodogram')

  Returns: float, total-/relative power
  """
  # (1) Calculate PSD and freqs
  if method == 'multitaper':
    psd, freqs = psd_array_multitaper(s, fs, fmin, fmax, adaptive=MT_ADAPTIVE,
                                      normalization='full', verbose=False)
  elif method == 'welch':
    nperseg = int(fs / WEL_FREQ_RESOLUTION)
    noverlap = nperseg // 2
    freqs, psd = welch(s, fs, nperseg=nperseg, noverlap=noverlap)
  elif method == 'periodogram':
    freqs, psd = periodogram(s, fs)
  else:
    raise ValueError(f'Unknown method: `{method}`')

  # (1.1) Convert the unit of psd to mu V^2/Hz
  assert isinstance(psd, np.ndarray)
  psd = psd.ravel() * 1e12

  # (2) Calculate and return result accordingly
  band_dict = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
               'beta': (13, 30), 'sigma': (12, 16), 'total': (fmin, fmax)}
  freq_res = freqs[1] - freqs[0]

  # (2.0) Calculate all band and return a dict
  band = band.lower()
  if band == '*':
    power_dict = {}
    for k, (low, high) in band_dict.items():
      power_dict[k] = simps(psd[(freqs > low) & (freqs <= high)], dx=freq_res)
    return power_dict

  # (2.1) Calculate band power
  assert band in band_dict, f'Unknown band: `{band}`'
  low, high = band_dict[band]
  band_power = simps(psd[(freqs > low) & (freqs <= high)], dx=freq_res)
  if band_ref is None: return band_power

  # (2.2) Calculate relative power
  assert band_ref in band_dict
  low_r, high_r = band_dict[band_ref]
  band_ref_power = simps(psd[(freqs > low_r) & (freqs <= high_r)], dx=freq_res)
  return band_power / (band_ref_power + 1e-6)


# For compatibility
estimate_total_power = estimate_power
estimate_delta_power_rela = lambda s, fs: estimate_power(s, fs, band_ref='delta')
estimate_theta_power_rela = lambda s, fs: estimate_power(s, fs, band_ref='theta')
estimate_alpha_power_rela = lambda s, fs: estimate_power(s, fs, band_ref='alpha')
estimate_beta_power_rela = lambda s, fs: estimate_power(s, fs, band_ref='beta')



class PowerProbes(ProbeGroup):

  probe_keys = ['POWER-30', 'PR-DELTA_TOTAL', 'PR-THETA_TOTAL', 'PR-ALPHA_TOTAL',
                'PR-BETA_TOTAL', 'PR-SIGMA_TOTAL', 'PR-DELTA_THETA',
                'PR-DELTA_ALPHA', 'PR-THETA_ALPHA']
  default_method = 'welch'

  def __init__(self, fs):
    super().__init__()
    self.fs = fs

  def _generate_feature_dict(self, array) -> dict:
    power_dict = estimate_power(array, self.fs, band='*',
                                method=self.default_method)
    feature_dict = OrderedDict()
    feature_dict['POWER-30'] = power_dict['total']

    pr = lambda key1, key2: power_dict[key1] / (power_dict[key2] + 1e-6)
    for pk in self.probe_keys[1:]:
      key1, key2 = pk.split('-')[1].split('_')
      feature_dict[pk] = pr(key1.lower(), key2.lower())

    # Check and return
    assert len(feature_dict) == len(self.probe_keys)
    return feature_dict
