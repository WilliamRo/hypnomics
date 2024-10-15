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
import numpy as np



_BUFFER = {}

def estimate_power(s: np.ndarray, fs: float, fmin=0.5, fmax=30,
                   band='total', use_buffer=True) -> float:
  """Estimate total-/relative-power of a signal.

  Args:
    s: np.ndarray, input signal
    fs: float, sampling frequency
    fmin: float, lower bound of frequency band
    fmax: float, upper bound of frequency band
    band: str, should be in ('total', 'delta', 'theta', 'alpha', 'beta')

  Returns: float, total-/relative power
  """
  from mne.time_frequency import psd_array_multitaper
  from scipy.integrate import simps

  psd, freqs = None, None
  array_key = str(s)
  key = (array_key, fmin, fmax, fs)
  if use_buffer and key in _BUFFER: psd, freqs = _BUFFER[key]
  if psd is None:
    psd, freqs = psd_array_multitaper(s, fs, fmin, fmax,
                                      adaptive=True, n_jobs=5,
                                      normalization='full', verbose=False)
    if use_buffer: _BUFFER[key] = (psd, freqs)

  # Convert the unit of psd to mu V^2/Hz
  psd = psd.ravel() * 1e12

  # Calculate total power
  freq_res = freqs[1] - freqs[0]
  total_power = simps(psd, dx=freq_res)

  # Calculate and return result accordingly
  band = band.lower()
  if band == 'total': return total_power

  band_dict = {'delta': (0.5, 4), 'theta': (4, 8),
               'alpha': (8, 12), 'beta': (12, 30), 'sigma': (12, 16)}
  assert band in band_dict
  low, high = band_dict[band]
  band_power = simps(psd[(freqs > low) & (freqs <= high)], dx=freq_res)
  return band_power / (total_power + 1e-6)


estimate_total_power = estimate_power
estimate_delta_power_rela = lambda s, fs: estimate_power(s, fs, band='delta')
estimate_theta_power_rela = lambda s, fs: estimate_power(s, fs, band='theta')
estimate_alpha_power_rela = lambda s, fs: estimate_power(s, fs, band='alpha')
estimate_beta_power_rela = lambda s, fs: estimate_power(s, fs, band='beta')
