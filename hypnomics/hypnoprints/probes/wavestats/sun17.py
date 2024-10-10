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
# ===-=========================================================================-
import numpy as np
import scipy.stats as stats



# region: Time Domain

def mean_absolute_gradient(s: np.ndarray, *_) -> float:
  return np.mean(np.abs(s[1:] - s[:-1]))


def kurtosis(s: np.ndarray, *_) -> float:
  return stats.kurtosis(s)


def sample_entropy(s: np.ndarray, *_) -> float:
  import antropy as ant
  return ant.sample_entropy(s)

# endregion: Time Domain

# region: Frequency Domain

_BUFFER = {}
BAND_DICT = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12),
             'beta': (12, 30), 'sigma': (12, 20), 'total': (0.5, 20)}

def _estimate_power(s: np.ndarray, fs: float, band: str):
  """Calculate power list of each 2-s sub-epoch for a given band.
  """
  from mne.time_frequency import psd_array_multitaper
  from scipy.integrate import simps

  band = band.lower()
  assert band in BAND_DICT

  # (0) Segment each epoch into 2-second subepochs
  fs = int(fs)
  sub_epochs = [s[i:i+fs*2] for i in range(len(s) // fs - 1)]

  # (1) Calculate psd for each subepoch
  array_key = str(s)
  key = ('psd_list', array_key, fs)

  # (1.1) Use buffer if available
  if key in _BUFFER:
    psd_list, freqs = _BUFFER[key]
  else:
    psd_list = []
    for se in sub_epochs:
      psd, freqs = psd_array_multitaper(
        se, fs, 0.5, 20, adaptive=True, normalization='full',
        verbose=False)

      # Convert the unit of psd to mu V^2/Hz
      psd_list.append(psd.ravel() * 1e12)

    # Put results into buffer
    _BUFFER[key] = (psd_list, freqs)

  # (2) Calculate power list of a given band
  key = (f'band_power_{band}', array_key)
  if key in _BUFFER: return _BUFFER[key]

  freq_res = freqs[1] - freqs[0]
  low, high = BAND_DICT[band]
  power_list = [simps(psd[(freqs > low) & (freqs <= high)], dx=freq_res)
                for psd in psd_list]
  power_array = np.array(power_list)
  _BUFFER[key] = power_array
  return power_array


STAT_DICT = {
  '95th percentile': lambda x: np.percentile(x, 95),
  'min': np.min,
  'mean': np.mean,
  'std': np.std,
}

def relative_power_stats(s: np.ndarray, fs: float, band1, band2, stat_key) -> float:
  """Calculate the statistics of relative power between two bands.
  """
  b1 = _estimate_power(s, fs, band1)
  b2 = _estimate_power(s, fs, band2)
  rel_power = b1 / (b2 + 1e-6)
  return STAT_DICT[stat_key](rel_power)


def band_kurtosis(s: np.ndarray, fs: float, band: str) -> float:
  b = _estimate_power(s, fs, band)
  return stats.kurtosis(b)

# endregion: Frequency Domain

