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
from collections import OrderedDict
from hypnomics.hypnoprints.probes.probe_group import ProbeGroup
from scipy.signal import welch, periodogram

import numpy as np



def spectral_edge_frequency(signal: np.ndarray, fs: float,
                            edge_percent: float = 0.95) -> float:
  """Calculate the Spectral Edge Frequency (SEF) of a signal.

  Args:
    signal (np.ndarray): Input signal array.
    fs (float): Sampling frequency of the signal.
    edge_percent (float): The percentage of total power to consider for SEF.

  Returns:
    float: The spectral edge frequency in Hz.
  """
  from scipy.signal import welch

  # (1) Compute Power Spectral Density (PSD)
  f, Pxx = welch(signal.flatten(), fs=fs, nperseg=fs * 2)

  # (2) Compute cumulative power
  cumulative_power = np.cumsum(Pxx)
  total_power = cumulative_power[-1]

  # (3) Find the frequency where cumulative power reaches the edge_percent
  target_power = edge_percent * total_power
  sef_index = np.where(cumulative_power >= target_power)[0][0]
  sef = f[sef_index]

  return sef



def permutation_entropy(signal: np.ndarray) -> float:
  import antropy as ant
  return ant.perm_entropy(signal, normalize=True)



def lempel_ziv(signal: np.ndarray) -> float:
  import antropy as ant
  x_bin = signal > np.median(signal)
  return ant.lziv_complexity(x_bin, normalize=True)



class BandSpectralCentroid(ProbeGroup):

  band_keys = ['theta', 'alpha', 'sigma']
  default_method = 'welch'

  def __init__(self, fs):
    super().__init__()
    self.fs = fs


  @property
  def probe_keys(self) -> list:
    return [self.bk2pk(bk) for bk in self.band_keys]

  @classmethod
  def bk2pk(cls, band_key: str) -> str:
    return f'SC_{band_key.upper()}'


  def _generate_feature_dict(self, x, nperseg=None) -> dict:
    """Calculates the Spectral Centroid (Center of Mass) for specified bands.

    Parameters:
    - x: 1D numpy array (the EEG epoch)
    - nperseg: Length of Welch segment (defaults to len(x) for smooth spectrum)

    Returns:
    - Dictionary with probe_keys
    """
    # 1. Calculate PSD using Welch's method
    #    nperseg=len(x) gives the highest frequency resolution for the epoch
    if nperseg is None: nperseg = len(x)

    assert self.default_method == 'welch'
    freqs, psd = welch(x, fs=self.fs, nperseg=nperseg)

    # Define bands of interest
    bands = {
      'theta': (4, 8),
      'alpha': (8, 12),
      'sigma': (12, 16)
    }

    features = {}

    for bk in self.band_keys:
      low, high = bands[bk]
      pk = self.bk2pk(bk)

      # 2. Find indices for this band
      #    We use >= low and < high to avoid overlap issues
      idx = np.where((freqs >= low) & (freqs < high))[0]

      if len(idx) == 0:
        features[pk] = 0.0
        continue

      # Extract the relevant slice of Frequency and Power
      band_freqs = freqs[idx]
      band_psd = psd[idx]

      # 3. Calculate Centroid: Sum(f * P(f)) / Sum(P(f))
      #    Add epsilon to denominator to prevent division by zero if power is 0
      total_band_power = np.sum(band_psd)

      if total_band_power == 0:
        features[pk] = 0.0
      else:
        centroid = np.sum(band_freqs * band_psd) / total_band_power
        features[pk] = centroid

    return features

