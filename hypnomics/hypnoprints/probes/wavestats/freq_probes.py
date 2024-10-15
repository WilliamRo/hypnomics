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
# ====-======================================================================-==
from scipy.signal import stft

import numpy as np



def estimate_freq_stft_v1(s: np.ndarray, fs: float, fmin=0.5, fmax=20) -> float:
  """Estimate average frequency of a signal based on STFT.

  Args:
    s: np.ndarray, input signal
    fs: float, sampling frequency
    fmin: float, minimum frequency to consider
    fmax: float, maximum frequency to consider

  Returns: float, average frequency
  """
  # Calculate STFT and spectrum
  f, t, Zxx = stft(s, fs=fs, nperseg=256)
  spectrum = np.abs((Zxx))

  # Clip spectrum
  h_mask = f > fmin
  f, spectrum = f[h_mask], spectrum[h_mask]
  l_mask = f < fmax
  f, spectrum = f[l_mask], spectrum[l_mask]

  # Calculate mean frequency
  dom_f = np.sum(f[..., np.newaxis] * spectrum, axis=0) / (np.sum(
    spectrum, axis=0) + 1e-6)
  return np.average(dom_f)



def estimate_average_freq(s: np.ndarray, fs: float, fmin=0.5, fmax=35) -> float:
  """Estimate average frequency of a signal based on FFT."""
  N = len(s)
  X = np.fft.fft(s)
  X_mag = np.abs(X) / N
  f = np.fft.fftfreq(N, 1 / fs)
  mask = (f > fmin) & (f < fmax)
  avg_f = np.sum(f[mask] * X_mag[mask]) / (np.sum(X_mag[mask]) + 1e-6)
  return avg_f
