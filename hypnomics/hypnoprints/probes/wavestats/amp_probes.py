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
# ==-=========================================================================-=
import numpy as np



def estimate_amp_envelope_v1(s: np.ndarray, fs, window_size=1.0) -> float:
  """Estimate average amplitude of a signal based on signal envelope.

  Args:
    s: np.ndarray, input signal
    fs: float, sampling frequency
    window_size: float, window size in seconds

  Returns: float, average amplitude
  """
  # Calculate envelope
  size = int(window_size * fs)
  upper, lower = calculate_envelop(s, size)
  mean_amp = np.average(upper - lower)
  return mean_amp


# region: Utilities

def calculate_envelop(s: np.ndarray, size: int):
  """Calculate envelope of a signal

  Args:
    s: np.ndarray, input signal
    size: int, window size

  Returns: np.ndarray, upper and lower envelopes
  """
  p = (size - 1) // 2

  shifted_signals = [s]
  for i in range(1, p + 1):
    # i = 1, 2, ..., p
    s_l = np.concatenate([[s[0]] * i, s[:-i]])
    s_r = np.concatenate([s[i:], [s[-1]] * i])
    shifted_signals.extend([s_l, s_r])

  aligned_signals = np.stack(shifted_signals, axis=-1)
  upper = np.max(aligned_signals, axis=-1)
  lower = np.min(aligned_signals, axis=-1)

  return upper, lower

# endregion: Utilities
