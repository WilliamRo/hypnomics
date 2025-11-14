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
"""This module implements Higuchi's fractal dimension algorithm.

Reference:
[1] Higuchi, T. "Approach to an irregular time series on the basis of the
    fractal theory". Physica D: Nonlinear Phenomena. 1988: 31(2), 277-283.
"""
import numpy as np



def calculate_L_m_k(x: np.ndarray, k: int, m: int) -> float:
  """Calculate the length of the curve for given k and m."""
  N = len(x)
  num_segments = int(np.floor((N - m) / k))

  L_m_k = 0.0
  for i in range(1, num_segments + 1):
    L_m_k += abs(x[m + i * k - 1] - x[m + (i - 1) * k - 1])
  L_m_k *= (N - 1) / (num_segments * k)
  L_m_k /= k

  return L_m_k



def calculate_mean_L_k(x: np.ndarray, k: int) -> float:
  """Calculate the mean length of the curve for given k."""
  L_k = 0.0
  for m in range(1, k + 1):
    L_k += calculate_L_m_k(x, k, m)

  return L_k / k



def higuchi_fd(x: np.ndarray, k_max: int) -> float:
  """Calculate the Higuchi's fractal dimension of a time series.

  Args:
    x: 1D numpy array, the time series data.
    k_max: int, the maximum value of k.
  Returns:
    float, the estimated fractal dimension.
  """
  L = np.zeros(k_max)
  for k in range(1, k_max + 1):
    L[k - 1] = calculate_mean_L_k(x, k)
  ln_L = np.log(L)
  ln_k = np.log(np.arange(1, k_max + 1))
  coeffs = np.polyfit(ln_k, ln_L, 1)

  return -coeffs[0]



if __name__ == '__main__':
  """The numerical application to simulate data and test the algorithm described
  in Higuchi's paper [1] using Brownian motion with fractal dimension D = 1.5.
  """
  import matplotlib.pyplot as plt

  # (1) Generate the simulated data Y(i) (i = 1, 2, ..., N) with the fractal
  #     dimension D = 1.5
  N = 2 ** 15
  np.random.seed(42)
  Z = np.random.normal(0, 1, N + 1000)
  Y = np.cumsum(Z)[1000:]

  # (2) Calculate L(k) for k = 2^1, 2^2, ..., 2^10
  k_list = [2 ** i for i in range(1, 11)]
  L_list = [calculate_mean_L_k(Y, k) for k in k_list]

  # (3) Plot ln(L(k)) versus ln(k) and perform linear regression
  ln_k = np.log2(k_list)
  ln_L = np.log2(L_list)
  coeffs = np.polyfit(ln_k, ln_L, 1)
  fit_line = np.polyval(coeffs, ln_k)
  fd_estimate = -coeffs[0]

  plt.plot(ln_k, ln_L, 'o')
  plt.plot(ln_k, fit_line, 'r-', alpha=0.5,
           label=f'Fit line (slope = {coeffs[0]:.4f})')
  plt.xlabel('$\log_2(k)$')
  plt.ylabel('$\log_2(L(k))$')
  plt.title(f'Higuchi Fractal Dimension Estimate: D = {fd_estimate:.4f}')
  plt.legend()
  plt.grid()
  plt.show()






