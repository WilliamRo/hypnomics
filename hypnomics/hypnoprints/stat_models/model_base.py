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
# ==-=======================================================================-===
from roma import Nomear
from scipy import stats

import numpy as np



class HypnoModelBase(Nomear):

  # region: APIs

  def calc_distance(self, data_dict_1: dict, data_dict_2: dict):
    raise NotImplementedError

  # endregion: APIs

  # region: Common Estimators

  KDE_KERNEL = 'gaussian'
  KDE_PADDING = 0.1
  KDE_POINTS = 100

  def estimate_kde_1d(self, x: np.ndarray):
    """TODO: deprecated"""
    # Generate X grid
    xmin, xmax = np.min(x), np.max(x)
    pad = (xmax - xmin) * self.KDE_PADDING
    xmin, xmax = xmin - pad, xmax + pad
    X = np.linspace(xmin, xmax, self.KDE_POINTS)

    # Estimate KDE
    assert self.KDE_KERNEL == 'gaussian', "!! Only support Gaussian kernel !!"
    kde = stats.gaussian_kde(x)
    Y = kde(X)
    return X, Y



  def calc_distance_tv(self, p: stats.gaussian_kde, q: stats.gaussian_kde,
                       p_modifier=1.0, q_modifier=1.0, p_shifts=None):
    """Calculate total variation distance between two distributions."""
    # Set `kde_1_shifts` to default if not provided
    if p_shifts is None: p_shifts = [0.0] * p.d
    assert len(p_shifts) == p.d == q.d

    # Determine bounds
    low_bounds = [min(p_data.min() + c, q_data.min())
                  for p_data, q_data, c in zip(p.dataset, q.dataset, p_shifts)]
    high_bounds = [max(p_data.max() + c, q_data.max())
                   for p_data, q_data, c in zip(p.dataset, q.dataset, p_shifts)]

    # E.g., X = [0.1, 0.2, ...], Y = [0.4, 0.5, ...]
    grid_data = [np.linspace(low, high, self.KDE_POINTS)
                 for low, high in zip(low_bounds, high_bounds)]
    q_positions = np.vstack([grid.ravel() for grid in np.meshgrid(*grid_data)])
    p_positions = q_positions - np.reshape(p_shifts, (-1, 1))

    P, Q = p(p_positions) * p_modifier, q(q_positions) * q_modifier

    d = np.prod([grid[1] - grid[0] for grid in grid_data])

    tvd = 0.5 * np.sum(np.abs(P - Q) * d)
    return tvd



  def calc_integral(self, p: stats.gaussian_kde, p_modifier=1.0):
    low_bounds = [data.min() for data in p.dataset]
    high_bounds = [data.max() for data in p.dataset]

    return p.integrate_box(low_bounds, high_bounds) * p_modifier

  # endregion: Common Estimators
