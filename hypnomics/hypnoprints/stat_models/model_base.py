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
                       p_modifier=1.0, q_modifier=1.0):
    """Calculate total variation distance between two distributions."""
    xmin = min(p.dataset.min(), q.dataset.min())
    xmax = max(p.dataset.max(), q.dataset.max())

    # Define a common set of points for evaluation
    X = np.linspace(xmin, xmax, self.KDE_POINTS)
    d = X[1] - X[0]

    # Evaluate the densities
    P, Q = p(X) * p_modifier, q(X) * q_modifier

    tvd = 0.5 * np.sum(np.abs(P - Q) * d)
    return tvd

  # endregion: Common Estimators
