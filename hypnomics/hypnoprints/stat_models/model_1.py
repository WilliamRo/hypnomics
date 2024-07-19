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
from .model_base import HypnoModelBase
from scipy import stats

import numpy as np



class HypnoModel1(HypnoModelBase):
  r"""Model description:

     \psi_{m, n}(t) = \gamma_m + \upsilon_{m, n}(t)

     Random process \psi_{m, n}(t) represents short-time features extracted by
     some extractor f. If exist some f, s.t., Var[\upsilon_{m, n}(t)] = 0,
     then the PDF of \psi_{m, n}(t) is reduced to

     p_{\psi_{m, n}(t)}(x, t) = p_{\gamma_m}(x - c), in which c is a constant.
   """

  def __init__(self, cond_keys=('N1', 'N2', 'N3', 'R'), pdf_estimator='kde',
               distance_estimator='tv'):
    self.cond_keys = cond_keys
    self.distance_estimator = distance_estimator
    self.pdf_estimator = pdf_estimator


  def calc_distance(self, data_dict_1: dict, data_dict_2: dict):
    # Calculate total number for estimating p(cond)
    N_1, N_2 = [sum([len(d[k]) for k in self.cond_keys])
                for d in (data_dict_1, data_dict_2)]

    # Estimate constant c
    c_pool = []
    for key in self.cond_keys:
      if key not in data_dict_1 and key not in data_dict_2: continue

      x_1, x_2 = data_dict_1[key], data_dict_2[key]
      _c = np.median(x_2) - np.median(x_1)
      weight = (len(x_1) + len(x_2)) / (N_1 + N_2)
      c_pool.append(_c * weight)

    c = np.sum(c_pool)

    # Calculate distances
    distances = []
    for key in self.cond_keys:
      if key not in data_dict_1 and key not in data_dict_2: continue
      assert key in data_dict_1 and key in data_dict_2, f"!! Key {key} not found !!"

      x_1, x_2 = data_dict_1[key], data_dict_2[key]
      m_1, m_2 = len(x_1) / N_1, len(x_2) / N_2

      # Apply shift
      x_1 = np.array(x_1) + c
      # Estimate PDF
      kde_1, kde_2 = [stats.gaussian_kde(x) for x in (x_1, x_2)]

      # Calculate distance
      distances.append(self.calc_distance_tv(kde_1, kde_2, m_1, m_2))

    return sum(distances)

