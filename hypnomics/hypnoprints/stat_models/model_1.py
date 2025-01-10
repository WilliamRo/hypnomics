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

  MIN_POINTS = 10

  def __init__(self, cond_keys=('N1', 'N2', 'N3', 'R'), pdf_estimator='kde',
               distance_estimator='tv'):
    self.cond_keys = cond_keys
    self.distance_estimator = distance_estimator
    self.pdf_estimator = pdf_estimator


  def est_gauss_kde(self, x: np.ndarray, key_prefix, key_suffix):
    if key_prefix is None: return stats.gaussian_kde(x)

    key = ('GAUSS_KDE_BUFFER', key_prefix, key_suffix)
    return self.get_from_pocket(key, initializer=lambda: stats.gaussian_kde(x))


  def calc_upsilon(self, data_dict_1, data_dict_2, N_1=None, N_2=None,
                   cond_keys=None):
    assert cond_keys is not None

    # Calculate N_1, N_2 if not provided
    if N_1 is None or N_2 is None:
      N_1, N_2 = [sum([len(d[k]) for k in cond_keys])
                  for d in (data_dict_1, data_dict_2)]

    c_pool = []
    for key in cond_keys:
      if key not in data_dict_1 or key not in data_dict_2: continue

      x_1, x_2 = data_dict_1[key], data_dict_2[key]

      if len(x_1) == 0 or len(x_2) == 0: continue

      _c = np.median(x_2) - np.median(x_1)
      weight = (len(x_1) + len(x_2)) / (N_1 + N_2)

      c_pool.append(_c * weight)

    c = np.sum(c_pool)
    return c


  def calc_distance(self, data_dict_1: dict, data_dict_2: dict,
                    data_1_key=None, data_2_key=None, conditional=True,
                    shift_compensation=True):
    """data_dict =
      {'W': [...], 'N1': [...], 'N2': [...], 'N3': [...], 'R': [...]} """
    cond_keys = self.cond_keys

    # Check if conditional
    if not conditional:
      data_dict_1 = {'*': np.concatenate([data_dict_1[k] for k in cond_keys])}
      data_dict_2 = {'*': np.concatenate([data_dict_2[k] for k in cond_keys])}
      cond_keys = ['*']

    # Calculate total number for estimating p(cond)
    N_1, N_2 = [sum([len(d[k]) for k in cond_keys])
                for d in (data_dict_1, data_dict_2)]

    # Estimate constant c
    if shift_compensation:
      c = self.calc_upsilon(data_dict_1, data_dict_2, N_1, N_2,
                            cond_keys=cond_keys)
    else:
      c = 0.0

    # Calculate distances for each condition (sleep stage)
    distances = []
    for key in cond_keys:
      x_1, x_2 = data_dict_1[key], data_dict_2[key]
      m_1, m_2 = len(x_1) / N_1, len(x_2) / N_2

      # Estimate PDF
      kde_1, kde_2 = [
        self.est_gauss_kde(x, k, key) if len(x) > self.MIN_POINTS else None
        for x, k in zip((x_1, x_2), (data_1_key, data_2_key))]

      if any([kde is None for kde in (kde_1, kde_2)]):
        if all([kde is None for kde in (kde_1, kde_2)]): distances.append(0.)
        else:
          kde, m = (kde_1, m_1) if kde_2 is None else (kde_2, m_2)
          assert isinstance(kde, stats.gaussian_kde)
          distances.append(self.calc_integral(kde, m))
      else:
        # Calculate distance
        distances.append(self.calc_distance_tv(kde_1, kde_2, m_1, m_2,
                                               p_shifts=[c]))

    return sum(distances)


  def calc_joint_distance(self, data_dict_pair_1, data_dict_pair_2,
                          key_pair_1=None, key_pair_2=None):
    """data_dict =
      {'W': [...], 'N1': [...], 'N2': [...], 'N3': [...], 'R': [...]} """

    # Unpack data
    data_1_pk1, data_1_pk2 = data_dict_pair_1
    data_2_pk1, data_2_pk2 = data_dict_pair_2

    # Sanity check
    N_1, N_2 = [sum([len(d[k]) for k in self.cond_keys])
                for d in (data_1_pk1, data_2_pk1)]
    N_1_, N_2_ = [sum([len(d[k]) for k in self.cond_keys])
                  for d in (data_1_pk2, data_2_pk2)]
    assert N_1 == N_1_ and N_2 == N_2_, "!! Data size mismatched !!"

    # Estimate constant c
    c_1 = self.calc_upsilon(data_1_pk1, data_2_pk1, N_1, N_2)
    c_2 = self.calc_upsilon(data_1_pk2, data_2_pk2, N_1, N_2)

    # Calculate distances for each condition (sleep stage)
    distances = []
    for key in self.cond_keys:
      x_1_pk1, x_1_pk2 = data_1_pk1[key], data_1_pk2[key]
      x_2_pk1, x_2_pk2 = data_2_pk1[key], data_2_pk2[key]
      m_1, m_2 = len(x_1_pk1) / N_1, len(x_2_pk1) / N_2

      # Estimate PDF
      kde_1 = (self.est_gauss_kde(np.stack([x_1_pk1, x_1_pk2]), key_pair_1, key)
               if len(x_1_pk1) > self.MIN_POINTS else None)
      kde_2 = (self.est_gauss_kde(np.stack([x_2_pk1, x_2_pk2]), key_pair_2, key)
               if len(x_2_pk1) > self.MIN_POINTS else None)

      if any([kde is None for kde in (kde_1, kde_2)]):
        if all([kde is None for kde in (kde_1, kde_2)]): distances.append(0.)
        else:
          kde, m = (kde_1, m_1) if kde_2 is None else (kde_2, m_2)
          distances.append(self.calc_integral(kde, m))
      else:
        # Calculate distance
        distances.append(self.calc_distance_tv(kde_1, kde_2, m_1, m_2,
                                               p_shifts=[c_1, c_2]))

    return sum(distances)
