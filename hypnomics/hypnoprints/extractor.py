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
# =-===========================================================================-
from collections import OrderedDict
from hypnomics.freud.nebula import Nebula
from pictor.xomics.misc.distribution import remove_outliers
from pictor.xomics.misc.distribution import remove_outliers_for_list
from roma import console
from roma import Nomear

import numpy as np



class Extractor(Nomear):
  """Extractor signature:

     def extract_xx(nebula, label) -> dict:
       feature_dict = ...
       return feature_dict
  """

  DEFAULT_SETTINGS = {
    'include_proportion': True,
    'include_stage_shift': True,
    'include_channel_shift': True,
    'include_stage_wise_covariance': True,
  }

  def __init__(self, **settings):
    # Initialize settings
    self.settings = self.DEFAULT_SETTINGS.copy()
    self.settings.update(settings)

    # Add build-in extractors according to settings
    if self.settings['include_proportion']:
      self.extractors.append(self.extract_proportion)

    if self.settings['include_stage_shift']:
      self.extractors.append(self.extract_stage_shift)

    if self.settings['include_channel_shift']:
      self.extractors.append(self.extract_channel_shift)

    if self.settings['include_stage_wise_covariance']:
      self.extractors.append(self.extract_stage_wise_covariance)

  # region: Properties

  @Nomear.property()
  def extractors(self): return []

  # endregion: Properties

  # region: Public Methods

  def extract(self, nebula: Nebula, return_dict=False):
    feature_dict = OrderedDict()

    for label in nebula.labels:
      x_dict = OrderedDict()

      for extractor in self.extractors: x_dict.update(extractor(nebula, label))

      if return_dict: feature_dict[label] = x_dict
      else: feature_dict[label] = np.array(list(x_dict.values()))

    return feature_dict

  # endregion: Public Methods

  # region: Private Methods

  def _calc_mean(self, data):
    if self.settings.get('remove_outliers', True):
      data = remove_outliers(data)
    return np.mean(data)

  # endregion: Private Methods

  # region: Build-in Extractors

  def extract_stage_wise_covariance(self, nebula: Nebula, label,
                                    probe_keys=None):
    """Extract stage-wise covariance."""
    x_dict = OrderedDict()

    if probe_keys is None: probe_keys = nebula.probe_keys
    for ck in nebula.channels:
      for sk in nebula.STAGE_KEYS:
        n_probes = len(probe_keys)
        fn_suffix = f'{sk}_{ck.split(" ")[1]}'
        for i in range(n_probes):
          pi = probe_keys[i]
          cloud_i = nebula.data_dict[(label, ck, pi)][sk]
          for j in range(i, n_probes):
            pj = probe_keys[j]
            cloud_j = nebula.data_dict[(label, ck, pj)][sk]

            MAX_LEN = 5
            if i == j:
              fn_prefix = f'STD_{pi[:MAX_LEN]}'
            else:
              fn_prefix = f'COV_{pi[:MAX_LEN]}/{pj[:MAX_LEN]}'

            # TODO: Handle empty cloud
            if len(cloud_i) < 2:
              value = 0
            else:
              value = (np.std(cloud_i, dtype=float) if i == j
                           else np.cov(cloud_i, cloud_j)[0, 1])

            assert not np.isnan(value)
            x_dict[f'{fn_prefix}_{fn_suffix}'] = value

    return x_dict

  def extract_stage_shift(self, nebula: Nebula, label):
    """Extract stage shift from N2."""
    x_dict = OrderedDict()

    for ck in nebula.channels:
      for pk in nebula.probe_keys:
        n2_cloud = nebula.data_dict[(label, ck, pk)]['N2']
        mu_N2 = self._calc_mean(n2_cloud)

        for sk in nebula.STAGE_KEYS:
          if sk == 'N2': continue
          key = f'SS_{sk}_{pk}_{ck.split(" ")[1]}'

          cloud = nebula.data_dict[(label, ck, pk)][sk]
          if len(cloud) == 0:
            # TODO: Handle empty cloud
            value = 0
          else:
            mu_sk = self._calc_mean(cloud)
            value = mu_sk - mu_N2

          x_dict[key] = value

    return x_dict

  def extract_channel_shift(self, nebula: Nebula, label, ch0_i=0):
    x_dict = OrderedDict()

    for pk in nebula.probe_keys:
      for sk in nebula.STAGE_KEYS:
        cloud = nebula.data_dict[(label, nebula.channels[ch0_i], pk)][sk]
        if len(cloud) == 0:
          ch0_mu = 0
        else:
          ch0_mu = self._calc_mean(cloud)

        for i, ck in enumerate(nebula.channels):
          if i == ch0_i: continue
          key = f'CS_{ck.split(" ")[1]}_{sk}_{pk}'

          cloud = nebula.data_dict[(label, ck, pk)][sk]
          if len(cloud) == 0:
            value = 0
          else:
            mu = self._calc_mean(cloud)
            value = mu - ch0_mu

          x_dict[key] = value

    return x_dict

  @staticmethod
  def extract_proportion(nebula: Nebula, label):
    """Extract proportion of each sleep stage."""
    x_dict = OrderedDict()

    total = nebula.get_epoch_total(label, excludes=['W'])
    for sk in nebula.STAGE_KEYS:
      if sk == 'W': continue
      x_key = f'P_{sk}'
      x_dict[x_key] = nebula.epoch_num_dict[label][sk] / total

    return x_dict

  # endregion: Build-in Extractors
