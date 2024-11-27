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
    'include_statistical_features': 1,
    'include_inter_stage_features': 1,
    'include_inter_channel_features': 1,

    # Deprecated features
    'include_proportion': False,
    'include_stage_mean': False,
    'include_stage_shift': False,
    'include_stage_wise_covariance': False,
    'include_channel_shift': False,
    'include_all_mean_std': False,
  }

  def __init__(self, probe_keys=None, **settings):
    # Sanity check
    for k in settings:
      assert k in self.DEFAULT_SETTINGS, f"!! Invalid setting: {k} !!"

    # Initialize settings
    self.settings = self.DEFAULT_SETTINGS.copy()

    for k in settings.keys():
      assert k in self.settings, f"!! Invalid setting: {k} !!"
    self.settings.update(settings)

    # Add build-in extractors according to settings
    # Group I - Macro features
    if self.settings['include_proportion']:
      self.extractors.append(self.extract_proportion)

    # Group II - Statistical features
    if self.settings['include_statistical_features']:
      self.extractors.append(self.extract_statistical_features)

    # Group III - Inter-stage features
    if self.settings['include_inter_stage_features']:
      self.extractors.append(self.extract_inter_stage_features)

    # Group IV - Inter-channel features
    if self.settings['include_inter_channel_features']:
      self.extractors.append(self.extract_inter_channel_features)

    # Group X - Deprecated features
    if self.settings['include_stage_wise_covariance']:
      self.extractors.append(self.extract_stage_wise_covariance)

    if self.settings['include_stage_shift']:
      self.extractors.append(self.extract_stage_shift)

    if self.settings['include_channel_shift']:
      self.extractors.append(self.extract_channel_shift)

    if self.settings['include_stage_mean']:
      self.extractors.append(self.extract_stage_mean)

    if self.settings['include_all_mean_std']:
      self.extractors.append(self.extract_mean_std)

    # Attributes
    self.probe_keys = probe_keys
    self.reference_stage = 'N2'
    self.reference_channel_index = 0
    self._x_dict_buffer = None

  # region: Properties

  @Nomear.property()
  def extractors(self): return []

  # endregion: Properties

  # region: Public Methods

  def extract(self, nebula: Nebula, return_dict=False):
    feature_dict = OrderedDict()

    for label in nebula.labels:
      x_dict = OrderedDict()
      self._x_dict_buffer = x_dict

      for extractor in self.extractors: x_dict.update(extractor(nebula, label))

      if return_dict: feature_dict[label] = x_dict
      else: feature_dict[label] = np.array(list(x_dict.values()))

    return feature_dict

  # endregion: Public Methods

  # region: Private Methods

  def _calc_mean(self, data):
    # TODO: Handle empty cloud
    if len(data) == 0: return 0

    if self.settings.get('remove_outliers', True):
      data = remove_outliers(data)
    return np.mean(data)

  def _calc_std(self, data):
    # TODO: Handle empty cloud
    if len(data) == 0: return 0

    if self.settings.get('remove_outliers', True):
      data = remove_outliers(data)
    return np.std(data)

  def _get_ck(self, channel_key: str):
    ck_list = channel_key.split(" ")
    if len(ck_list) == 1:
      # e.g., ck = 'Fpz-Cz'
      _ck = ck_list[0]
    else:
      assert len(ck_list) == 2
      # e.g., ck = 'EEG Fpz-Cz'
      _ck = ck_list[1]
    return _ck

  def _get_cloud(self, neb: Nebula, pid, ck, pk, sk, remove_none=True):
    _cloud = neb.data_dict[(pid, ck, pk)][sk]

    # Remove invalid values from _cloud
    if remove_none: _cloud = np.array(
      [x for x in _cloud if not any([np.isnan(x), np.isinf(x)])])

    return _cloud

  def _remove_none(self, cloud_1, cloud_2):
    cloud_1, cloud_2 = np.array(cloud_1), np.array(cloud_2)
    nan_mask = np.logical_or(np.isnan(cloud_1), np.isnan(cloud_2))
    mask = np.logical_not(nan_mask)
    return cloud_1[mask], cloud_2[mask]

  # endregion: Private Methods

  # region: Build-in Extractors

  def extract_statistical_features(self, nebula: Nebula, label):
    """Extract statistical features, including mean, STD, covariance of each pair
    of probes in each sleep stage in each channel."""
    # Set probe keys
    if self.probe_keys is None: probe_keys = nebula.probe_keys
    else: probe_keys = self.probe_keys
    n_probes = len(probe_keys)

    # Traverse each channel and stage
    x_dict = OrderedDict()
    pairs = [(ck, sk) for ck in nebula.channels for sk in nebula.STAGE_KEYS]
    for ck, sk in pairs:
      # E.g., 'EEG Fpz-Cz' -> 'Fpz-Cz'
      ck_short = self._get_ck(ck)
      sk_ck = f'{sk}_{ck_short}'
      for i in range(n_probes):
        pi = probe_keys[i]
        # cloud_i without None
        cloud_i = self._get_cloud(nebula, label, ck, pi, sk)
        # (1) AVG
        x_dict[f'AVG({pi})_{sk_ck}'] = self._calc_mean(cloud_i)

        # (2) STD
        x_dict[f'STD({pi})_{sk_ck}'] = self._calc_std(cloud_i)

        # (3) COR: Pearson's correlation coefficient
        cloud_i_w_none = self._get_cloud(nebula, label, ck, pi, sk, False)
        for j in range(i + 1, n_probes):
          pj = probe_keys[j]
          cloud_j_w_none = self._get_cloud(nebula, label, ck, pj, sk, False)
          _cloud_i, _cloud_j = self._remove_none(cloud_i_w_none, cloud_j_w_none)
          if len(_cloud_i) < 2: value = 0
          else:
            value = np.corrcoef(_cloud_i, _cloud_j)[0, 1]
            # TODO
            if np.isnan(value): value = 0

          # assert not np.isnan(value) TODO
          x_dict[f'COR({pi},{pj})_{sk_ck}'] = value

    # Return results
    return x_dict

  def extract_inter_stage_features(self, nebula: Nebula, label):
    # (1) Get buffer
    if len(self._x_dict_buffer) == 0:
      x_dict_buffer = self.extract_statistical_features(nebula, label)
    else:
      x_dict_buffer = self._x_dict_buffer
    rsk = self.reference_stage
    assert rsk in nebula.STAGE_KEYS

    # (2) Set probe keys
    if self.probe_keys is None: probe_keys = nebula.probe_keys
    else: probe_keys = self.probe_keys
    n_probes = len(probe_keys)

    # (3) Extract features
    x_dict = OrderedDict()
    pairs = [(ck, sk) for ck in nebula.channels
             for sk in nebula.STAGE_KEYS if sk != rsk]
    for ck, sk in pairs:
      # E.g., 'EEG Fpz-Cz' -> 'Fpz-Cz'
      ck_short = self._get_ck(ck)
      rsk_ck = f'{rsk}_{ck_short}'
      sk_ck = f'{sk}_{ck_short}'
      skr2sk_ck = f'{rsk}->{sk}_{ck_short}'
      for i in range(n_probes):
        pi = probe_keys[i]
        # (1) AVG
        avg_i = x_dict_buffer[f'AVG({pi})_{sk_ck}']
        avg_r = x_dict_buffer[f'AVG({pi})_{rsk_ck}']
        x_dict[f'IS_AVG({pi})_{skr2sk_ck}'] = avg_i - avg_r

        # (2) STD
        std_i = x_dict_buffer[f'STD({pi})_{sk_ck}']
        std_r = x_dict_buffer[f'STD({pi})_{rsk_ck}']
        x_dict[f'IS_STD({pi})_{skr2sk_ck}'] = std_i - std_r

        # (3) COR: Pearson's correlation coefficient
        for j in range(i + 1, n_probes):
          pj = probe_keys[j]
          cor_i_j = x_dict_buffer[f'COR({pi},{pj})_{sk_ck}']
          cor_i_j_r = x_dict_buffer[f'COR({pi},{pj})_{rsk_ck}']
          x_dict[f'IS_COR({pi},{pj})_{skr2sk_ck}'] = cor_i_j - cor_i_j_r

    # Return results
    return x_dict

  def extract_inter_channel_features(self, nebula: Nebula, label):
    # (1) Get buffer
    if len(self._x_dict_buffer) == 0:
      x_dict_buffer = self.extract_statistical_features(nebula, label)
    else:
      x_dict_buffer = self._x_dict_buffer

    # (2) Set probe keys
    if self.probe_keys is None: probe_keys = nebula.probe_keys
    else: probe_keys = self.probe_keys
    n_probes = len(probe_keys)

    # (3) Extract features
    x_dict = OrderedDict()
    ref_channels = nebula.channels[self.reference_channel_index]
    pairs = [(ck, sk) for ck in nebula.channels if ck != ref_channels
             for sk in nebula.STAGE_KEYS]
    for ck, sk in pairs:
      # E.g., 'EEG Fpz-Cz' -> 'Fpz-Cz'
      rck_short = self._get_ck(ref_channels)
      ck_short = self._get_ck(ck)
      sk_rck = f'{sk}_{rck_short}'
      sk_ck = f'{sk}_{ck_short}'
      sk_rck2ck = f'{sk}_{rck_short}->{ck_short}'
      for i in range(n_probes):
        pi = probe_keys[i]
        # (1) AVG
        avg_i = x_dict_buffer[f'AVG({pi})_{sk_ck}']
        avg_r = x_dict_buffer[f'AVG({pi})_{sk_rck}']
        x_dict[f'IC_AVG({pi})_{sk_rck2ck}'] = avg_i - avg_r

        # (2) STD
        std_i = x_dict_buffer[f'STD({pi})_{sk_ck}']
        std_r = x_dict_buffer[f'STD({pi})_{sk_rck}']
        x_dict[f'IC_STD({pi})_{sk_rck2ck}'] = std_i - std_r

        # (3) COR: Pearson's correlation coefficient
        for j in range(i + 1, n_probes):
          pj = probe_keys[j]
          cor_i_j = x_dict_buffer[f'COR({pi},{pj})_{sk_ck}']
          cor_i_j_r = x_dict_buffer[f'COR({pi},{pj})_{sk_rck}']
          x_dict[f'IC_COR({pi},{pj})_{sk_rck2ck}'] = cor_i_j - cor_i_j_r

    # Return results
    return x_dict

  # region: Deprecated features

  def extract_stage_wise_covariance(self, nebula: Nebula, label,
                                    probe_keys=None):
    """Extract stage-wise covariance."""
    x_dict = OrderedDict()

    if probe_keys is None: probe_keys = nebula.probe_keys
    for ck in nebula.channels:
      for sk in nebula.STAGE_KEYS:
        n_probes = len(probe_keys)
        fn_suffix = f'{sk}_{self._get_ck(ck)}'
        for i in range(n_probes):
          pi = probe_keys[i]
          cloud_i = self._get_cloud(nebula, label, ck, pi, sk,
                                    remove_none=False)

          for j in range(i, n_probes):
            pj = probe_keys[j]
            cloud_j = self._get_cloud(nebula, label, ck, pj, sk,
                                      remove_none=False)

            _cloud_i, _cloud_j = self._remove_none(cloud_i, cloud_j)

            MAX_LEN = 5
            if i == j:
              fn_prefix = f'STD_{pi[:MAX_LEN]}'
            else:
              fn_prefix = f'COV_{pi[:MAX_LEN]}/{pj[:MAX_LEN]}'

            # TODO: Handle empty cloud
            if len(_cloud_i) < 2:
              value = 0
            else:
              value = (np.std(_cloud_i, dtype=float) if i == j
                       else np.cov(_cloud_i, _cloud_j)[0, 1])

            assert not np.isnan(value)
            x_dict[f'{fn_prefix}_{fn_suffix}'] = value

    return x_dict

  def extract_stage_shift(self, nebula: Nebula, label):
    """Extract stage shift from N2."""
    x_dict = OrderedDict()

    for ck in nebula.channels:
      for pk in nebula.probe_keys:
        n2_cloud = self._get_cloud(nebula, label, ck, pk, 'N2')
        mu_N2 = self._calc_mean(n2_cloud)

        for sk in nebula.STAGE_KEYS:
          if sk == 'N2': continue
          key = f'SS_{sk}_{pk}_{self._get_ck(ck)}'

          cloud = self._get_cloud(nebula, label, ck, pk, sk)
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
        cloud = self._get_cloud(nebula, label, nebula.channels[ch0_i], pk, sk)
        if len(cloud) == 0:
          ch0_mu = 0
        else:
          ch0_mu = self._calc_mean(cloud)

        for i, ck in enumerate(nebula.channels):
          if i == ch0_i: continue
          key = f'CS_{self._get_ck(ck)}_{sk}_{pk}'

          cloud = self._get_cloud(nebula, label, ck, pk, sk)
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

  def extract_stage_mean(self, nebula: Nebula, label):
    """Extract mean of each sleep stage for each channel."""
    x_dict = OrderedDict()

    for ck in nebula.channels:
      for pk in nebula.probe_keys:
        for sk in nebula.STAGE_KEYS:
          key = f'AVG_{sk}_{pk}_{self._get_ck(ck)}'

          cloud = self._get_cloud(nebula, label, ck, pk, sk)
          if len(cloud) == 0:
            # TODO: Handle empty cloud
            value = 0
          else:
            value = self._calc_mean(cloud)

          x_dict[key] = value

    return x_dict

  def extract_mean_std(self, nebula: Nebula, label):
    """Extract mean and STD of all sleep stage for each channel."""
    x_dict = OrderedDict()

    key_methods = (('AVG', self._calc_mean), ('STD', self._calc_std))

    for ck in nebula.channels:
      for pk in nebula.probe_keys:
        for k, f in key_methods:
          key = f'{k}_{pk}_{ck.split(" ")[1]}'
          cloud = np.concatenate([nebula.data_dict[(label, ck, pk)][sk]
                                  for sk in nebula.STAGE_KEYS])
          assert len(cloud) > 0
          value = f(cloud)

          x_dict[key] = value

    return x_dict

  # endregion: Deprecated features

  # endregion: Build-in Extractors
