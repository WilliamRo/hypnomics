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
# ====-==================================================================-======
from collections import OrderedDict
from roma import Nomear



class ProbeGroup(Nomear):

  probe_keys = []

  def generate_clouds(self, segments, channel_index, sg=None, tr=None) -> dict:
    """Returns a dictionary of clouds for each probe key.

    Args:
      segments: a dictionary of lists of multichannel physiological data
        e.g., {'W': [array_1, ...], 'N1': [...], ...},
              where array_1.shape = (L, n_channels)
      channel_index: int, index of the channel to be processed
      sg: SignalGroup, optional, the signal group object
    """
    from hypnomics.hypnoprints.hp_extractor import STAGE_KEYS

    # (0) Sanity check
    assert len(self.probe_keys) > 0

    # (1) Initialization
    clouds_dict = OrderedDict()
    for pk in self.probe_keys:
      clouds_dict[pk] = OrderedDict()
      for sk in STAGE_KEYS: clouds_dict[pk][sk] = []

    # (2) Generate clouds for each stage
    for sk in STAGE_KEYS:
      for array in segments[sk]:
        assert len(array.shape) == 2
        feature_dict = self._generate_feature_dict(array[:, channel_index])
        assert len(feature_dict) == len(clouds_dict)
        for pk, value in feature_dict.items():
          clouds_dict[pk][sk].append(value)

    # (-1) ..
    return clouds_dict


  def _generate_feature_dict(self, array) -> dict:
    raise NotImplementedError


  def __call__(self, segments, channel_index) -> dict:
    return self.generate_clouds(segments, channel_index)
