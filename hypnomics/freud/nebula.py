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
# ==-========================================================================-==
from collections import OrderedDict
from roma import Nomear
from roma import io



class Nebula(Nomear):

  def __init__(self, time_resolution: int, name: str = 'Nebula'):
    assert 30 % time_resolution == 0, "!! Time resolution should be a factor of 30 !!"
    self.time_resolution = time_resolution
    self.name = name

  # region: Properties

  @property
  def delta(self): return self.time_resolution

  @Nomear.property(local=True)
  def labels(self): return []

  @Nomear.property(local=True)
  def channels(self): return []

  @Nomear.property(local=True)
  def probe_keys(self): return []

  @Nomear.property(local=True)
  def data_dict(self): return OrderedDict()

  # endregion: Properties

  # region: Public Methods

  def to_walker_results(self, x_key='FREQ-20', y_key='AMP-1') -> dict:
    """Return old-version FPViewer display format"""
    fps = OrderedDict()

    probe_dict = OrderedDict()
    probe_dict[x_key] = ('', [''])
    probe_dict[y_key] = ('', [''])
    fps['meta'] = (self.labels, self.channels, probe_dict)

    for (sg_label, chn, probe_key), clouds in self.data_dict.items():
      fps[(sg_label, chn, (probe_key, '', ''))] = clouds

    return fps

  # endregion: Public Methods

  # region: IO

  @staticmethod
  def load(path: str, verbose=True) -> 'Nebula':
    return io.load_file(path, verbose=verbose)

  def save(self, path: str, verbose=True):
    io.save_file(self, path, verbose=verbose)

  # endregion: IO

