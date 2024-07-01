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

import numpy as np

from hypnomics.hypnoprints.hp_extractor import STAGE_KEYS
from pictor.xomics.misc.distribution import remove_outliers
from roma import Nomear
from roma import io



# STAGE_KEYS = ('W', 'N1', 'N2', 'N3', 'R')

class Nebula(Nomear):
  """Nebula.data_dict =
  {
    (clouds_label, channel, probe_key):
    {'W': [...], 'N1': [...], 'N2': [...], 'N3': [...], 'R': [...]}
  }
  """

  STAGE_KEYS = STAGE_KEYS

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

  @property
  def epoch_num_dict(self) -> dict:
    od = OrderedDict()
    ck, pk = self.channels[0], self.probe_keys[0]
    for label in self.labels:
      od[label] = OrderedDict()
      for sk in STAGE_KEYS:
        od[label][sk] = len(self.data_dict[(label, ck, pk)][sk])
    return od

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

  def dual_view(self, x_key='FREQ-20', y_key='AMP-1', viewer_class=None,
                title=None, fig_size=(10, 6), **configs):
    """Visualize nebula in a 2-D space"""
    # Set default arguments
    if title is None: title = f'Delta={self.delta}'

    # Use default viewer (to be deprecated) if not specified
    if viewer_class is None:
      from sc.fp_viewer import FPViewer

      fps = self.to_walker_results(x_key=x_key, y_key=y_key)
      fpv = FPViewer(walker_results=fps, title=title, figure_size=fig_size)
      for k, v in configs.items(): fpv.plotters[0].set(k, v)
      fpv.show()
    else:
      # Use specified viewer
      viewer = viewer_class(nebula=self, x_key=x_key, y_key=y_key,
                            title=title, figure_size=fig_size)
      for k, v in configs.items(): viewer.plotters[0].set(k, v)
      viewer.show()

  def set_labels(self, labels: list, check_sub_set=True):
    if check_sub_set:
      for lb in labels:
        assert lb in self.labels, f"!! Label `{lb}` not found !!"
    self.put_into_pocket('labels', labels, local=True, exclusive=False)

  # endregion: Public Methods

  # region: IO

  @staticmethod
  def load(path: str, verbose=True) -> 'Nebula':
    assert path.endswith('.nebula'), "!! Nebula file should end with '.nebula' !!"
    return io.load_file(path, verbose=verbose)

  def save(self, path: str, verbose=True):
    if not path.endswith('.nebula'): path = f'{path}.nebula'
    io.save_file(self, path, verbose=verbose)

  # endregion: IO

  # region: Clouds Analysis

  def get_epoch_total(self, label, stage_keys=STAGE_KEYS, excludes=()) -> int:
    return sum([self.epoch_num_dict[label][sk]
                for sk in stage_keys if sk not in excludes])

  # endregion: Clouds Analysis

  # region: Overridden Methods

  def __getitem__(self, item):
    if isinstance(item, (list, tuple)):
      neb = Nebula(self.time_resolution, name=f'{self.name}')
      for chn in self.channels: neb.channels.append(chn)
      for pk in self.probe_keys: neb.probe_keys.append(pk)

      for label in item:
        assert label in self.labels, f"!! Label `{label}` not found !!"
        neb.labels.append(label)
        for chn in self.channels:
          for pk in self.probe_keys:
            neb.data_dict[(label, chn, pk)] = self.data_dict[(label, chn, pk)]

      return neb

  # endregion: Overridden Methods

  # region: Lab Methods

  # region: Reference

  def get_center(self, label, chn, pk, stage_key):
    cloud = self.data_dict[(label, chn, pk)][stage_key]
    cloud = remove_outliers(cloud)
    return np.mean(cloud)

  # endregion: Reference

  # endregion: Lab Methods

