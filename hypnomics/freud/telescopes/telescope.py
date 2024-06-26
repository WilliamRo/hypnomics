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
# ===-===================================================================-======
from pictor import Pictor
from hypnomics.freud.nebula import Nebula

from .hans import Hans



class Telescope(Pictor):

  class Keys(Pictor.Keys):
    CHANNELS = 'ChAnNeLs'

  def __init__(self, nebular: Nebula, x_key: str, y_key: str,
               title='Telescope', figure_size=(10, 6), **kwargs):
    # Call parent's constructor
    super(Telescope, self).__init__(title, figure_size=figure_size)
    self.nebular = nebular
    self.x_key = x_key
    self.y_key = y_key
    self._initialize_scope()

    # Add plotter
    self.add_plotter(Hans(self))

  # region: Properties

  @property
  def selected_clouds(self): return self.get_element(self.Keys.OBJECTS)

  @property
  def selected_channel(self): return self.get_element(self.Keys.CHANNELS)

  @property
  def selected_pair(self):
    # e.g., 'sleepedf-SC4001E'
    sg_label = self.selected_clouds
    # e.g., 'EEG Fpz-Cz'
    channel_label = self.selected_channel

    pair_dict = {}
    for key in (self.x_key, self.y_key):
      pair_dict[key] = self.nebular.data_dict[(sg_label, channel_label, key)]
    return pair_dict

  # endregion: Properties

  # region: Private Methods

  def _initialize_scope(self):
    # Set axis
    self.objects = self.nebular.labels
    self.set_to_axis(self.Keys.CHANNELS, self.nebular.channels)

    # Initialize shortcuts
    self.shortcuts._library.pop('Escape')
    self._register_key('l', 'Next channel', self.Keys.CHANNELS, 1)
    self._register_key('h', 'Previous channel', self.Keys.CHANNELS, -1)

  def _register_key(self, btn, des, key, v):
    self.shortcuts.register_key_event(
      [btn], lambda: self.set_cursor(key, v, refresh=True),
      description=des, color='yellow')

  # endregion: Private Methods
