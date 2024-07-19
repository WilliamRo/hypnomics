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

from .aristotle import Aristotle
from .galileo import Galileo
from .hans import Hans



class Telescope(Pictor):

  class Keys(Pictor.Keys):
    CHANNELS = 'ChAnNeLs'
    PROBES = 'PrObEs'

  def __init__(self, nebula: Nebula, x_key: str, y_key: str,
               title='Telescope', figure_size=(10, 6),
               plotters=('Hans', 'Galileo'), **kwargs):
    # Call parent's constructor
    super(Telescope, self).__init__(title, figure_size=figure_size)
    self.nebula = nebula
    self.x_key = x_key
    self.y_key = y_key
    self._initialize_scope()

    # Add plotter
    if isinstance(plotters, str):
      _plotters = []
      if 'H' in plotters: _plotters.append('Hans')
      if 'G' in plotters: _plotters.append('Galileo')
      if 'A' in plotters: _plotters.append('Aristotle')
      plotters = _plotters

    for plotter_key in plotters:
      plotter_class = {
        'Hans': Hans, 'Galileo': Galileo, 'Aristotle': Aristotle}[plotter_key]
      self.add_plotter(plotter_class(self))

    # Set kwargs
    self.kwargs = kwargs

  # region: Properties

  @property
  def selected_clouds(self): return self.get_element(self.Keys.OBJECTS)

  @property
  def selected_channel(self): return self.get_element(self.Keys.CHANNELS)

  @property
  def selected_probe(self): return self.get_element(self.Keys.PROBES)

  @property
  def selected_cluster_dict(self) -> dict:
    """May be used by (1) dual viewers, (2) single viewers."""
    # e.g., 'sleepedf-SC4001E'
    sg_label = self.selected_clouds
    # e.g., 'EEG Fpz-Cz'
    channel_label = self.selected_channel

    probe_keys = [self.x_key, self.y_key]
    if self.selected_probe not in probe_keys:
      probe_keys.append(self.selected_probe)

    cluster_dict = {}
    for key in probe_keys:
      cluster_dict[key] = self.nebula.data_dict[(sg_label, channel_label, key)]
    return cluster_dict

  def get_stage_dict(self, sg_label, channel_label=None, probe_key=None):
    """Currently this function is used by aristotle only"""
    if channel_label is None: channel_label = self.selected_channel
    if probe_key is None: probe_key = self.selected_probe
    return self.nebula.data_dict[(sg_label, channel_label, probe_key)]

  # endregion: Properties

  # region: Private Methods

  def _initialize_scope(self):
    # Set axis
    self.objects = self.nebula.labels
    self.set_to_axis(self.Keys.CHANNELS, self.nebula.channels)
    self.set_to_axis(self.Keys.PROBES, self.nebula.probe_keys)

    # Initialize shortcuts
    self.shortcuts._library.pop('Escape')
    self._register_key('l', 'Next channel', self.Keys.CHANNELS, 1)
    self._register_key('h', 'Previous channel', self.Keys.CHANNELS, -1)

    self._register_key('N', 'Next probe', self.Keys.PROBES, 1)
    self._register_key('P', 'Previous probe', self.Keys.PROBES, -1)

  def _register_key(self, btn, des, key, v):
    self.shortcuts.register_key_event(
      [btn], lambda: self.set_cursor(key, v, refresh=True),
      description=des, color='yellow')

  # endregion: Private Methods

  # region: Public Methods

  def save(self):
    """Save nebula to file"""
    import tkinter as tk

    file_path = tk.filedialog.asksaveasfilename(
      title='Save as', filetypes=[('NEBULA files', '*.nebula')])
    if file_path is not None: self.nebula.save(file_path)

  # endregion: Public Methods
