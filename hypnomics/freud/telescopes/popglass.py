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
from collections import OrderedDict
from hypnomics.freud.nebula import Nebula
from hypnomics.freud.studio.hypno_studio import HypnoStudio
from pictor import Pictor
from pictor.plotters.plotter_base import Plotter
from roma import console

import matplotlib.pyplot as plt



class PopGlass(Pictor):

  def __init__(self, nebula: Nebula, x_key: str, y_key: str,
               title='PopGlass', figure_size=(10, 6), meta_keys=(), **kwargs):
    # Call parent's constructor
    super().__init__(title, figure_size)

    self.nebula = nebula
    self.x_key = x_key
    self.y_key = y_key

    self.objects = self.nebula.labels
    self.add_plotter(PopStudio(self))

    self.meta_keys = meta_keys

    # Set kwargs
    self.kwargs = kwargs


  def save(self):
    """Save nebula to file"""
    import tkinter as tk

    file_path = tk.filedialog.asksaveasfilename(
      title='Save as', filetypes=[('NEBULA files', '*.nebula')])
    if file_path is not None: self.nebula.save(file_path)



class PopStudio(Plotter):

  def __init__(self, pictor):
    # Call parent's constructor
    super().__init__(self.plot, pictor)

    self.new_settable_attr('pad', 0.2, float,
                           'Padding of galaxy border')
    self.new_settable_attr('bottom_ratio', 0.0, float,
                           'Height ratio of bottom panel')
    self.new_settable_attr('bottom_plotter', None, str,
                           'Function to plot bottom panel')
    self.new_settable_attr('outlier_coef', 0, float,
                           '`alpha` coefficient for removing outliers')

  @property
  def nebula(self) -> Nebula: return self.pictor.nebula

  @property
  def probe_keys(self): return self.pictor.x_key, self.pictor.y_key

  # region: Private Methods

  def _get_buffer_key(self, psg_label):
    pad = self.get('pad')
    return f'{psg_label}_{"_".join(self.probe_keys)}_pad{pad}_buffer'

  # endregion: Private Methods

  # region: Public Methods

  def preload(self, overwrite:int=0):
    """Preload all buffer_dict"""
    console.show_status('Preload all fingerprints ...')
    N = len(self.nebula.labels)
    n_channels = len(self.nebula.channels)
    with self.pictor.busy('Preloading fingerprints ...'):
      for i, psg_label in enumerate(self.nebula.labels):
        console.print_progress(i, N)
        _key = self._get_buffer_key(psg_label)

        # Continue if already exist
        if self.nebula.in_pocket(_key) and not overwrite: continue

        _bd = {}
        # Use HypnoStudio to preload
        HypnoStudio.plot_distribution(
          [None] * n_channels, self.nebula, psg_label,
          self.nebula.channels, self.probe_keys, pad=self.get('pad'),
          align_to_galaxy=True, buffer=_bd)

        self.nebula.put_into_pocket(_key, _bd, local=True, exclusive=False)

    console.show_status(f'Preloaded {N} groups of fingerprints.')

  def plot(self, x, fig: plt.Figure):
    # Create layout
    n_channels = len(self.nebula.channels)
    n_cols = min(3, n_channels)

    br = self.get('bottom_ratio')
    bottom_panel = None
    axes = HypnoStudio.make_layout(fig, n_channels, n_cols, hg_ratio=br)
    if br > 0: axes, bottom_panel = axes

    # ~ Get buffer
    buffer_key = self._get_buffer_key(x)
    buffer_dict = self.nebula.get_from_pocket(buffer_key, {})

    # Plot joint-distribution
    HypnoStudio.plot_distribution(
      axes, self.nebula, x, self.nebula.channels, self.probe_keys,
      pad=self.get('pad'), align_to_galaxy=True, buffer=buffer_dict,
      outlier_coef=self.get('outlier_coef'))

    # ~ Put buffer into nebula's pocket if not exist
    if not self.nebula.in_pocket(buffer_key):
      self.nebula.put_into_pocket(buffer_key, buffer_dict, local=True)

    # ~ Plot at bottom panel
    b_plotter = self.get('bottom_plotter')
    if callable(b_plotter): b_plotter(bottom_panel, x)

    # Display super title
    # properties = self.nebula.meta.get(x, {})

    meta: dict = self.nebula.meta.get(x, {})
    properties = OrderedDict(
      (mk, meta[mk]) for mk in self.pictor.meta_keys
    )

    prop_str = ', '.join([f'{k}: {v}' for k, v in properties.items()])
    if prop_str: prop_str = f' | {prop_str}'
    fig.suptitle(f'{x}{prop_str}')

  # endregion: Public Methods
