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
from hypnomics.freud.nebula import Nebula
from hypnomics.freud.studio.hypno_studio import HypnoStudio
from pictor import Pictor
from pictor.plotters.plotter_base import Plotter

import matplotlib.pyplot as plt



class PopGlass(Pictor):

  def __init__(self, nebula: Nebula, x_key: str, y_key: str,
               title='PopGlass', figure_size=(10, 6), **kwargs):
    # Call parent's constructor
    super().__init__(title, figure_size)

    self.nebula = nebula
    self.x_key = x_key
    self.y_key = y_key

    self.objects = self.nebula.labels
    self.add_plotter(PopStudio(self))

    # Set kwargs
    self.kwargs = kwargs



class PopStudio(Plotter):

  def __init__(self, pictor):
    # Call parent's constructor
    super().__init__(self.plot, pictor)

    self.new_settable_attr('pad', 0.2, float,
                           'Padding of galaxy border')

  @property
  def nebula(self) -> Nebula: return self.pictor.nebula

  @property
  def probe_keys(self): return self.pictor.x_key, self.pictor.y_key

  def plot(self, x, fig: plt.Figure):
    # Create layout
    n_channels = len(self.nebula.channels)
    n_cols = min(3, n_channels)

    axes = HypnoStudio.make_layout(fig, n_channels, n_cols)
    HypnoStudio._plot_distribution(
      axes, self.nebula, x, self.nebula.channels, self.probe_keys,
      pad=self.get('pad'), align_to_galaxy=True)

    fig.suptitle(f'{x}')

