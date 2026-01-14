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
from hypnomics.freud.flow import Flow
from pictor.plotters.plotter_base import Plotter
from pictor.xomics.misc.distribution import remove_outliers_for_list
from roma import console

import matplotlib.pyplot as plt
import numpy as np



class Poincare(Plotter):

  STAGE_COLORS = {
    # see https://matplotlib.org/stable/gallery/color/named_colors.html
    'W': 'forestgreen', 'N1': 'gold', 'N2': 'orange', 'N3': 'royalblue',
    'R': 'lightcoral', 'X': 'gray',
  }

  def __init__(self, pictor):
    # Call parent's constructor
    super().__init__(self.plot, pictor)

    self.new_settable_attr('traj', True, bool,
                           'option to show trajectory')

    self.new_settable_attr('xmin', None, float, 'x-min')
    self.new_settable_attr('xmax', None, float, 'x-max')
    self.new_settable_attr('ymin', None, float, 'y-min')
    self.new_settable_attr('ymax', None, float, 'y-max')

    self.new_settable_attr('margin', 0.2, float, 'margin')

  # region: Properties

  @property
  def flow(self) -> Flow: return self.pictor.nebula

  @property
  def selected_traj(self) -> str: return self.pictor.selected_clouds

  @property
  def stages(self) -> list:
    return self.flow.stages[self.selected_traj]

  @property
  def selected_channel(self) -> str: return self.pictor.selected_channel

  @property
  def x_key(self): return self.pictor.x_key

  @property
  def y_key(self): return self.pictor.y_key

  # endregion: Properties

  # region: Plotting Methods

  def plot(self, ax: plt.Axes, fig: plt.Figure):
    """
    res_dict = {pk_1: [...], pk_2: [...]}
    """
    # (0) Get selected data pair
    res_dict: dict = self.pictor.selected_cluster_dict
    # assert len(res_dict) == 2

    # (1) Plot data
    x_key, y_key = self.pictor.x_key, self.pictor.y_key

    Xs, Ys = res_dict[x_key], res_dict[y_key]
    Cs = self.stages  # Color index
    assert len(Xs) == len(Ys) == len(Cs)

    # PLOT
    i2s = ['W', 'N1', 'N2', 'N3', 'R', 'X']
    if self.get('traj'):
      # Plot trajectory with color
      for i in range(1, len(Xs)):
        c = Cs[i]
        ax.plot([Xs[i-1], Xs[i]], [Ys[i-1], Ys[i]],
                c=self.STAGE_COLORS[i2s[c]], alpha=0.5)
    else:
      # Plot scatter with color and legend
      ax.scatter(Xs, Ys, c=[self.STAGE_COLORS[i2s[c]] for c in Cs], alpha=0.5)

    # (2) Set title, axis labels, and legend
    traj_label = self.selected_traj
    channel_label = self.selected_channel

    ax.set_title(f'{traj_label} ({channel_label}){self.pictor.meta_suffix}')

    ax.set_xlabel(self.pictor.x_key)
    ax.set_ylabel(self.pictor.y_key)

    # Set color label
    ax.legend()

    # (3) Set limits
    X_all, Y_all = remove_outliers_for_list(Xs, Ys)

    xmin, xmax = np.min(X_all), np.max(X_all)
    ymin, ymax = np.min(Y_all), np.max(Y_all)

    m = self.get('margin')
    xm, ym = (xmax - xmin) * m, (ymax - ymin) * m
    xmin, xmax = xmin - xm, xmax + xm
    ymin, ymax = ymin - ym, ymax + ym

    xmin = self.get('xmin') if self.get('xmin') is not None else xmin
    xmax = self.get('xmax') if self.get('xmax') is not None else xmax
    ymin = self.get('ymin') if self.get('ymin') is not None else ymin
    ymax = self.get('ymax') if self.get('ymax') is not None else ymax

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

  # endregion: Plotting Methods

  # region: Private Methods

  # endregion: Private Methods

  # region: Shortcuts

  def register_shortcuts(self):
    self.register_a_shortcut('Left', lambda: self.set_lim('xmin'), 'Set xmin')
    self.register_a_shortcut('Right', lambda: self.set_lim('xmax'), 'Set xmax')
    self.register_a_shortcut('Down', lambda: self.set_lim('ymin'), 'Set ymin')
    self.register_a_shortcut('Up', lambda: self.set_lim('ymax'), 'Set ymax')

    self.register_a_shortcut('t', lambda: self.flip('traj'),
                             'Flip trajectory option')

  # endregion: Shortcuts

  # region: MISC

  def set_lim(self, key):
    if self.get(key) is not None:
      self.set(key)
      return

    ax = self.pictor.canvas.axes2D
    lim_dict = {}
    lim_dict['xmin'], lim_dict['xmax'] = ax.get_xlim()
    lim_dict['ymin'], lim_dict['ymax'] = ax.get_ylim()
    self.set(key, value=lim_dict[key])

  # endregion: MISC
