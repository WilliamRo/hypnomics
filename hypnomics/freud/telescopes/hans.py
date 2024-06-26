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
# ====-========================================================================-
from matplotlib.patches import Rectangle
from pictor.plotters.plotter_base import Plotter
from pictor.xomics.misc.distribution import remove_outliers_for_list
from hypnomics.freud.nebula import Nebula

import matplotlib.pyplot as plt
import numpy as np



class Hans(Plotter):

  def __init__(self, pictor):
    # Call parent's constructor
    super().__init__(self.plot, pictor)

    self.new_settable_attr('show_scatter', True, bool,
                           'Option to show scatter')
    self.new_settable_attr('show_rect', False, bool,
                           'Option to show region of each stage')
    self.new_settable_attr('show_kde', True, bool,
                           'Option to show KDE for each stage')
    self.new_settable_attr('show_vector', False, bool,
                           'Option to vector KDE for each stage')

    self.new_settable_attr('xmin', None, float, 'x-min')
    self.new_settable_attr('xmax', None, float, 'x-max')
    self.new_settable_attr('ymin', None, float, 'y-min')
    self.new_settable_attr('ymax', None, float, 'y-max')
    self.new_settable_attr('scatter_alpha', 0.5, float, 'scatter_alpha')

    self.new_settable_attr('margin', 0.2, float, 'margin')
    self.new_settable_attr('pm', 0.02, float, 'Percentile margin for kde plot')
    self.new_settable_attr(
      'iw', False, bool, 'Option to ignore wake for axis limits')
    self.new_settable_attr(
      'io', False, bool, 'Option to ignore ourliers for axis limits')

  # region: Properties


  # endregion: Properties

  # region: Plotting Methods

  def plot(self, ax: plt.Axes, fig: plt.Figure):
    """
    res_dict = {<bm_key>: {'W': array_w, 'N1': array_n1, ...}, ...}
    """
    res_dict: dict = self.pictor.selected_pair

    assert len(res_dict) == 2

    colors = {           # see https://matplotlib.org/stable/gallery/color/named_colors.html
      'W': 'forestgreen', 'N1': 'gold', 'N2': 'orange', 'N3': 'royalblue',
      'R': 'lightcoral'
    }

    # (1) Plot scatter/KDE/vector of each stage
    bm1_key, bm2_key = list(res_dict.keys())
    X, Y = None, None
    for stage_key, color in colors.items():
      if stage_key not in res_dict[bm1_key]: continue
      data1, data2 = res_dict[bm1_key][stage_key], res_dict[bm2_key][stage_key]

      if len(data1) < 2: continue

      # Convert data2 to micro-volt TODO
      # data2 = data2 * 1e6

      if self.get('show_scatter'):
        alpha = self.get('scatter_alpha')
        ax.scatter(data1, data2, c=color, label=stage_key, alpha=alpha)

      # show region if required
      # if self.get('show_rect'): self.show_bounds(ax, data1, data2, color)

      # show gauss is required
      if self.get('show_kde'): self.show_kde(ax, data1, data2, color)

      # show vector is required
      if self.get('show_vector'): self.show_vector(ax, data1, data2, color)

      # Gather data, note that data[12].shape.__len__ == 1
      if self.get('iw'):
        if stage_key == 'W': continue
      if X is None: X, Y = data1, data2
      else: X, Y = np.concatenate([X, data1]), np.concatenate([Y, data2])

    # (2) Set title, axis labels, and legend
    clouds_label = self.pictor.selected_clouds
    channel_label = self.pictor.selected_channel
    ax.set_title(f'{clouds_label} ({channel_label})')

    ax.set_xlabel(self.pictor.x_key)

    # (2.1) Set limits
    d = 100 * self.get('pm') / 2
    q1, q2 = d, 100 - d

    # Remove outliers if required
    if self.get('io'): X, Y = remove_outliers_for_list(X, Y)

    xmin, xmax = np.percentile(X, q1), np.percentile(X, q2)
    ymin, ymax = np.percentile(Y, q1), np.percentile(Y, q2)

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

    # (2.2) MISC
    ax.set_ylabel(self.pictor.y_key)
    ax.legend()

  def show_vector(self, ax: plt.Axes, m1, m2, color):
    mu1, mu2 = np.mean(m1), np.mean(m2)
    # Calculate covariance matrix
    cov = np.cov(m1, m2)
    assert cov[0, 1] == cov[1, 0]
    k = cov[0, 1] / cov[0, 0]
    x1, y1 = mu1, mu2
    step = np.sqrt(cov[0, 0])
    x2, y2 = mu1 + step, mu2 + step * k

    ax.plot(x1, y1, 's', color=color)
    ax.plot([x1, x2], [y1, y2], '-', color=color)

  def show_kde(self, ax: plt.Axes, m1, m2, color):
    from scipy import stats

    m1, m2 = remove_outliers_for_list(m1, m2, alpha=1.5)

    d = 100 * self.get('pm') / 2
    q1, q2 = d, 100 - d

    xmin, xmax = np.percentile(m1, q1), np.percentile(m1, q2)
    ymin, ymax = np.percentile(m2, q1), np.percentile(m2, q2)

    # Set margin
    m = self.get('margin')
    xm, ym = (xmax - xmin) * m, (ymax - ymin) * m
    xmin, xmax = xmin - xm, xmax + xm
    ymin, ymax = ymin - ym, ymax + ym

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    ax.contour(X, Y, Z, colors=color)

  # endregion: Plotting Methods

  # region: Private Methods

  # endregion: Private Methods

  # region: Shortcuts

  def register_shortcuts(self):
    self.register_a_shortcut('s', lambda: self.flip('show_scatter'),
                             'Toggle `show_scatter`')
    self.register_a_shortcut('r', lambda: self.flip('show_rect'),
                             'Toggle `show_rect`')
    self.register_a_shortcut('g', lambda: self.flip('show_kde'),
                             'Toggle `show_kde`')
    self.register_a_shortcut('v', lambda: self.flip('show_vector'),
                             'Toggle `show_vector`')
    self.register_a_shortcut('w', lambda: self.flip('iw'), 'Toggle `iw`')
    self.register_a_shortcut('o', lambda: self.flip('io'), 'Toggle `io`')

    self.register_a_shortcut('Left', lambda: self.set_lim('xmin'), 'Set xmin')
    self.register_a_shortcut('Right', lambda: self.set_lim('xmax'), 'Set xmax')
    self.register_a_shortcut('Down', lambda: self.set_lim('ymin'), 'Set ymin')
    self.register_a_shortcut('Up', lambda: self.set_lim('ymax'), 'Set ymax')

  # endregion: Shortcuts
