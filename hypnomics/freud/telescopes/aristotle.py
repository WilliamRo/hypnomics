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
# ==-=========================================================================-=
from hypnomics.freud.nebula import Nebula
from pictor.plotters.plotter_base import Plotter
from pictor.xomics.misc.distribution import remove_outliers_for_list
from roma import console

import matplotlib.pyplot as plt
import numpy as np



class Aristotle(Plotter):

  STAGE_COLORS = {
    # see https://matplotlib.org/stable/gallery/color/named_colors.html
    'W': 'forestgreen', 'N1': 'gold', 'N2': 'orange', 'N3': 'royalblue',
    'R': 'lightcoral'
  }

  def __init__(self, pictor):
    # Call parent's constructor
    super().__init__(self.plot, pictor)

    self.new_settable_attr('show_kde', False, bool,
                           'Option to show KDE for each stage')

    self.new_settable_attr('xmin', None, float, 'x-min')
    self.new_settable_attr('xmax', None, float, 'x-max')
    self.new_settable_attr('ymax', None, float, 'y-max')

    self.new_settable_attr('mod', True, bool, 'Option to apply modifier')
    self.new_settable_attr('dev', False, bool, 'Option to toggle developer mode')

    self.new_settable_attr(
      'show_all', False, bool, 'Option to show overall distribution')
    self.new_settable_attr(
      'iw', False, bool, 'Option to ignore wake for axis limits')
    self.new_settable_attr(
      'io', True, bool, 'Option to ignore ourliers for axis limits')

    self.new_settable_attr('margin', 0.2, float, 'margin')

  # region: Properties

  @property
  def nebula(self) -> Nebula: return self.pictor.nebula

  @property
  def selected_clouds(self) -> str: return self.pictor.selected_clouds

  @property
  def selected_channel(self) -> str: return self.pictor.selected_channel

  @property
  def selected_probe(self) -> str: return self.pictor.selected_probe

  # endregion: Properties

  # region: Plotting Methods

  def plot(self, ax: plt.Axes, fig: plt.Figure, linestyle='-', cloud_key=None):
    """
    res_dict = {'W': array_w, 'N1': array_n1, ...}
    """
    # TODO: dev option
    in_comparison = cloud_key is not None
    if not in_comparison: cloud_key = self.selected_clouds

    res_dict = self.pictor.get_stage_dict(cloud_key)

    total_n = sum([len(res_dict[k]) for k in res_dict.keys() if k != 'W'])

    X_all, Y_all = None, None
    for stage_key, color in self.STAGE_COLORS.items():
      # Ignore wake stage if required
      if self.get('iw') and stage_key == 'W': continue

      # Get Xs
      if stage_key not in res_dict: continue
      Xs = res_dict[stage_key]
      if len(Xs) < 2: continue

      # Plot Xs
      if self.get('show_kde'):
        modifier = 1.0 if stage_key == 'W' else len(Xs) / total_n
        self.show_kde(ax, Xs, color, stage_key, modifier=modifier,
                      linestyle=linestyle)
      else: self.show_histogram(ax, Xs, color, stage_key)

      # Gather Xs
      if X_all is None: X_all = Xs
      else: X_all = np.concatenate([X_all, Xs])

    # Show overall distribution if necessary
    if self.get('show_all'):
      color = 'grey'
      stage_key = 'All'
      if self.get('show_kde'): self.show_kde(
        ax, X_all, color, stage_key, linestyle=linestyle)
      else: self.show_histogram(ax, X_all, color, stage_key)

    # TODO: dev
    if in_comparison: return

    title_suffix = ''
    # TODO: dev
    if self.get('dev'):
      # Plot next object
      CK = self.pictor.Keys.OBJECTS
      current_index = self.pictor.cursors[CK]
      next_index = (current_index + 1) % len(self.pictor.axes[CK])
      next_cloud_key = self.pictor.axes[CK][next_index]

      # Distinguish same/different individual
      same_individual = self.selected_clouds[:5] == next_cloud_key[:5]

      # Calculate TV distance (auto shifted)
      from hypnomics.hypnoprints.stat_models.model_1 import HypnoModel1
      hm = HypnoModel1()
      next_dict = self.pictor.get_stage_dict(next_cloud_key)
      d = hm.calc_distance(res_dict, next_dict)

      linestyle = '--' if same_individual else ':'
      self.plot(ax, fig, linestyle=linestyle, cloud_key=next_cloud_key)

      if same_individual: title_suffix = f' (Same, d = {d:.2f})'
      else: title_suffix = f' (Not same, d = {d:.2f})'

    # (2) Set title, axis labels, and legend
    clouds_label = self.selected_clouds
    channel_label = self.selected_channel
    ax.set_xlabel(self.selected_probe)
    ax.legend()

    title = f'{clouds_label} ({channel_label}){title_suffix}'
    ax.set_title(title + self.pictor.meta_suffix)

    # Set axis limits
    global_ymax = self.get('ymax')
    if global_ymax is not None: ax.set_ylim([0, global_ymax])

  def show_histogram(self, ax: plt.Axes, m, color, stage_key):
    if len(m) == 0: return

    # Show histogram
    alpha = 0.5
    n_bins = 50
    ax.hist(m, bins=n_bins, color=color, alpha=alpha, label=stage_key)


  def show_kde(self, ax: plt.Axes, values, color, stage_key,
               modifier: float = 1.0, linestyle='-'):
    if len(values) == 0: return
    m = self.get('margin')
    X, Y = self.calc_kde_1d(values, margin=m)
    if self.get('mod'): Y *= modifier

    # Plot KDE
    alpha = 0.8
    if stage_key == 'W': alpha = 0.2
    ax.plot(X, Y, color=color, label=stage_key, alpha=alpha,
            linewidth=3, linestyle=linestyle)

    # Update y_max
    global_y_max = ax.get_ylim()[1]
    this_y_max = np.max(Y) * 1.1
    if this_y_max > global_y_max: ax.set_ylim([0, this_y_max])


  def calc_kde_1d(self, x, margin):
    from scipy import stats

    # (1) Remove outliers further than 1.5 * IQR(25, 75)
    if self.get('io'): x = remove_outliers_for_list(x, alpha=1.5)

    xmin, xmax = np.min(x), np.max(x)

    # (2) Set margin
    xm = (xmax - xmin) * margin
    xmin, xmax = xmin - xm, xmax + xm

    # (3) Get KDE
    kde = stats.gaussian_kde(x)
    X = np.linspace(xmin, xmax, 100)
    Y = kde(X)
    return X, Y

  # endregion: Plotting Methods

  # region: Private Methods

  # endregion: Private Methods

  # region: Shortcuts

  def register_shortcuts(self):
    self.register_a_shortcut('g', lambda: self.flip('show_kde'),
                             'Toggle `show_kde`')
    self.register_a_shortcut('w', lambda: self.flip('iw'), 'Toggle `iw`')
    self.register_a_shortcut('o', lambda: self.flip('io'), 'Toggle `io`')
    self.register_a_shortcut('a', lambda: self.flip('show_all'),
                             'Toggle `show_all`')
    self.register_a_shortcut('m', lambda: self.flip('mod'), 'Toggle `mod`')
    self.register_a_shortcut('d', lambda: self.flip('dev'), 'Toggle `dev`')

    self.register_a_shortcut('Left', lambda: self.set_lim('xmin'), 'Set xmin')
    self.register_a_shortcut('Right', lambda: self.set_lim('xmax'), 'Set xmax')

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
