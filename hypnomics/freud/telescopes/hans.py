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
from hypnomics.freud.nebula import Nebula
from pictor.plotters.plotter_base import Plotter
from pictor.xomics.misc.distribution import remove_outliers_for_list
from roma import console

import matplotlib.pyplot as plt
import numpy as np



class Hans(Plotter):

  STAGE_COLORS = {
    # see https://matplotlib.org/stable/gallery/color/named_colors.html
    'W': 'forestgreen', 'N1': 'gold', 'N2': 'orange', 'N3': 'royalblue',
    'R': 'lightcoral'
  }

  def __init__(self, pictor):
    # Call parent's constructor
    super().__init__(self.plot, pictor)

    self.new_settable_attr('show_scatter', True, bool,
                           'Option to show scatter for each stage')
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

    self.new_settable_attr('conditional', True, bool, 'Conditional')

    self.new_settable_attr(
      'iw', False, bool, 'Option to ignore wake for axis limits')
    self.new_settable_attr(
      'io', True, bool, 'Option to ignore outliers for axis limits')

  # region: Properties

  @property
  def nebula(self) -> Nebula: return self.pictor.nebula

  @property
  def selected_clouds(self) -> str: return self.pictor.selected_clouds

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
    res_dict = {<bm_key>: {'W': array_w, 'N1': array_n1, ...}, ...}
    """
    # (0) Get selected data pair
    res_dict: dict = self.pictor.selected_cluster_dict
    # assert len(res_dict) == 2

    # (1) Plot scatter/KDE/vector of each stage
    x_key, y_key = self.pictor.x_key, self.pictor.y_key
    X_all, Y_all = None, None
    stage_colors = self.STAGE_COLORS

    # (1.1) Merge stages in non-conditional case
    if not self.get('conditional'):
      stage_colors = {'All': 'grey'}
      res_dict = {
        x_key: {'All': np.concatenate(
          [res_dict[x_key][sk] for sk in self.STAGE_COLORS.keys()])},
        y_key: {'All': np.concatenate(
          [res_dict[y_key][sk] for sk in self.STAGE_COLORS.keys()])}
      }

    for stage_key, color in stage_colors.items():
      if stage_key not in res_dict[x_key]: continue
      Xs, Ys = res_dict[x_key][stage_key], res_dict[y_key][stage_key]

      if len(Xs) < 2: continue

      # (1.1) Plot scatter if required
      label = stage_key
      if self.get('show_scatter'):
        alpha = self.get('scatter_alpha')
        ax.scatter(Xs, Ys, c=color, label=label, alpha=alpha)
        label = None

      # (1.2) Plot KDE if required
      if self.get('show_kde'):
        self.show_kde(ax, Xs, Ys, color, label, stage_key)
        label = None

      # (1.3) Plot vector if required
      if self.get('show_vector'):
        self.show_vector(ax, Xs, Ys, color, label)

      # (1.4) Gather data, note that data[12].shape.__len__ == 1
      if self.get('iw'):
        # Ignore wake stage if required
        if stage_key == 'W': continue

      if X_all is None:
        X_all, Y_all = Xs, Ys
      else:
        X_all, Y_all = np.concatenate([X_all, Xs]), np.concatenate([Y_all, Ys])

    # (2) Set title, axis labels, and legend
    clouds_label = self.selected_clouds
    channel_label = self.selected_channel
    ax.set_title(f'{clouds_label} ({channel_label}){self.pictor.meta_suffix}')

    ax.set_xlabel(self.pictor.x_key)
    ax.set_ylabel(self.pictor.y_key)

    ax.legend()

    # (3) Set limits

    # Remove outliers if required
    if self.get('io'): X_all, Y_all = remove_outliers_for_list(X_all, Y_all)

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

  def show_vector(self, ax: plt.Axes, m1, m2, color, label):
    mu1, mu2 = np.mean(m1), np.mean(m2)
    # (1) Calculate covariance matrix
    cov = np.cov(m1, m2)
    assert cov[0, 1] == cov[1, 0]
    k = cov[0, 1] / cov[0, 0]
    x1, y1 = mu1, mu2
    step = np.sqrt(cov[0, 0])
    x2, y2 = mu1 + step, mu2 + step * k

    # (2) Plot cluster center
    if label == 'W':
      ax.plot(x1, y1, 's', color=color)
    else:
      proportion = len(m1) / self.nebula.get_epoch_total(
        self.selected_clouds, excludes='W')

      alpha = 0.5
      total_size = 500
      ax.scatter([x1], [y1], s=total_size, c='none', marker='o',
                 edgecolors='grey', alpha=alpha)
      ax.scatter([x1], [y1], s=proportion * total_size, color=color, alpha=1.0)

    # (3) Plot covariance direction
    ax.plot([x1, x2], [y1, y2], '-', color=color, label=label)

  def show_kde(self, ax: plt.Axes, m1, m2, color, label, stage_key):
    if len(m1) == 0: return

    # (1) Calculate KDE
    m = self.get('margin')
    kde_key = '::'.join(['HANS_KDE', self.selected_clouds,
                         self.selected_channel, stage_key, str(m)])
    if not self.nebula.in_pocket(kde_key):
      X, Y, Z = self.calc_kde(m1, m2, m)
      self.nebula.put_into_pocket(kde_key, (X, Y, Z), local=True)
    else:
      X, Y, Z = self.nebula.get_from_pocket(kde_key)

    # (2) Plot contour
    ax.contour(X, Y, Z, colors=color)

    # (-1) Prepare label if necessary
    if label is not None:
      mu1, mu2 = np.mean(m1), np.mean(m2)
      ax.plot([mu1, mu1], [mu2, mu2], '-', color=color, label=label)

  @staticmethod
  def calc_kde(m1, m2, margin):
    from scipy import stats

    # (1) Remove outliers further than 1.5 * IQR(25, 75)
    m1, m2 = remove_outliers_for_list(m1, m2, alpha=1.5)

    xmin, xmax = np.min(m1), np.max(m1)
    ymin, ymax = np.min(m2), np.max(m2)

    # (2) Set margin
    m = margin
    xm, ym = (xmax - xmin) * m, (ymax - ymin) * m
    xmin, xmax = xmin - xm, xmax + xm
    ymin, ymax = ymin - ym, ymax + ym

    # (3) Get KDE
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    return X, Y, Z

  def calculate_all_kde(self):
    N = len(self.nebula.labels)
    console.show_status(f'Calculating KDE for {N} nights ...')

    m = self.get('margin')

    for i, label in enumerate(self.nebula.labels):
      console.print_progress(i, N)
      for ck in self.nebula.channels:
        for sk in self.STAGE_COLORS.keys():
          kde_key = '::'.join(['HANS_KDE', label, ck, sk, str(m)])
          if self.nebula.in_pocket(kde_key): continue

          m1 = self.nebula.data_dict[(label, ck, self.x_key)][sk]
          m2 = self.nebula.data_dict[(label, ck, self.y_key)][sk]

          try:
            X, Y, Z = self.calc_kde(m1, m2, m)
            self.nebula.put_into_pocket(kde_key, (X, Y, Z), local=True)
          except:
            console.warning(f'Failed to calculate KDE for {label}/{sk}.')
            continue

    console.show_status(f'Calculated KDE for {N} nights.')
  cak = calculate_all_kde

  # endregion: Plotting Methods

  # region: Private Methods

  # endregion: Private Methods

  # region: Shortcuts

  def register_shortcuts(self):
    self.register_a_shortcut('s', lambda: self.flip('show_scatter'),
                             'Toggle `show_scatter`')
    self.register_a_shortcut('g', lambda: self.flip('show_kde'),
                             'Toggle `show_kde`')
    self.register_a_shortcut('v', lambda: self.flip('show_vector'),
                             'Toggle `show_vector`')
    self.register_a_shortcut('w', lambda: self.flip('iw'), 'Toggle `iw`')
    self.register_a_shortcut('o', lambda: self.flip('io'), 'Toggle `io`')
    self.register_a_shortcut('c', lambda: self.flip('conditional'), 'Toggle `conditional`')

    self.register_a_shortcut('Left', lambda: self.set_lim('xmin'), 'Set xmin')
    self.register_a_shortcut('Right', lambda: self.set_lim('xmax'), 'Set xmax')
    self.register_a_shortcut('Down', lambda: self.set_lim('ymin'), 'Set ymin')
    self.register_a_shortcut('Up', lambda: self.set_lim('ymax'), 'Set ymax')

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
