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
from pictor.plotters.plotter_base import Plotter
from pictor.xomics.misc.distribution import remove_outliers_for_list
from roma import console

import matplotlib.pyplot as plt
import numpy as np



class Galileo(Plotter):

  STAGE_COLORS = {
    # see https://matplotlib.org/stable/gallery/color/named_colors.html
    'W': 'forestgreen', 'N1': 'gold', 'N2': 'orange', 'N3': 'royalblue',
    'R': 'lightcoral'
  }

  def __init__(self, pictor):
    # Call parent's constructor
    super().__init__(self.plot, pictor)

    self.new_settable_attr('show_scatter', False, bool,
                           'Option to show scatter for each stage')
    self.new_settable_attr('show_kde', False, bool,
                           'Option to show KDE for each stage')
    self.new_settable_attr('show_vector', True, bool,
                           'Option to vector KDE for each stage')

    self.new_settable_attr('xmax', 3.0, float, 'x-max')
    self.new_settable_attr('ymax', 160e-6, float, 'y-max')
    self.new_settable_attr('scatter_alpha', 0.5, float, 'scatter_alpha')

    self.new_settable_attr('margin', 0.2, float, 'margin')

    self.new_settable_attr(
      'iw', False, bool, 'Option to ignore wake for axis limits')
    self.new_settable_attr(
      'io', False, bool, 'Option to ignore ourliers for axis limits')

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
    res_dict: dict = self.pictor.selected_pair
    assert len(res_dict) == 2

    # (1) Plot scatter/KDE/vector of each stage
    x_key, y_key = list(res_dict.keys())
    X_all, Y_all = None, None

    # PATCH
    x0 = self.nebula.get_center(
      self.selected_clouds, self.nebula.channels[0], self.x_key, 'N2')
    y0 = self.nebula.get_center(
      self.selected_clouds, self.nebula.channels[0], self.y_key, 'N2')

    for stage_key, color in self.STAGE_COLORS.items():
      if stage_key not in res_dict[x_key]: continue

      # PATCH
      Xs, Ys = res_dict[x_key][stage_key], res_dict[y_key][stage_key]
      Xs, Ys = Xs - x0, Ys - y0

      if len(Xs) < 2: continue

      # (1.1) Plot scatter if required
      label = stage_key
      if self.get('show_scatter'):
        alpha = self.get('scatter_alpha')
        ax.scatter(Xs, Ys, c=color, label=label, alpha=alpha)
        label = None

      # (1.2) Plot KDE if required
      if self.get('show_kde'):
        self.show_kde(ax, Xs, Ys, color, label, stage_key, x0, y0)
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
    ax.set_title(f'{clouds_label} ({channel_label})')

    ax.set_xlabel(self.pictor.x_key)
    ax.set_ylabel(self.pictor.y_key)

    ax.legend()

    # (3) Set limits

    # Remove outliers if required
    if self.get('io'): X_all, Y_all = remove_outliers_for_list(X_all, Y_all)

    xmax, ymax = np.max(np.abs(X_all)), np.max(np.abs(Y_all))

    xmax = self.get('xmax') if self.get('xmax') is not None else xmax
    ymax = self.get('ymax') if self.get('ymax') is not None else ymax

    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-ymax, ymax)

  def show_vector(self, ax: plt.Axes, m1, m2, color, label):
    m1, m2 = remove_outliers_for_list(m1, m2, alpha=1.5)
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

  def show_kde(self, ax: plt.Axes, m1, m2, color, label, stage_key, x0, y0):
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
    X, Y = X - x0, Y - y0
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

    self.register_a_shortcut('Left', lambda: self.set_lim('xmin'), 'Set xmin')
    self.register_a_shortcut('Right', lambda: self.set_lim('xmax'), 'Set xmax')
    self.register_a_shortcut('Down', lambda: self.set_lim('ymin'), 'Set ymin')
    self.register_a_shortcut('Up', lambda: self.set_lim('ymax'), 'Set ymax')

  # endregion: Shortcuts
