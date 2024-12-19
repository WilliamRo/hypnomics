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
from hypnomics.freud.file_manager import FileManager
from hypnomics.freud.freud import Freud
from hypnomics.freud.nebula import Nebula
from matplotlib.gridspec import GridSpec
from pictor.objects.signals.signal_group import SignalGroup, Annotation
from pictor.xomics.misc.distribution import remove_outliers_for_list
from roma import console, io
from roma import Nomear
from scipy import stats

import matplotlib.pyplot as plt
import numpy as np
import os



class HypnoStudio(Nomear):
  """This class is for generating figures of hypnoprints."""

  STAGE_COLORS = {
    # see https://matplotlib.org/stable/gallery/color/named_colors.html
    'W': 'forestgreen', 'N1': 'gold', 'N2': 'orange', 'N3': 'royalblue',
    'R': 'lightcoral'
  }

  def __init__(self, work_dir, sg_dir, neb_dir, **kwargs):
    # Set directory paths
    if not os.path.exists(sg_dir):
      raise FileNotFoundError(f'`{sg_dir}` does not exist.')
    self.sg_dir = sg_dir

    if not os.path.exists(neb_dir):
      raise FileNotFoundError(f'`{neb_dir}` does not exist.')
    self.neb_dir = neb_dir

    if not os.path.exists(work_dir): os.mkdir(work_dir)
    console.show_status(f'work directory set to `{os.path.abspath(work_dir)}`',
                        prompt='[HypnoStudio]')

    # Initialize a file manager
    self.file_manager = FileManager(work_dir=neb_dir)

    # Set kwargs
    self.kwargs = kwargs

  # region: Properties

  # endregion: Properties

  # region: Public Methods

  def take_one_photo(self, sg_path, neb_path, channels, time_resolution,
                     pk_1, pk_2=None, show_figure=False, fig_size=(12, 6),
                     **kwargs) -> plt.Figure:
    """Take a photo of a given signal group file."""
    # (1) Load files, get psg_label
    assert os.path.exists(sg_path) and os.path.exists(neb_path)
    sg: SignalGroup = io.load_file(sg_path, verbose=True)
    psg_label = sg.label

    probe_keys = [pk_1]
    if pk_2 is not None: probe_keys.append(pk_2)

    # (2) Create figure
    fig: plt.Figure = plt.figure(figsize=fig_size)

    # (2.1) Set figure layout
    n_channels = len(channels)
    n_cols = kwargs.get('n_cols', min(3, n_channels))
    hg_ratio = kwargs.get('hypnogram_ratio', 0.2)
    axes, ax_hypnogram = self.make_layout(fig, n_channels, n_cols, hg_ratio)

    # (3) Plot hypnofingerprints
    # (3.1) Generate nebula
    freud = Freud(self.neb_dir)
    nebula = freud.load_nebula(sg_labels=[psg_label],
                               channels=channels,
                               time_resolution=time_resolution,
                               probe_keys=probe_keys)

    # (3.2) Plot distribution
    self.plot_distribution(axes, nebula, psg_label, channels, probe_keys,
                           **self.kwargs)

    # (4) Plot hypnogram
    self._plot_hypnogram(ax_hypnogram, sg)

    # photo_filename = self.get_photo_filename(psg_label, pk_1, pk_2)
    # (9) ...
    # (9.1) Set title
    properties = kwargs.get('properties', {})
    prop_str = ', '.join([f'{k}: {v}' for k, v in properties.items()])
    fig.suptitle(f'{psg_label} ({prop_str})')

    # (9.2) Finalize
    fig.tight_layout()

    if show_figure: plt.show()
    return fig

  def get_photo_filename(self, psg_label, pk_1, pk_2=None):
    """Get the photo filename of a given signal group file."""
    assert isinstance(pk_1, str)
    pk_list = [_pk for _pk in (pk_1, pk_2) if isinstance(_pk, str)]
    pk_str = ','.join(pk_list)
    return f'{psg_label}({pk_str})'

  # endregion: Public Methods

  # region: Private Methods

  def _plot_joint_distribution(self, ax: plt.Axes):
    pass

  @classmethod
  def make_layout(cls, fig: plt.Figure, n_channels, n_cols, hg_ratio=0.):
    n_rows = n_channels // n_cols

    # Calculate hypnogram ratio
    height_ratios = [(1 - hg_ratio) / n_rows] * n_rows + [hg_ratio]
    gs = GridSpec(n_rows + 1, n_cols, figure=fig, height_ratios=height_ratios)
    axes = []
    for row in range(n_rows):
      for col in range(n_cols):
        axes.append(fig.add_subplot(gs[row, col]))

    if hg_ratio > 0:
      # Create a new subplot that spans the entire last row
      ax_hypnogram = fig.add_subplot(gs[n_rows, :])
      return axes, ax_hypnogram

    return axes

  @classmethod
  def plot_distribution(cls, axes: list[plt.Axes], nebula: Nebula, psg_label,
                        channels, probe_keys, **kwargs):
    # (0) Sanity check
    assert len(axes) == len(channels)

    # (1) Plot (joint-)distribution for each channel
    # (1.1) Get global range
    if kwargs.get('align_to_galaxy', True):
      ref_centers = [nebula.get_default_ref_center(psg_label, pk)
                     for pk in probe_keys]
      ranges = [[v + rc for v in nebula.galaxy_borders[pk]]
                for pk, rc in zip(probe_keys, ref_centers)]
    else:
      ranges = [cls._get_range(nebula, psg_label, channels, pk)
                for pk in probe_keys]

    # Apply padding
    pad = kwargs.get('pad', 0.2)
    ranges = [(v_min - pad * (v_max - v_min), v_max + pad * (v_max - v_min))
              for v_min, v_max in ranges]

    for ax, ck in zip(axes, channels):
      # (1.2) Get data
      clouds = [nebula.data_dict[(psg_label, ck, pk)] for pk in probe_keys]

      # (1.3) Plot clouds
      if len(clouds) == 2:
        cls._plot_kde_2D(ax, ck, clouds[0], clouds[1],
                         ranges[0], ranges[1], *probe_keys, **kwargs)
      else:
        cls._plot_kde_1D(ax, clouds[0], ranges[0])

    # ~ return buffer dict

  @classmethod
  def _plot_kde_2D(cls, ax: plt.Axes, channel, cloud_1: dict, cloud_2: dict,
                   xrange, yrange, pk1, pk2, **kwargs):
    # ~ Get buffer dict
    buffer_dict = kwargs.get('buffer', {})

    # (1) Plot KDE for each stage
    xmin, xmax = xrange
    ymin, ymax = yrange
    for sk, color in cls.STAGE_COLORS.items():
      if sk not in cloud_1 or len(cloud_1[sk]) < 5: continue
      x, y = cloud_1[sk], cloud_2[sk]

      # ~ Remove outliers if required
      alpha = kwargs.get('outlier_coef', 0)
      if alpha > 0:
        x, y = remove_outliers_for_list(x, y, alpha=alpha)

      # (1.*) Plot scatter
      if kwargs.get('plot_scatter', False):
        ax.scatter(x, y, color=color, alpha=0.2)
        continue

      # (1.1) Plot KDE contour
      # TODO: remove outliers?
      X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
      positions = np.vstack([X.ravel(), Y.ravel()])

      # ~ Try to get Z from buffer dict
      _key = (channel, sk)
      if _key not in buffer_dict:
        kernel = stats.gaussian_kde(np.vstack([x, y]))
        Z = np.reshape(kernel(positions).T, X.shape)
        buffer_dict[_key] = Z
      else:
        Z = buffer_dict[_key]

      # (1.1.1) Determine levels
      levels = np.linspace(Z.min(), Z.max(), 8)[1:]

      # (1.1.2) Make contour for wake stage transparent
      alpha = 0.2 if sk == 'W' else 1.0
      # ~ ax can be None (for preloading)
      if ax: ax.contour(X, Y, Z, colors=color, levels=levels, alpha=alpha)

    if ax is None: return
    # (2) Set axes styles
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_xlabel(pk1), ax.set_ylabel(pk2)
    if 'EEG' in channel: channel = channel.split(' ')[1]
    ax.set_title(channel)
    ax.set_xlim(xmin, xmax), ax.set_ylim(ymin, ymax)

  def _plot_kde_1D(self, ax: plt.Axes, cloud: dict):
    pass

  @classmethod
  def _get_range(cls, nebula: Nebula, lb, channels: list, pk, pad=0.2):
    values = []
    for ck in channels:
      cloud = nebula.data_dict[(lb, ck, pk)]
      non_wake_values = [cloud[sk] for sk in ('N1', 'N2', 'N3', 'R')]
      values.extend(np.concatenate(non_wake_values))

    # Get range within percentile edge
    pe = 1
    v_min, v_max = np.percentile(values, pe), np.percentile(values, 100 - pe)
    v_range = v_max - v_min
    return v_min - pad * v_range, v_max + pad * v_range

  def _plot_hypnogram(self, ax: plt.Axes, sg: SignalGroup):
    # (1) Extract ticks and stages from annotation
    annotation: Annotation = sg.annotations['stage Ground-Truth']
    ticks, stages = annotation.curve
    ticks = ticks / 3600

    # (2) Plot hypnogram
    # (2.1) Plot background
    for i, sk in enumerate(('W', 'N1', 'N2', 'N3', 'R')):
      color = self.STAGE_COLORS[sk]
      r = 0.45
      ax.axhspan(i - r, i + r, color=color, alpha=0.2)

    # (2.2) Plot stages
    N = len(ticks)
    for i in range(N - 1):
      t1, t2 = ticks[i], ticks[i + 1]
      s1, s2 = stages[i], stages[i + 1]
      if s1 > 4: continue

      line_style = '-'
      if s2 > 4 and i + 3 < N:
        line_style = ':'
        t2, s2 = ticks[i + 3], stages[i + 3]

      ax.plot([t1, t2], [s1, s2], line_style, color='black', alpha=0.8)

    # (3) Set styles
    ax.set_xlim(ticks[0], ticks[-1])
    ax.set_ylim(-0.5, 4.5)
    ax.set_xlabel('Time (hour)')
    ax.set_yticks([0, 1, 2, 3, 4], ['W', 'N1', 'N2', 'N3', 'R'])
    ax.invert_yaxis()

  # endregion: Private Methods
