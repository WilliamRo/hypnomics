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
import numpy as np

from hypnomics.freud.file_manager import FileManager
from hypnomics.freud.freud import Freud
from hypnomics.freud.nebula import Nebula
from matplotlib.gridspec import GridSpec
from pictor.objects.signals.signal_group import SignalGroup, Annotation
from roma import console, io
from roma import Nomear

import matplotlib.pyplot as plt
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
    console.show_status(f'HypnoStudio created, '
                        f'work directory set to `{os.path.abspath(work_dir)}`')

    # Initialize a file manager
    self.file_manager = FileManager(work_dir=neb_dir)

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
    n_rows = n_channels // n_cols

    hg_ratio = kwargs.get('hypnogram_ratio', 0.2)
    height_ratios = [(1 - hg_ratio) / n_rows] * n_rows + [hg_ratio]
    gs = GridSpec(n_rows + 1, n_cols, figure=fig, height_ratios=height_ratios)
    axes = []
    for row in range(n_rows):
      for col in range(n_cols):
        axes.append(fig.add_subplot(gs[row, col]))

    # (2.2) Create a new subplot that spans the entire last row
    ax_hypnogram = fig.add_subplot(gs[n_rows, :])

    # (3) Plot hypnofingerprints
    # (3.1) Generate nebula
    freud = Freud(self.neb_dir)
    nebula = freud.load_nebula(sg_labels=[psg_label],
                               channels=channels,
                               time_resolution=time_resolution,
                               probe_keys=probe_keys)

    # (3.2) Plot distribution
    self._plot_distribution(axes, nebula, channels, probe_keys)

    # (4) Plot hypnogram
    self._plot_hypnogram(ax_hypnogram, sg, line_width=10)

    # photo_filename = self.get_photo_filename(psg_label, pk_1, pk_2)
    # (9) ...
    # (9.1) Set title
    properties = kwargs.get('properties', {})
    prop_str = ', '.join([f'{k}: {v}' for k, v in properties.items()])
    pk_str = 'x'.join(probe_keys)
    fig.suptitle(f'{psg_label}({prop_str}) | {pk_str}')

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

  def _plot_distribution(self, axes: list[plt.Axes], nebula: Nebula, channels,
                         probe_keys):
    # (0) Sanity check
    assert len(axes) == len(channels)
    assert len(nebula.labels) == 1

    # (1) Plot (joint-)distribution for each channel
    for ax, ck in zip(axes, channels):
      # (1.1) Get data
      clouds = [nebula.data_dict[(nebula.labels[0], ck, pk)] for pk in probe_keys]

      # (1.2) Plot clouds
      if len(clouds) == 2: self._plot_kde_2D(ax, ck, *clouds)
      else: self._plot_kde_1D(ax, clouds[0])

  def _plot_kde_2D(self, ax: plt.Axes, channel, cloud_1: dict, cloud_2: dict):
    # (1) Plot KDE for each stage
    for sk, color in self.STAGE_COLORS.items():
      if sk not in cloud_1: continue
      x1, x2 = cloud_1[sk], cloud_2[sk]
      ax.scatter(x1, x2, color=color, alpha=0.2)

    # (2) Set axes styles
    ax.set_xticks([]), ax.set_yticks([])
    if 'EEG' in channel: channel = channel.split(' ')[1]
    ax.set_title(channel)

  def _plot_kde_1D(self, ax: plt.Axes, cloud: dict):
    pass

  def _plot_hypnogram(self, ax: plt.Axes, sg: SignalGroup, line_width=20):
    # (1) Extract ticks and stages from annotation
    annotation: Annotation = sg.annotations['stage Ground-Truth']
    ticks, stages = annotation.curve
    ticks = ticks / 3600

    # (2) Plot hypnogram
    # (2.1) Plot background
    colors = ['forestgreen', 'gold', 'orange', 'royalblue', 'lightcoral']
    for i, c in enumerate(colors):
      ax.plot([ticks[0], ticks[-1]], [i, i], color=c,
              linewidth=line_width, alpha=0.2)

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
