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
import os.path
from collections import OrderedDict
from hypnomics.hypnoprints.hp_extractor import STAGE_KEYS
from pictor.xomics.misc.distribution import remove_outliers
from roma import Nomear
from roma import io

import numpy as np



class Flow(Nomear):
  """Flow.data_dict =
    {(traj_label, channel, probe_key): [v_1, v_2, ..., v_T], ...}

    Flow.stages = {traj_label: [...]}

    Flow.meta = {traj_label: {meta_key: meta_value, ...}, ...}
  """

  # STAGE_KEYS = ('W', 'N1', 'N2', 'N3', 'R')
  STAGE_KEYS = STAGE_KEYS

  def __init__(self, time_resolution: int, name: str = 'Flow'):
    assert 30 % time_resolution == 0, "!! Time resolution should be a factor of 30 !!"
    self.time_resolution = time_resolution
    self.name = name

  # region: Properties

  # region: -META

  @property
  def delta(self): return self.time_resolution

  @Nomear.property(local=True)
  def labels(self): return []

  @Nomear.property(local=True)
  def stages(self): return OrderedDict()

  @Nomear.property(local=True)
  def meta(self): return OrderedDict()

  @Nomear.property(local=True)
  def channels(self): return []

  @Nomear.property(local=True)
  def probe_keys(self): return []

  @Nomear.property(local=True)
  def data_dict(self): return OrderedDict()

  # endregion: -META

  # region: -IO

  @staticmethod
  def load(path: str, verbose=True) -> 'Flow':
    assert path.endswith('.flow'), "!! Flow file should end with '.flow' !!"
    return io.load_file(path, verbose=verbose)

  def save(self, path: str, verbose=True):
    if not path.endswith('.flow'): path = f'{path}.flow'
    io.save_file(self, path, verbose=verbose)

  def export_to_csv(self, tgt_dir: str):
    """Export flow data to CSV files. File structure:
    tgt_dir/
      channel_1/
        sg_label_1.csv
        sg_label_2.csv
        ...
      channel_2/
        ...

    Each CSV file contains:
      Sheet 1: time, stage, probe_1, probe_2, ...
      Sheet 2: meta information
    """
    import pandas as pd

    # Check and create target directory
    os.makedirs(tgt_dir, exist_ok=True)

    # Iterate through channels
    for channel in self.channels:
      channel_dir = os.path.join(tgt_dir, channel)
      os.makedirs(channel_dir, exist_ok=True)

      # Iterate through trajectories
      for traj_label in self.labels:
        # Prepare data for DataFrame
        time_points = list(range(len(
          self.data_dict[(traj_label, channel, self.probe_keys[0])])))
        time_points = [tp * self.time_resolution for tp in time_points]

        stage_data = self.stages.get(traj_label, [5] * len(time_points))

        data = {
          'time': time_points,
          'stage': stage_data,
        }

        for probe_key in self.probe_keys:
          data[probe_key] = self.data_dict.get(
            (traj_label, channel, probe_key), [None] * len(time_points))

        df = pd.DataFrame(data)

        # Prepare meta information
        meta_info = self.meta.get(traj_label, {})
        meta_df = pd.DataFrame(list(meta_info.items()),
                               columns=['meta_key', 'meta_value'])
        # Drop 'name'
        meta_df = meta_df[meta_df['meta_key'] != 'name']

        # Write to Excel file with two sheets
        file_path = os.path.join(channel_dir, f'{traj_label}.xlsx')
        with pd.ExcelWriter(file_path) as writer:
          df.to_excel(writer, sheet_name='traj', index=False)
          meta_df.to_excel(writer, sheet_name='meta', index=False)

  # endregion: -IO

  # region: -Visualization

  def visualize2D(self, x_key='FREQ-20', y_key='AMP-1', viewer_class=None,
                  title=None, fig_size=(10, 6), viewer_configs=None,
                  **probe_1_configs):
    """Visualize flow in a 2-D space"""
    # Set default arguments
    if title is None: title = f'Delta={self.delta}'
    if viewer_configs is None: viewer_configs = {}

    # Use specified viewer
    viewer = viewer_class(nebula=self, x_key=x_key, y_key=y_key,
                          title=title, figure_size=fig_size, **viewer_configs)
    for k, v in probe_1_configs.items(): viewer.plotters[0].set(k, v)
    viewer.show()

  # endregion: -Visualization
