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
# ====-======================================================================-==
from hypnomics.freud.nebula import Nebula
from pictor.objects.signals.signal_group import SignalGroup
from roma import console
from roma import finder
from roma import io
from roma import Nomear

import os



class FileManager(Nomear):
  """Rule of organization for cloud files:
  ROOT -> PID -> Channel -> Time Resolution -> <feature_name>.clouds

  ROOT                                            Lv1: Working directory
   ├── PID_001                                    Lv2: Patient ID
   ├── PID_002
   │    ├── Channel_1                             Lv3: Channel name
   │    │    ├── 30s                              Lv4: Time resolution
   │    │    │    ├── AMP_128.clouds              Lv5: Feature name
   │    │    │    └── FREQ_20.clouds
   │    │    ├── 10s
   ⋮    ⋮    ⋮
   │    ├── Channel_2
   ⋮    ⋮
   ├── PID_003
   ⋮
  """

  def __init__(self, work_dir: str):
    self._check_path(work_dir, create=True)
    self.work_dir = work_dir

  # region: Private Methods

  def _check_path(self, path, create=True, return_false_if_not_exist=False):
    if not os.path.exists(path):
      if not create:
        if return_false_if_not_exist: return False
        else: raise FileNotFoundError(f"!! `{path}` not found !!")
      os.makedirs(path)
      console.show_status(f"Created `{path}`.")
    return True

  def _check_hierarchy(self, sg_label: str, channel=None, time_resolution=None,
                       feature_name=None, create_if_not_exist=True):
    sg_path = os.path.join(self.work_dir, sg_label)
    self._check_path(sg_path, create=create_if_not_exist)

    if channel is not None:
      channel_path = os.path.join(sg_path, channel)
      self._check_path(channel_path, create=create_if_not_exist)

      if time_resolution is not None:
        tr_path = os.path.join(channel_path, f"{time_resolution}s")
        self._check_path(tr_path, create=create_if_not_exist)

        if feature_name is not None:
          feature_path = os.path.join(tr_path, f"{feature_name}.clouds")
          b_exist = self._check_path(feature_path, create=False,
                                     return_false_if_not_exist=True)
          return feature_path, b_exist
        else: return tr_path
      else: return channel_path
    else: return sg_path

  def _get_signal_group_generator(self, sg_path: str, pattern: str,
                                  progress_bar=False):
    sg_file_list = finder.walk(sg_path, pattern=pattern)
    for i, file_path in enumerate(sg_file_list):
      if progress_bar: console.print_progress(i, len(sg_file_list))
      sg: SignalGroup = io.load_file(file_path, verbose=True)
      yield sg
      del sg

  # endregion: Private Methods

  # region: Public Methods

  def load_nebula(self, sg_labels: list, channels: list, time_resolution: int,
                  probe_keys: list, name='Nebula') -> Nebula:
    nebula = Nebula(time_resolution, name=name)

    for sg_label in sg_labels:
      nebula.labels.append(sg_label)
      for channel in channels:
        nebula.channels.append(channel)
        for probe_key in probe_keys:
          nebula.probe_keys.append(probe_key)
          clouds_path, b_exist = self._check_hierarchy(
            sg_label, channel=channel, time_resolution=time_resolution,
            feature_name=probe_key, create_if_not_exist=False)

          assert b_exist, f'`{clouds_path}` not found.'
          clouds = io.load_file(clouds_path)
          nebula.data_dict[(sg_label, channel, probe_key)] = clouds

    return nebula

  def get_sampling_frequency(self, sg_path: str, pattern: str, channels: list):
    """Get sampling frequency from the first signal group file in the
    given list."""
    sg_file_list = finder.walk(sg_path, pattern=pattern)
    sg: SignalGroup = io.load_file(sg_file_list[0], verbose=True)
    sg = sg.extract_channels(channels)
    return sg.digital_signals[0].sfreq

  # endregion: Public Methods
