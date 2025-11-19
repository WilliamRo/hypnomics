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
from hypnomics.hypnoprints.probes.probe_group import ProbeGroup
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

  def _check_hierarchy(self, sg_label: str, channel=None,
                       band_key=None, time_resolution=None,
                       feature_name=None, create_if_not_exist=True,
                       return_false_if_not_exist=False):
    sg_path = os.path.join(self.work_dir, sg_label)
    self._check_path(sg_path, create=create_if_not_exist,
                     return_false_if_not_exist=return_false_if_not_exist)

    if channel is not None:
      if isinstance(channel, (list, tuple)):
        # This will only happen in the case of (2) in `load_nebula`
        channel_paths = [os.path.join(sg_path, ch) for ch in channel]
        channel_paths = [cp for cp in channel_paths if os.path.exists(cp)]
        assert len(channel_paths) == 1, f'More than one channel path found: {channel_paths}'
        channel_path = channel_paths[0]
      else:
        channel_path = os.path.join(sg_path, channel)
        self._check_path(channel_path, create=create_if_not_exist,
                         return_false_if_not_exist=return_false_if_not_exist)

      if band_key is not None:
        band_path = os.path.join(channel_path, f'{band_key}.band')
        b_exist = self._check_path(
          band_path, create=False,
          return_false_if_not_exist=return_false_if_not_exist)
        return band_path, b_exist

      if time_resolution is not None:
        tr_path = os.path.join(channel_path, f"{time_resolution}s")
        self._check_path(tr_path, create=create_if_not_exist,
                         return_false_if_not_exist=return_false_if_not_exist)

        if feature_name is not None:
          feature_path = os.path.join(tr_path, f"{feature_name}.clouds")
          b_exist = self._check_path(feature_path, create=False,
                                     return_false_if_not_exist=True)
          return feature_path, b_exist
        else: return tr_path
      else: return channel_path
    else: return sg_path

  def _check_band_buffers(self, sg_path, psg_label, channels, bands):
    for ck in channels:
      if psg_label is None:
        assert sg_path is not None and sg_path[-1] != '/'
        sg_label = os.path.basename(sg_path)
        sg_label = sg_label.split('(')[0]
      else:
        sg_label = psg_label

      b_exist_list = [
        self._check_hierarchy(
          sg_label, channel=ck, band_key=bk, create_if_not_exist=False,
          return_false_if_not_exist=True)[1]
        for bk in bands]

      if not all(b_exist_list): return False
    return True

  def _check_cloud(self, sg_path, psg_label, channels, time_resolutions,
                   extractor_dict):
    for ck in channels:
      for tr in time_resolutions:
        for pk, probe in extractor_dict.items():
          if psg_label is None:
            assert sg_path is not None and sg_path[-1] != '/'
            sg_label = os.path.basename(sg_path)
            sg_label = sg_label.split('(')[0]
          else:
            sg_label = psg_label

          if isinstance(probe, ProbeGroup):
            pk_list = probe.probe_keys
          else:
            pk_list = [pk]

          b_exist_list = [
            self._check_hierarchy(
              sg_label, channel=ck, time_resolution=tr, feature_name=pk,
              create_if_not_exist=False, return_false_if_not_exist=True)[1]
            for pk in pk_list]

          if not all(b_exist_list): return False
    return True

  def _get_signal_group_generator(
      self, sg_path: str, pattern: str, progress_bar=False,
      sg_file_list=None, **kwargs):

    # Get sg_file_list
    if sg_file_list is None:
      _sg_file_list = finder.walk(sg_path, pattern=pattern)
    else:
      _sg_file_list = sg_file_list
      console.show_status(f'Using provided sg_file_list (N={len(sg_file_list)})'
                          f' for creating sg_generator ...')

    # ...
    N = kwargs.get('max_n_sg', None)
    if N is not None:
      _sg_file_list = _sg_file_list[:N]
      console.show_status(f'Selected first {N} signal group files ...')

    for i, file_path in enumerate(_sg_file_list):
      sg: SignalGroup = io.load_file(file_path, verbose=True)
      console.show_status(f'Signal group `{sg.label}` has been loaded.')

      if progress_bar: console.print_progress(i, len(_sg_file_list))
      yield sg
      # sg._cloud_pocket.clear()
      del sg

  # endregion: Private Methods

  # region: Public Methods

  def load_nebula(self, sg_labels: list, channels: list, time_resolution: int,
                  probe_keys: list, name='Nebula', verbose=False) -> Nebula:
    """Load a Nebula object from the given signal group labels, channels,
    time resolution, and probe keys.

    Args
    ----
    channels: list of channel names. Can be:
    (1) ['EEG Fpz-Oz', 'EEG Fz-Cz'];
    (2) [('EEG F3-REF', 'EEG F3-CLE', 'EEG F3-LER'),
         ('EEG F4-REF', 'EEG F4-CLE', 'EEG F4-LER')].
        In this case, each `channel` in the list is a tuple of channel names.
        The second to the last channels will be renamed to the first channel.
    """
    nebula = Nebula(time_resolution, name=name)

    if verbose: console.show_status('Loading nebula ...')
    for i, sg_label in enumerate(sg_labels):
      if verbose and i % 10 == 0: console.print_progress(i, len(sg_labels))
      nebula.labels.append(sg_label)

      for channel in channels:
        # Case (2)
        if isinstance(channel, (tuple, list)):
          channel, aliases = channel[0], channel[1:]
        else:
          assert isinstance(channel, str), f'Invalid channel: {channel}'
          channel, aliases = channel, [channel]

        # Add channel keys to Nebula if has not registered yet
        if channel not in nebula.channels:
          nebula.channels.append(channel)

        for probe_key in probe_keys:
          if probe_key not in nebula.probe_keys:
            nebula.probe_keys.append(probe_key)

          # Load clouds
          assert isinstance(aliases, (list, tuple))
          clouds_path, b_exist = self._check_hierarchy(
            sg_label, channel=aliases, time_resolution=time_resolution,
            feature_name=probe_key, create_if_not_exist=False)

          assert b_exist, f'`{clouds_path}` not found.'
          clouds = io.load_file(clouds_path)
          nebula.data_dict[(sg_label, channel, probe_key)] = clouds

    if verbose:
      console.show_status(f'Loaded nebula from {len(sg_labels)} PSG records.')

    return nebula

  def get_sampling_frequency(self, sg_path: str, pattern: str, channels: list):
    """Get sampling frequency from the first signal group file in the
    given list."""
    sg_file_list = finder.walk(sg_path, pattern=pattern)
    sg: SignalGroup = io.load_file(sg_file_list[0], verbose=True)
    sg = sg.extract_channels(channels)
    return sg.digital_signals[0].sfreq

  # endregion: Public Methods
