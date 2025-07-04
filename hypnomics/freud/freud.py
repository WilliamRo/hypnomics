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
"""Freud knows everything about hypnomics."""
from .file_manager import FileManager

from collections import OrderedDict

from hypnomics.hypnoprints.hp_extractor import DEFAULT_STAGE_KEY
from hypnomics.hypnoprints.hp_extractor import get_stage_map_dict
from hypnomics.hypnoprints.hp_extractor import get_sg_stage_epoch_dict
from hypnomics.hypnoprints.probes.probe_group import ProbeGroup
from hypnomics.hypnoprints.hp_extractor import STAGE_KEYS

from pictor.objects.signals.signal_group import DigitalSignal
from pictor.objects.signals.signal_group import SignalGroup, Annotation

from roma import console
from roma import finder
from roma import io

import os
import numpy as np
import threading



class Freud(FileManager):

  def __init__(self, work_dir: str):
    super().__init__(work_dir)


  # region: Public Methods

  def generate_clouds(self, sg_path: str, pattern: str, channels: list,
                      time_resolutions: list, extractor_dict: dict=None,
                      overwrite=False, sg_file_list=None, **kwargs):
    """Generate clouds from signal group and save in a hierarchy structure."""
    # (0) Get configurations
    CHANNEL_SHOULD_EXIST = kwargs.get('channel_should_exist', True)
    SKIP_INVALID = kwargs.get('skip_invalid', False)
    PARA_CHANNEL = kwargs.get('parallel_channel', False)
    # SG_PARA_N = kwargs.get('sg_para_number', 1)

    # Create sg_generator, this is for more accuracy progress bar
    def sg_filter(_path=None, _sg_label=None):
      return not self._check_cloud(_path, _sg_label, channels, time_resolutions,
                                   extractor_dict)

    if sg_file_list is None:
      sg_file_list = finder.walk(sg_path, pattern=pattern)
    else:
      console.show_status(f'Before filtering: using provided sg_file_list (N={len(sg_file_list)}).')
    n_all_files = len(sg_file_list)

    if not overwrite:
      sg_file_list = [path for path in sg_file_list if sg_filter(path)]

    # Return sg_file_list if required
    if kwargs.get('return_sg_file_list', False):
      return sg_file_list, n_all_files

    sg_generator = self._get_signal_group_generator(
      sg_path, pattern=pattern, progress_bar=True,
      sg_file_list=sg_file_list, **kwargs)

    from hypnomics.hypnoprints.probes.wavestats.power_probes import _BUFFER

    for sg in sg_generator:
      if not sg_filter(_sg_label=sg.label):
        console.warning(f'Clouds of `{sg.label}` has been created.')

      self._generate_clouds_in_sg(
        sg=sg, channels=channels, time_resolutions=time_resolutions,
        extractor_dict=extractor_dict, overwrite=overwrite,
        SKIP_INVALID=SKIP_INVALID, CHANNEL_SHOULD_EXIST=CHANNEL_SHOULD_EXIST,
        PARA_CHANNEL=PARA_CHANNEL)

      # Clear buffer
      _BUFFER.clear()


  def _generate_clouds_in_sg(self, sg: SignalGroup, channels, time_resolutions,
                             extractor_dict, overwrite, SKIP_INVALID,
                             CHANNEL_SHOULD_EXIST, PARA_CHANNEL):

    # Define a function to generate clouds in a channel
    def generate_clouds_in_channel(channel, progress=True):
      ds: DigitalSignal = sg.digital_signals[0]

      chn_index = ds.channels_names.index(channel)
      # (tr)
      for tr in time_resolutions:
        # Sanity check for time_resolutions
        if 30 % tr != 0: raise NotImplementedError(
          "!! Time resolution should be a factor of 30 !!")

        # This is to save processing time if clouds have already been saved
        segments = None

        # (tr-pk) Extract clouds using specified extractors
        for feature_key, extractor in extractor_dict.items():
          # (1) Check if the cloud already exists
          cloud_path_list, b_exist_list, pk_list = [], [], []
          if isinstance(extractor, ProbeGroup): pk_list = extractor.probe_keys
          else: pk_list = [feature_key]

          for pk in pk_list:
            cloud_path, b_exist = self._check_hierarchy(
              sg.label, channel=channel, time_resolution=tr,
              feature_name=pk, create_if_not_exist=True,
              return_false_if_not_exist=True)
            cloud_path_list.append(cloud_path)
            b_exist_list.append(b_exist)

          if all(b_exist_list) and not overwrite: continue

          if progress: console.print_progress()

          # (2) Initialize segments if necessary
          if segments is None:
            segments = get_sg_stage_epoch_dict(sg, DEFAULT_STAGE_KEY, tr)

          # (3) Generate and save clouds
          if isinstance(extractor, ProbeGroup):
            # (3.1) If extractor is a ProbeGroup
            clouds_dict: dict = extractor.generate_clouds(segments, chn_index)

            for pk, clouds in clouds_dict.items():
              # (3.1.1) Get cloud path
              cloud_path, b_exist = self._check_hierarchy(
                sg.label, channel=channel, time_resolution=tr,
                feature_name=pk, create_if_not_exist=True)

              # (3.1.2) Save clouds
              io.save_file(clouds, cloud_path, verbose=True)

          else:
            # (3.2) If extractor is a callable function
            clouds = OrderedDict()

            do_not_save = False
            for sk in STAGE_KEYS:
              # clouds[sk] = [extractor(s[:, chn_index]) for s in segments[sk]]
              # Sometimes s is float16, causing kurtosis estimation to yield
              #   nan value.
              # (3.2.1) Extract a cloud for each stage
              clouds[sk] = [extractor(s[:, chn_index].astype(np.float32))
                            for s in segments[sk]]

              if any(np.isnan(clouds[sk])) or any(np.isinf(clouds[sk])):
                console.warning(
                  f"!! Invalid value detected in {sg.label}-{channel}-{feature_key} !!")
                if SKIP_INVALID:
                  do_not_save = True
                  continue
                else:
                  # TODO: currently let downstream handle the NaN issue
                  pass
                  # raise ValueError('!! Invalid value detected !!')

            # (3.2.3) Save clouds
            assert len(cloud_path_list) == 1
            cloud_path = cloud_path_list[0]
            if not do_not_save: io.save_file(clouds, cloud_path, verbose=True)

    # Run in threads
    threads = []
    for i, channel in enumerate(channels):
      ds: DigitalSignal = sg.digital_signals[0]

      if not CHANNEL_SHOULD_EXIST and channel not in ds.channels_names:
        console.warning(f'`{channel}` not found!')
        continue

      # Check sg folder first to avoid conflicts
      self._check_hierarchy(sg.label, create_if_not_exist=True)

      # Extract clouds for each channel
      if PARA_CHANNEL:
        t = threading.Thread(target=generate_clouds_in_channel,
                             args=(channel, True))
        threads.append(t)
        t.start()
      else:
        generate_clouds_in_channel(channel)

    # Wait for all threads to finish
    for t in threads: t.join()


  def generate_macro_features(self, sg_path: str = None, pattern: str = None,
                              sg_file_list=None,
                              config: str = 'alpha', overwrite=False, **kwargs):
    """Generate macro features from signal group and save in a hierarchy
    structure.

    Args
    ----
    config: str, optional, default='alpha'.
      configs:
        - 'alpha': (a) Percentage of W/R/N1/N2/N3, totally 5 features
                   (b) Transition probability from W/R/N1/N2/N3 to W/R/N1/N2/N3,
                       totally 25 features
                   (c) Transition per hour, totally 1 feature.
                   Totally 31 features. Ref: sun2019 (a & b)
    epoch_len: int, optional, default=30. For some subsets in MASS dataset,
               epoch_len = 20 s
    """

    assert config == 'alpha', f"!! Unsupported config: {config} !!"

    EPOCH_LEN = kwargs.get('epoch_len', 30)

    # Create a sg_generator
    sg_generator = self._get_signal_group_generator(
      sg_path, pattern=pattern, sg_file_list=sg_file_list,
      progress_bar=True, **kwargs)

    # Extract macro features for each sg file
    for sg in sg_generator:
      # Check path
      cloud_path = self._check_hierarchy(sg.label, create_if_not_exist=True)
      macro_path = os.path.join(cloud_path, f'macro_{config}.od')
      if os.path.exists(macro_path) and not overwrite: continue

      # Calculate macro features
      anno: Annotation = sg.annotations[DEFAULT_STAGE_KEY]
      map_dict = get_stage_map_dict(sg, DEFAULT_STAGE_KEY)
      stages = [map_dict[a] for a in anno.annotations]
      interval_stage = list(zip(anno.intervals, stages))

      # Remove wake stage from both end
      try:
        while interval_stage[0][1] == 0: interval_stage.pop(0)
        while interval_stage[-1][1] == 0: interval_stage.pop(-1)
      except:
        console.warning(f'!! No valid stage found in {sg.label} !!')
        continue

      x_dict = OrderedDict()

      # (1) Calculate percentage of W/N1/N2/N3/R
      # (1.1) Calculate total duration
      T = sum([t[1] - t[0] for t, _ in interval_stage])
      for i, sk in enumerate(STAGE_KEYS):
        duration = sum([t[1] - t[0] for t, s in interval_stage if s == i])
        x_dict[f'{sk}_Percentage'] = duration / T

      # (2) Calculate transition probability from W/N1/N2/N3/R to W/N1/N2/N3/R
      matrix = np.zeros((5, 5), dtype=np.float32)
      # (2.1) Count transition
      transition_count = -1
      for i, (t, si) in enumerate(interval_stage):
        duration = t[1] - t[0]
        assert duration - duration // EPOCH_LEN * EPOCH_LEN < 1e-6

        # Add transition from previous stage to current stage
        if i > 0:
          s0 = interval_stage[i - 1][1]
          if s0 != si: transition_count += 1
          if s0 is not None: matrix[s0, si] += 1

        if si is None: continue

        # Add transition in current stage interval
        matrix[si, si] += duration / EPOCH_LEN - 1

      # (2.2) Calculate transition probability
      matrix = matrix / (np.sum(matrix, axis=1, keepdims=True) + 1e-6)

      for i, sk_i in enumerate(STAGE_KEYS):
        for j, sk_j in enumerate(STAGE_KEYS):
          x_dict[f'Transition_Probability_{sk_i}_to_{sk_j}'] = matrix[i, j]

      # (3) Calculate transition per hour
      x_dict['Transition_per_Hour'] = transition_count / (T / 3600)

      # Save macro feature vector
      io.save_file(x_dict, macro_path, verbose=True)

  # endregion: Public Methods



if __name__ == '__main__':
  pass