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
import numpy as np
from tensorflow.python.keras.backend import dtype

from .file_manager import FileManager

from collections import OrderedDict

from hypnomics.hypnoprints.hp_extractor import DEFAULT_STAGE_KEY
from hypnomics.hypnoprints.hp_extractor import get_stage_map_dict
from hypnomics.hypnoprints.hp_extractor import STAGE_KEYS
from hypnomics.hypnoprints.hp_extractor import get_sg_stage_epoch_dict

from pictor.objects.signals.signal_group import DigitalSignal
from pictor.objects.signals.signal_group import SignalGroup, Annotation

from roma import console
from roma import io

import os



class Freud(FileManager):

  def __init__(self, work_dir: str):
    super().__init__(work_dir)


  # region: Public Methods

  def generate_clouds(self, sg_path: str, pattern: str, channels: list,
                      time_resolutions: list, extractor_dict: dict=None,
                      overwrite=False, **kwargs):
    """Generate clouds from signal group and save in a hierarchy structure."""
    # (0) Get configurations
    CHANNEL_SHOULD_EXIST = kwargs.get('channel_should_exist', True)
    SKIP_INVALID = kwargs.get('skip_invalid', False)

    # (*)
    sg_generator = self._get_signal_group_generator(
      sg_path, pattern=pattern, progress_bar=True, **kwargs)

    for sg in sg_generator:
      for channel in channels:
        ds: DigitalSignal = sg.digital_signals[0]

        if not CHANNEL_SHOULD_EXIST and channel not in ds.channels_names:
          continue

        chn_index = ds.channels_names.index(channel)
        for tr in time_resolutions:
          # Sanity check for time_resolutions
          if 30 % tr != 0: raise NotImplementedError(
              "!! Time resolution should be a factor of 30 !!")

          # This is to save processing time if clouds have already been saved
          segments = None

          # Extract clouds using specified extractors
          for feature_key, extractor in extractor_dict.items():
            cloud_path, b_exist = self._check_hierarchy(
              sg.label, channel=channel, time_resolution=tr,
              feature_name=feature_key, create_if_not_exist=True)
            if b_exist and not overwrite: continue

            console.print_progress()

            # Initialize segments if necessary
            if segments is None:
              segments = get_sg_stage_epoch_dict(sg, DEFAULT_STAGE_KEY, tr)

            # Generate cloud dict and save
            clouds = OrderedDict()

            do_not_save = False
            for sk in STAGE_KEYS:
              # clouds[sk] = [extractor(s[:, chn_index]) for s in segments[sk]]
              # Sometimes s is float16, causing kurtosis estimation to yield
              #   nan value.
              clouds[sk] = [extractor(s[:, chn_index].astype(np.float32))
                            for s in segments[sk]]

              if any(np.isnan(clouds[sk])) or any(np.isinf(clouds[sk])):
                console.warning(f"!! Invalid value detected in {sg.label}-{channel}-{feature_key} !!")
                if SKIP_INVALID:
                  do_not_save = True
                  continue
                else:
                  # TODO: currently let downstream handle the NaN issue
                  pass
                  # raise ValueError('!! Invalid value detected !!')

            # Save clouds
            if not do_not_save: io.save_file(clouds, cloud_path, verbose=True)


  def generate_macro_features(self, sg_path: str, pattern: str,
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
      sg_path, pattern=pattern, progress_bar=True, **kwargs)

    # Extract macro features for each sg file
    for sg in sg_generator:
      # Check path
      sg_path = self._check_hierarchy(sg.label, create_if_not_exist=True)
      macro_path = os.path.join(sg_path, f'macro_{config}.od')
      if os.path.exists(macro_path) and not overwrite: continue

      # Calculate macro features
      anno: Annotation = sg.annotations[DEFAULT_STAGE_KEY]
      map_dict = get_stage_map_dict(sg, DEFAULT_STAGE_KEY)
      stages = [map_dict[a] for a in anno.annotations]
      interval_stage = list(zip(anno.intervals, stages))

      # Remove wake stage from both end
      while interval_stage[0][1] == 0: interval_stage.pop(0)
      while interval_stage[-1][1] == 0: interval_stage.pop(-1)

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