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
from hypnomics.hypnoprints.hp_extractor import STAGE_KEYS
from hypnomics.hypnoprints.hp_extractor import get_sg_stage_epoch_dict

from pictor.objects.signals.signal_group import DigitalSignal
from pictor.objects.signals.signal_group import SignalGroup

from roma import console
from roma import io



class Freud(FileManager):

  def __init__(self, work_dir: str):
    super().__init__(work_dir)


  # region: Public Methods

  def generate_clouds(self, sg_path: str, pattern: str, channels: list,
                      time_resolutions: list, extractor_dict: dict=None,
                      overwrite=False, **kwargs):
    """Generate clouds from signal group and save in a hierarchy structure."""
    sg_generator = self._get_signal_group_generator(
      sg_path, pattern=pattern, progress_bar=True, **kwargs)

    for sg in sg_generator:
      for channel in channels:
        ds: DigitalSignal = sg.digital_signals[0]
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
            for sk in STAGE_KEYS:
              clouds[sk] = [extractor(s[:, chn_index]) for s in segments[sk]]

            # Save clouds
            io.save_file(clouds, cloud_path, verbose=True)

  # endregion: Public Methods



if __name__ == '__main__':
  pass