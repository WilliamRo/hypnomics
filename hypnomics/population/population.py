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
"""Hypnomics unifies the analysis of individual and the population."""
from hypnomics.freud.nebula import Nebula
from roma import console, io, Nomear

import numpy as np
import os



class Population(Nomear):

  def __init__(self, work_dir, neb_dir, **kwargs):
    # Set directory paths
    if not os.path.exists(neb_dir):
      raise FileNotFoundError(f'`{neb_dir}` does not exist.')
    self.neb_dir = neb_dir

    if not os.path.exists(work_dir): os.mkdir(work_dir)
    console.show_status(f'Work directory set to `{os.path.abspath(work_dir)}`',
                        prompt='[Population]')

    # Set kwargs
    self.kwargs = kwargs

  # region: Properties

  # endregion: Properties

  # region: Public Methods

  def gather(self, psg_id_list: list, channels: list, time_resolution,
             probe_keys: list, **kwargs):
    """Gather """
    pass

  # endregion: Public Methods
