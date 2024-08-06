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
# ======-=========================================================-=============
from .wavestats.amp_probes import estimate_amp_envelope_v1
from .wavestats.freq_probes import estimate_average_freq
from .wavestats.freq_probes import estimate_freq_stft_v1

from .wavestats.power_probes import estimate_total_power
from .wavestats.power_probes import estimate_delta_power_rela
from .wavestats.power_probes import estimate_theta_power_rela
from .wavestats.power_probes import estimate_alpha_power_rela
from .wavestats.power_probes import estimate_beta_power_rela



class ProbeLibrary(object):
  amplitude = estimate_amp_envelope_v1
  frequency_st = estimate_average_freq
  frequency_stft = estimate_freq_stft_v1
  total_power = estimate_total_power
  relative_power_delta = estimate_delta_power_rela
  relative_power_theta = estimate_theta_power_rela
  relative_power_alpha = estimate_alpha_power_rela
  relative_power_beta = estimate_beta_power_rela

  extractors = {
    'amplitude': amplitude,
    'frequency_ft': frequency_st,
    'frequency_stft': frequency_stft,
    'total_power': total_power,
    'relative_power_delta': relative_power_delta,
    'relative_power_theta': relative_power_theta,
    'relative_power_alpha': relative_power_alpha,
    'relative_power_beta': relative_power_beta,
  }


pl = ProbeLibrary
