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
# =-===========================================================================-
from roma import console

import numpy as np



class Stager:
  """Protocol for sleep staging models.

  Label convention: 0=W, 1=N1, 2=N2, 3=N3, 4=REM.
  """

  def predict(self, sg) -> np.ndarray:
    """Predict per-epoch stage labels for a SignalGroup.

    Returns:
      np.ndarray of int labels with shape (n_epochs,).
    """
    raise NotImplementedError

  def predict_proba(self, sg) -> np.ndarray:
    """Predict per-epoch stage probabilities for a SignalGroup.

    Returns:
      np.ndarray of float with shape (n_epochs, 5).
    """
    raise NotImplementedError

  def check_channel(self, sg) -> bool:
    """Check whether the SignalGroup has compatible channels.

    Returns True by default. Override to enforce channel requirements.
    Logs a warning when returning False.
    """
    return True

  def _warn_channel(self, sg, reason=''):
    """Log a channel incompatibility warning."""
    msg = f"Channel check failed for '{sg.label}'"
    if reason:
      msg += f': {reason}'
    console.show_status(msg, prompt='[Warning]')
