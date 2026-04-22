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
from hypnomics.protocol.stager import Stager
from pictor.xomics.evaluation.confusion_matrix import ConfusionMatrix
from roma import console

import numpy as np


STAGE_LABELS = ['W', 'N1', 'N2', 'N3', 'REM']
NUM_STAGES = len(STAGE_LABELS)



class SleepStagingEvaluator:
  """Model-agnostic evaluator for sleep staging.

  Accepts any Stager instance. Label convention: 0=W, 1=N1, 2=N2, 3=N3, 4=REM.
  """

  def __init__(self, model: Stager, dataset_name='unknown'):
    """Initialize evaluator.

    Args:
      model: a Stager instance.
      dataset_name: name for reporting.
    """
    self.model = model
    self.dataset_name = dataset_name

    # Per-subject results: list of (label, ConfusionMatrix) tuples
    self._results = []
    # Skipped subjects due to channel check
    self._skipped = []

  # region: Core API

  def evaluate(self, psgs, annotation_key='stage Ground-Truth'):
    """Run evaluation on a list of PSG objects.

    Args:
      psgs: list of PSG objects (from `hypnomics.protocol.psg`).
      annotation_key: key to retrieve ground-truth annotations from each PSG.
    """
    self._results = []
    self._skipped = []
    for psg in psgs:
      if not self.model.check_channel(psg):
        self._skipped.append(psg.label)
        continue

      y_true = self._get_ground_truth(psg, annotation_key)
      y_pred = np.asarray(self.model.predict(psg))

      assert len(y_true) == len(y_pred), (
        f"Length mismatch: {len(y_true)} vs {len(y_pred)} "
        f"for subject '{psg.label}'")

      # Drop out-of-range labels (e.g., Unknown=5) from both y_true and the
      # aligned y_pred so the confusion matrix only counts scorable epochs.
      valid = (y_true >= 0) & (y_true < NUM_STAGES)
      y_true, y_pred = y_true[valid], y_pred[valid]

      cm = ConfusionMatrix(NUM_STAGES, class_names=STAGE_LABELS)
      cm.fill(y_pred, y_true)
      self._results.append((psg.label, cm))

    return self

  def report(self):
    """Print aggregated results to console."""
    assert len(self._results) > 0, "No results. Call evaluate() first."

    n = len(self._results)
    cms = [cm for _, cm in self._results]

    if self._skipped:
      console.show_status(
        f'Skipped {len(self._skipped)} subjects (channel check)',
        prompt='[Warning]')

    accs = [cm.accuracy for cm in cms]
    kappas = [cm.cohen_kappa for cm in cms]
    macro_f1s = [cm.macro_F1 for cm in cms]

    console.show_info(f'Dataset: {self.dataset_name} (N={n})')
    console.show_info(f'Accuracy:  {np.mean(accs):.3f} +/- {np.std(accs):.3f}')
    console.show_info(
      f"Cohen's k: {np.mean(kappas):.3f} +/- {np.std(kappas):.3f}")
    console.show_info(
      f'Macro F1:  {np.mean(macro_f1s):.3f} +/- {np.std(macro_f1s):.3f}')

    # Per-stage F1
    console.show_info('Per-stage F1:')
    for i, label in enumerate(STAGE_LABELS):
      f1s = [cm.F1s[i] for cm in cms]
      console.show_info(
        f'  {label:>3s}: {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}')

    # Mean confusion matrix
    mean_cm = np.mean(
      [cm.confusion_matrix for cm in cms], axis=0)
    console.show_info('Confusion Matrix (mean, rows=predicted, cols=true):')
    header = '      ' + ''.join(f'{s:>6s}' for s in STAGE_LABELS)
    console.show_info(header)
    for i, label in enumerate(STAGE_LABELS):
      row = (f'  {label:>3s} '
             + ''.join(f'{mean_cm[i, j]:6.1f}' for j in range(NUM_STAGES)))
      console.show_info(row)

  # endregion: Core API

  # region: Internal

  @staticmethod
  def _get_ground_truth(psg, annotation_key):
    """Extract ground-truth stage labels from a PSG."""
    assert annotation_key in psg.annotation_keys, (
      f"Annotation '{annotation_key}' not found in psg '{psg.label}'. "
      f"Available: {psg.annotation_keys}")
    _intervals, labels, _names = psg.get_annotation(annotation_key)
    assert labels is not None, (
      f"Annotation '{annotation_key}' in psg '{psg.label}' is event-only "
      f"(no stage labels).")
    return np.asarray(labels)

  # endregion: Internal
