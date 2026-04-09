// =============================================================================
// Plugin: Slow Wave % — count slow waves per epoch (AASM IV-1.H)
// =============================================================================
// Slow wave: 0.5–2 Hz, peak-to-peak > 75 µV, measured over frontal regions.
// N3 criterion: ≥20% of epoch contains slow wave activity.

registerPlugin({
  name: 'slowwave',
  category: 'Analysis',
  description: 'Slow wave % for current epoch (frontal EEG)',
  channels: ['EEG'],
  execute() {
    // TODO: implement slow wave detection
    alert('Slow Wave % — not yet implemented');
  }
});
