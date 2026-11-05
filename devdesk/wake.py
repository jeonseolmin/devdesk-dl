"""Wake word "자비스" rules shared by the app's listener and ml/ (training metric, demo). Standard library only.

A score is the model's logit for one WIN-long window, scored every HOP samples; Trigger decides when that fires.
"""
import math

RATE = 16000
WIN = int(1.5 * RATE)  # one model input: 1.5 s
HOP = RATE // 10  # scoring step: 100 ms
REFRACTORY = 1.0  # s; one utterance fires once


def refractory_hops(hop_s, refractory_s=REFRACTORY):
    """Whole hops of quiet after a trigger, rounded up (oww: 1.0 s / 80 ms → 13); the metric counts the same."""
    return math.ceil(refractory_s / hop_s - 1e-9)


class Trigger:
    """step(score, hops) per scored window → fired? A score >= threshold fires, then nothing for refractory_s; a
    score held above it fires again after each refractory period (ml.wake.count_triggers counts exactly this)."""

    def __init__(self, threshold, hop_s, refractory_s=REFRACTORY):
        self.threshold, self.quiet = threshold, refractory_hops(hop_s, refractory_s)
        self.i, self.next_i = -1, 0

    def step(self, score, hops=1):
        """hops = how many hops of audio arrived since the last call (more than 1 when the scorer fell behind)."""
        self.i += hops
        if score >= self.threshold and self.i >= self.next_i:
            self.next_i = self.i + self.quiet
            return True
        return False
